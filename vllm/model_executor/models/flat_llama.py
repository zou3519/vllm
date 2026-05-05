# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Flat Llama model for low-latency optimization.

Architecture:
  FlatLlamaForCausalLM  (nn.Module — only for weight loading)
    └─ forward() calls flat_forward() from flat_llama_kernels.py
         └─ loops over transformer_layer() — all params passed explicitly

The nn.Module hierarchy exists only so vLLM's weight loader can populate
the tensors. After loading, all weights are extracted into flat lists and
the forward path is a single function with zero nn.Module dispatch.

Usage:
    vllm serve nvidia/Llama-3.3-70B-Instruct-NVFP4 \
        --hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' \
        --compilation-config '{"mode": "none", "cudagraph_mode": "full"}'
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


# ---------------------------------------------------------------------------
# Weight-holding nn.Modules (for checkpoint loading only)
# ---------------------------------------------------------------------------


class _FlatLlamaAttention(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // num_heads
        )
        self.q_size = (num_heads // tp_size) * head_dim
        self.kv_size = max(1, num_kv_heads // tp_size) * head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size, head_size=head_dim,
            total_num_heads=num_heads, total_num_kv_heads=num_kv_heads,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim, output_size=config.hidden_size,
            bias=False, quant_config=quant_config, reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=True,
        )
        self.attn = Attention(
            num_heads // tp_size, head_dim, head_dim**-0.5,
            num_kv_heads=max(1, num_kv_heads // tp_size),
            cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )


class _FlatLlamaMLP(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False, quant_config=quant_config, reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )


class _FlatLlamaDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.self_attn = _FlatLlamaAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.self_attn",
        )
        self.mlp = _FlatLlamaMLP(config, quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(self, positions, hidden_states, residual):
        raise RuntimeError("Use flat_forward() instead of per-layer forward()")


# ---------------------------------------------------------------------------
# Model backbone — loads weights via nn.Module, runs via flat_forward()
# ---------------------------------------------------------------------------


class FlatLlamaModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        tp_size = get_tensor_model_parallel_world_size()

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size, config.hidden_size, quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: _FlatLlamaDecoderLayer(
                vllm_config=vllm_config, prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size,
            )
        )

        # Computed after weight loading
        self._flat_params_extracted = False
        self._q_size = 0
        self._kv_size = 0
        self._eps = config.rms_norm_eps
        self._tp_size = tp_size
        self._intermediate_size = config.intermediate_size // tp_size

    def _extract_flat_params(self):
        """Pull all weights out of nn.Modules into flat lists."""
        if self._flat_params_extracted:
            return
        from .flat_llama_kernels import extract_all_layer_params

        layer0 = self.layers[self.start_layer]
        self._q_size = layer0.self_attn.q_size
        self._kv_size = layer0.self_attn.kv_size

        (
            self._input_ln_weights,
            self._post_attn_ln_weights,
            self._qkv_projs,
            self._o_projs,
            self._gate_up_projs,
            self._down_projs,
            self._rotary_embs,
            self._attns,
        ) = extract_all_layer_params(
            self.layers, self.start_layer, self.end_layer,
        )

        qm = getattr(layer0.self_attn.qkv_proj, "quant_method", None)
        self._nvfp4_backend = getattr(qm, "backend", None)
        self._final_norm_w = self.norm.weight.data
        self._flat_params_extracted = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        self._extract_flat_params()

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # Always use flat_forward — handles both BS=1 (fused kernels)
        # and general batch sizes (direct C++ ops, no nn.Module dispatch)
        from .flat_llama_kernels import SharedDecodeBuffers, flat_forward

        if not hasattr(self, "_shared_bufs") and self._nvfp4_backend is not None:
            self._shared_bufs = SharedDecodeBuffers.create(
                self.config.hidden_size,
                self._q_size,
                self._kv_size,
                self._intermediate_size,
                hidden_states.device,
            )

        # Prefill buffer cache: mutable list used by flat_forward to
        # lazily create and reuse pre-allocated FP4 buffers for prefill
        if not hasattr(self, "_prefill_bufs_cache"):
            self._prefill_bufs_cache: list = []

        bufs = getattr(self, "_shared_bufs", None)

        hidden_states = flat_forward(
            input_ids=None,
            positions=positions,
            embed_fn=None,
            input_ln_weights=self._input_ln_weights,
            post_attn_ln_weights=self._post_attn_ln_weights,
            eps=self._eps,
            qkv_projs=self._qkv_projs,
            o_projs=self._o_projs,
            gate_up_projs=self._gate_up_projs,
            down_projs=self._down_projs,
            rotary_embs=self._rotary_embs,
            attns=self._attns,
            q_size=self._q_size,
            kv_size=self._kv_size,
            final_norm_w=self._final_norm_w,
            bufs=bufs,
            backend=self._nvfp4_backend,
            start_layer=0,
            end_layer=len(self._qkv_projs),
            hidden_states_in=hidden_states,
            prefill_bufs_cache=self._prefill_bufs_cache,
            hidden_size=self.config.hidden_size,
            intermediate_size=self._intermediate_size,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name or "zero_point" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ---------------------------------------------------------------------------
# Top-level causal LM
# ---------------------------------------------------------------------------


class FlatLlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = FlatLlamaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(config.vocab_size, scale=logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["lm_head."] if self.config.tie_word_embeddings else None,
        )
        return loader.load_weights(weights)
