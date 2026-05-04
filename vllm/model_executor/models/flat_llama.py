# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Flat Llama model for low-latency optimization.

This module provides a "flat" Llama implementation where each decoder layer
forward is an explicit sequence of low-level op calls, bypassing the
nn.Module dispatch overhead. The forward pass is a simple loop making
fusion opportunities visible and easy to implement.

Usage:
    vllm serve nvidia/Llama-3.3-70B-Instruct-NVFP4 \
        --hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' \
        --compilation-config '{"mode": "none"}'
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    apply_nvfp4_linear,
)
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
# Flat op wrappers — thin, explicit, easy to swap with custom kernels
# ---------------------------------------------------------------------------


def flat_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from vllm.model_executor.layers.layernorm import ir

    return ir.ops.rms_norm(x, weight, eps)


def flat_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ops.fused_add_rms_norm(x, residual, weight, eps)
    return x, residual


def flat_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


def flat_nvfp4_linear(
    layer: nn.Module,
    x: torch.Tensor,
    backend,
) -> torch.Tensor:
    return apply_nvfp4_linear(backend=backend, layer=layer, x=x, bias=None)


# ---------------------------------------------------------------------------
# Per-layer flat forward
# ---------------------------------------------------------------------------


def flat_decoder_layer_forward(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    # Layer sub-modules (same hierarchy as original Llama for weight compat)
    self_attn,  # has .qkv_proj, .o_proj, .rotary_emb, .attn
    mlp,  # has .gate_up_proj, .down_proj
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    # Sizes
    q_size: int,
    kv_size: int,
    nvfp4_backend,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One decoder layer: pre-norm -> attn -> post-norm -> MLP.

    Operations (each is a candidate for fusion):
      1. fused_add_rms_norm
      2. NVFP4 QKV projection
      3. RoPE
      4. Attention (KV cache + flash-attn)
      5. NVFP4 O projection + TP all-reduce
      6. fused_add_rms_norm
      7. NVFP4 gate_up projection
      8. SiLU-and-mul
      9. NVFP4 down projection + TP all-reduce
    """
    eps = input_layernorm.variance_epsilon

    # 1. Pre-attention LayerNorm
    if residual is None:
        residual = hidden_states
        hidden_states = flat_rms_norm(
            hidden_states, input_layernorm.weight.data, eps
        )
    else:
        hidden_states, residual = flat_fused_add_rms_norm(
            hidden_states, residual, input_layernorm.weight.data, eps
        )

    # 2. QKV projection (NVFP4)
    qkv = flat_nvfp4_linear(self_attn.qkv_proj, hidden_states, nvfp4_backend)
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    # 3. Rotary positional embedding
    q, k = self_attn.rotary_emb(positions, q, k)

    # 4. Attention (KV cache update + compute)
    attn_output = self_attn.attn(q, k, v)

    # 5. Output projection (NVFP4) + TP reduce
    hidden_states = flat_nvfp4_linear(
        self_attn.o_proj, attn_output, nvfp4_backend
    )
    if tp_size > 1:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # 6. Post-attention LayerNorm
    hidden_states, residual = flat_fused_add_rms_norm(
        hidden_states,
        residual,
        post_attention_layernorm.weight.data,
        eps,
    )

    # 7. Gate+Up projection (NVFP4, merged)
    gate_up = flat_nvfp4_linear(mlp.gate_up_proj, hidden_states, nvfp4_backend)

    # 8. SwiGLU activation
    hidden_states = flat_silu_and_mul(gate_up)

    # 9. Down projection (NVFP4) + TP reduce
    hidden_states = flat_nvfp4_linear(mlp.down_proj, hidden_states, nvfp4_backend)
    if tp_size > 1:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    return hidden_states, residual


# ---------------------------------------------------------------------------
# Weight-holding layer — same sub-module names as LlamaDecoderLayer so that
# checkpoint weight names match exactly.
# ---------------------------------------------------------------------------


class _FlatLlamaAttention(nn.Module):
    """Holds attention weights with the same attribute names as LlamaAttention."""

    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // num_heads
        )

        self.num_heads = num_heads // tp_size
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )


class _FlatLlamaMLP(nn.Module):
    """Holds MLP weights with the same attribute names as LlamaMLP."""

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )


class FlatLlamaDecoderLayer(nn.Module):
    """Same weight hierarchy as LlamaDecoderLayer, flat forward."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.self_attn = _FlatLlamaAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = _FlatLlamaMLP(
            config, quant_config, prefix=f"{prefix}.mlp"
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self._nvfp4_backend = None

    @property
    def nvfp4_backend(self):
        if self._nvfp4_backend is None:
            qm = getattr(self.self_attn.qkv_proj, "quant_method", None)
            if qm is not None and hasattr(qm, "backend"):
                self._nvfp4_backend = qm.backend
        return self._nvfp4_backend

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return flat_decoder_layer_forward(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            self_attn=self.self_attn,
            mlp=self.mlp,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            q_size=self.self_attn.q_size,
            kv_size=self.self_attn.kv_size,
            nvfp4_backend=self.nvfp4_backend,
            tp_size=self.tp_size,
        )


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------


class FlatLlamaModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size, config.hidden_size, quant_config=quant_config
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: FlatLlamaDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[i](
                positions, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
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
            if (
                "rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name
            ):
                continue

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                loaded_weight = (
                    loaded_weight
                    if loaded_weight.dim() == 0
                    else loaded_weight[0]
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
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
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
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens
                )
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
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
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head."] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(weights)
