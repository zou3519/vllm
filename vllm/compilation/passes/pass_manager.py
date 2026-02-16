# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from torch import fx as fx

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.system_utils import set_env_var

from .vllm_inductor_pass import VllmInductorPass

if rocm_aiter_ops.is_enabled():
    from .fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
        RocmAiterSiluMulFp8GroupQuantFusionPass,
        RocmAiterTritonAddRMSNormPadFusionPass,
    )

if current_platform.is_cuda_alike():
    from .fusion.act_quant_fusion import ActivationQuantFusionPass
    from .fusion.attn_quant_fusion import AttnFusionPass
    from .fusion.qk_norm_rope_fusion import QKNormRoPEFusionPass
    from .fusion.rms_quant_fusion import RMSNormQuantFusionPass
    from .fusion.sequence_parallelism import SequenceParallelismPass
    from .utility.split_coalescing import SplitCoalescingPass

if current_platform.is_cuda():
    from .fusion.allreduce_rms_fusion import AllReduceFusionPass
    from .fusion.collective_fusion import AsyncTPPass

from .inductor_pass import (
    CustomGraphPass,
    InductorPass,
    get_pass_context,
)
from .utility.fix_functionalization import FixFunctionalizationPass
from .utility.noop_elimination import NoOpEliminationPass

logger = init_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def with_pattern_match_debug(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Function decorator that turns on inductor pattern match debug
    for the duration of the call.
    Used to avoid logging builtin Inductor pattern matching.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if (debug_val := envs.VLLM_PATTERN_MATCH_DEBUG) is not None:
            # optionally check rank here
            with set_env_var("TORCHINDUCTOR_PATTERN_MATCH_DEBUG", debug_val):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


class PostGradPassManager(CustomGraphPass):  # type: ignore[misc]
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It supports uuid for the Inductor code cache. That includes torch<2.6
    support using pickling (in .inductor_pass.CustomGraphPass).

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (NoopEliminationPass, FusionPass)
    3. config["post_grad_custom_post_pass"] (if it exists)
    4. fix_functionalization
    This way, all passes operate on a functionalized graph.
    """

    def __init__(self) -> None:
        self.passes: list[InductorPass] = []

    @with_pattern_match_debug
    def __call__(self, graph: fx.Graph) -> None:
        VllmInductorPass.dump_prefix = 0  # reset dump index

        compile_range = get_pass_context().compile_range
        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
                VllmInductorPass.dump_prefix += 1
            else:
                logger.debug("Skipping %s with compile range %s", pass_, compile_range)

        # post-cleanup goes before fix_functionalization
        # because it requires a functional graph
        self.post_cleanup(graph)
        VllmInductorPass.dump_prefix += 1

        # always run fix_functionalization last
        self.fix_functionalization(graph)
        VllmInductorPass.dump_prefix = None  # Cleanup index

    def configure(self, config: VllmConfig) -> None:
        import time as _time

        _t0 = _time.time()
        _pass_times: list[tuple[str, float]] = []
        self.pass_config = config.compilation_config.pass_config

        # Set the current vllm config to allow tracing CustomOp instances
        with set_current_vllm_config(config, check_compile=False):
            if self.pass_config.eliminate_noops:
                _t = _time.time()
                self.passes += [NoOpEliminationPass(config)]
                _pass_times.append(("NoOpEliminationPass", _time.time() - _t))

            if self.pass_config.enable_sp:
                _t = _time.time()
                self.passes += [SequenceParallelismPass(config)]
                _pass_times.append(("SequenceParallelismPass", _time.time() - _t))
                if self.pass_config.fuse_gemm_comms:
                    _t = _time.time()
                    self.passes += [AsyncTPPass(config)]
                    _pass_times.append(("AsyncTPPass", _time.time() - _t))

            if self.pass_config.fuse_allreduce_rms:
                _t = _time.time()
                self.passes += [AllReduceFusionPass(config)]
                _pass_times.append(("AllReduceFusionPass", _time.time() - _t))

            if self.pass_config.fuse_norm_quant:
                _t = _time.time()
                self.passes += [RMSNormQuantFusionPass(config)]
                _pass_times.append(("RMSNormQuantFusionPass", _time.time() - _t))
                if rocm_aiter_ops.is_enabled():
                    self.passes += [
                        RocmAiterRMSNormQuantFusionPass(config),
                    ]
            if self.pass_config.fuse_act_quant:
                _t = _time.time()
                self.passes += [ActivationQuantFusionPass(config)]
                _pass_times.append(("ActivationQuantFusionPass", _time.time() - _t))
                if rocm_aiter_ops.is_enabled():
                    self.passes += [RocmAiterSiluMulFp8GroupQuantFusionPass(config)]

            if self.pass_config.fuse_act_padding and rocm_aiter_ops.is_enabled():
                self.passes += [RocmAiterTritonAddRMSNormPadFusionPass(config)]

            if self.pass_config.fuse_attn_quant:
                _t = _time.time()
                self.passes += [AttnFusionPass(config)]
                _pass_times.append(("AttnFusionPass", _time.time() - _t))

            if self.pass_config.enable_qk_norm_rope_fusion:
                _t = _time.time()
                self.passes += [SplitCoalescingPass(config)]
                _pass_times.append(("SplitCoalescingPass", _time.time() - _t))
                _t = _time.time()
                self.passes += [QKNormRoPEFusionPass(config)]
                _pass_times.append(("QKNormRoPEFusionPass", _time.time() - _t))

            # needs a functional graph
            _t = _time.time()
            self.post_cleanup = PostCleanupPass(config)
            _pass_times.append(("PostCleanupPass", _time.time() - _t))
            _t = _time.time()
            self.fix_functionalization = FixFunctionalizationPass(config)
            _pass_times.append(("FixFunctionalizationPass", _time.time() - _t))

        _t_total = _time.time() - _t0
        with open("/tmp/nonstrict_compile_timing.log", "a") as _f:
            _f.write(
                f"  PassManager.configure: {_t_total:.3f}s "
                f"({len(self.passes)} passes)\n"
            )
            for name, dur in _pass_times:
                _f.write(f"    {name}: {dur:.3f}s\n")

    def add(self, pass_: InductorPass) -> None:
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self) -> str:
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        passes = []

        state: dict[str, Any] = {"pass_config": self.pass_config.compute_hash()}
        for pass_ in self.passes:
            passes.append(pass_.uuid())
        passes.append(self.fix_functionalization.uuid())

        # Include the compile range in the uuid to ensure that inductor
        # recompiles the graph for the new dynamic compile range.
        state["compile_range"] = str(get_pass_context().compile_range)
        state["passes"] = passes
        return InductorPass.hash_dict(state)
