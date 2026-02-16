# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch._dynamo.source import ConstantSource
from torch._dynamo.utils import dynamo_timed
from torch._guards import TracingContext, tracing
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatefulSymbolicContext,
)
from torch.nn.utils.stateless import _reparametrize_module


@dataclass
class _ModuleEntry:
    module: nn.Module
    names: list[str]
    flat_start: int
    flat_end: int = 0


@dataclass
class _ArgStructure:
    """Maps original args to positions in the flattened tensor list.

    Each entry in arg_map is either:
      ("module", entry_index)  — reconstructed via _reparametrize_module
      ("value", flat_index)    — passed through directly
    """

    modules: list[_ModuleEntry] = field(default_factory=list)
    arg_map: list[tuple[str, int]] = field(default_factory=list)


def _flatten_args(args):
    """Replace each nn.Module in args with its param/buffer tensors spliced inline."""
    flat = []
    structure = _ArgStructure()

    for a in args:
        if isinstance(a, nn.Module):
            names = [n for n, _ in a.named_parameters()] + [
                n for n, _ in a.named_buffers()
            ]
            tensors = [p for _, p in a.named_parameters()] + [
                b for _, b in a.named_buffers()
            ]
            entry = _ModuleEntry(module=a, names=names, flat_start=len(flat))
            flat.extend(tensors)
            entry.flat_end = len(flat)
            structure.arg_map.append(("module", len(structure.modules)))
            structure.modules.append(entry)
        else:
            structure.arg_map.append(("value", len(flat)))
            flat.append(a)

    return flat, structure


def _flatten_fn_with_structure(fn, structure):
    """Wrap fn to accept a flat tensor list, reconstructing module args via reparametrization."""

    def functional(*flat):
        with contextlib.ExitStack() as stack:
            args = []
            for kind, idx in structure.arg_map:
                if kind == "module":
                    entry = structure.modules[idx]
                    state = dict(
                        zip(entry.names, flat[entry.flat_start : entry.flat_end])
                    )
                    stack.enter_context(_reparametrize_module(entry.module, state))
                    args.append(entry.module)
                else:
                    args.append(flat[idx])
            return fn(*args)

    return functional


def _to_symbolic_tensors(flat, shape_env, fake_mode, tracing_ctx):
    """Replace real tensors with symbolic fake tensors for make_fx tracing.

    All tensors marked dynamic share the same symbolic size per axis,
    so a single guard covers all dynamic inputs.
    """
    shared_sym: dict[int, torch.SymInt] = {}
    out = []
    for i, val in enumerate(flat):
        if not isinstance(val, torch.Tensor):
            out.append(val)
            continue
        src = ConstantSource(f"arg_{i}")
        dynamic = getattr(val, "_dynamo_dynamic_indices", set())
        ctx = StatefulSymbolicContext(
            dynamic_sizes=[
                DimDynamic.DYNAMIC if d in dynamic else DimDynamic.STATIC
                for d in range(val.dim())
            ],
            tensor_source=src,
        )
        tracing_ctx.tensor_to_context[val] = ctx
        fake = fake_mode.from_tensor(val, symbolic_context=ctx, source=src)
        # Unify: reuse the symbol from the first tensor that introduced
        # each dynamic axis so all dynamic inputs share one symbol.
        if dynamic:
            shape = list(fake.shape)
            changed = False
            for d in dynamic:
                if d in shared_sym:
                    shape[d] = shared_sym[d]
                    changed = True
                else:
                    shared_sym[d] = shape[d]
            if changed:
                fake = fake.view(shape)
        out.append(fake)
    return out


def nonstrict_compile(fn, backend=None):
    if backend is None:
        from torch._inductor.compile_fx import compile_fx

        backend = compile_fx

    compiled = None
    is_module = isinstance(fn, nn.Module)

    def wrapper(*args):
        nonlocal compiled
        # When fn is an nn.Module, prepend it so _flatten_args extracts
        # its params/buffers as explicit graph inputs.
        if is_module:
            call_fn = lambda mod, *a: mod.forward(*a)  # noqa: E731
            all_args = (fn, *args)
        else:
            call_fn = fn
            all_args = args

        t0 = time.perf_counter()
        with dynamo_timed("nonstrict_compile.flatten_args"):
            flat, structure = _flatten_args(all_args)
        t_flatten = time.perf_counter() - t0

        if compiled is None:
            with dynamo_timed("nonstrict_compile"):
                t0 = time.perf_counter()
                with dynamo_timed("nonstrict_compile.setup"):
                    functional = _flatten_fn_with_structure(call_fn, structure)
                    shape_env = ShapeEnv()
                    fake_mode = FakeTensorMode(
                        shape_env=shape_env, allow_non_fake_inputs=True
                    )
                    tracing_ctx = TracingContext(fake_mode)
                t_setup = time.perf_counter() - t0

                # Suppress dynamo's nested FX trace error — during
                # make_fx tracing, any dynamo-decorated function
                # encountered should just be called directly.
                old_error_on_nested = torch._dynamo.config.error_on_nested_fx_trace
                torch._dynamo.config.error_on_nested_fx_trace = False
                try:
                    with tracing(tracing_ctx):
                        t0 = time.perf_counter()
                        with dynamo_timed("nonstrict_compile.fakify"):
                            fake_flat = _to_symbolic_tensors(
                                flat, shape_env, fake_mode, tracing_ctx
                            )
                        t_fakify = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        with dynamo_timed("nonstrict_compile.make_fx"):
                            gm = make_fx(functional, tracing_mode="symbolic")(
                                *fake_flat
                            )
                        t_make_fx = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        with dynamo_timed("nonstrict_compile.fix_metadata"):
                            # make_fx populates node.meta["val"] but
                            # some backends (e.g. vLLM's VllmBackend)
                            # expect "example_value". Copy for compat.
                            for node in gm.graph.nodes:
                                if (
                                    "val" in node.meta
                                    and "example_value" not in node.meta
                                ):
                                    node.meta["example_value"] = node.meta["val"]
                        t_metadata = time.perf_counter() - t0

                        # Keep fake_mode active via tracing context so
                        # detect_fake_mode() works in the backend.
                        t0 = time.perf_counter()
                        with dynamo_timed("nonstrict_compile.backend"):
                            compiled = backend(gm, flat)
                        t_backend = time.perf_counter() - t0
                finally:
                    torch._dynamo.config.error_on_nested_fx_trace = old_error_on_nested

            n_tensors = sum(1 for x in flat if isinstance(x, torch.Tensor))
            t_total = (
                t_flatten + t_setup + t_fakify + t_make_fx + t_metadata + t_backend
            )
            msg = (
                f"nonstrict_compile breakdown "
                f"(total={t_total:.3f}s):\n"
                f"  flatten_args: {t_flatten:.3f}s"
                f" ({len(flat)} flat args, {n_tensors} tensors)\n"
                f"  setup (ShapeEnv+FakeTensorMode):"
                f" {t_setup:.3f}s\n"
                f"  fakify (_to_symbolic_tensors):"
                f" {t_fakify:.3f}s\n"
                f"  make_fx (trace): {t_make_fx:.3f}s\n"
                f"  fix_metadata: {t_metadata:.3f}s\n"
                f"  backend: {t_backend:.3f}s\n"
            )
            with open("/tmp/nonstrict_compile_timing.log", "a") as f:
                f.write(msg)
        return compiled(*flat)

    return wrapper


if __name__ == "__main__":
    # Test 1: simple function
    def f(x, y):
        return x + y

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    result = nonstrict_compile(f)(x, y)
    assert torch.allclose(result, f(x, y))
    print("PASS: simple function")

    # Test 2: nn.Module
    mod = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    result = nonstrict_compile(mod)(x)
    assert torch.allclose(result, mod(x))
    print("PASS: nn.Module")

    # Test 3: dynamic shapes — the compiled graph handles varying batch sizes
    def g(x, y):
        return x.sum(0) + y.sum(0)

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.mark_dynamic(y, 0)
    compiled_g = nonstrict_compile(g)
    r1 = compiled_g(x, y)
    assert torch.allclose(r1, g(x, y))

    x2, y2 = torch.randn(5, 4), torch.randn(5, 4)
    r2 = compiled_g(x2, y2)
    assert torch.allclose(r2, g(x2, y2))
    print("PASS: dynamic shapes")

    # Test 4: function taking two nn.Modules
    def h(mod1, mod2, x):
        return mod1(x) + mod2(x)

    m1 = nn.Linear(4, 3)
    m2 = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    result = nonstrict_compile(h)(m1, m2, x)
    assert torch.allclose(result, h(m1, m2, x))
    print("PASS: function with two nn.Modules")

    # Test 5: nn.Module parameter mutation via .copy_()
    mod = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    compiled_mod = nonstrict_compile(mod)
    r1 = compiled_mod(x)
    assert torch.allclose(r1, mod(x))

    with torch.no_grad():
        mod.weight.copy_(torch.randn(3, 4))
        mod.bias.copy_(torch.randn(3))
    r2 = compiled_mod(x)
    assert torch.allclose(r2, mod(x))
    print("PASS: nn.Module parameter mutation")
