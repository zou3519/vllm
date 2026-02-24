# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import functools
from dataclasses import dataclass, field
from itertools import chain

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._dynamo.source import ConstantSource
from torch._guards import TracingContext, tracing
from torch._higher_order_ops.utils import reenter_make_fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental import proxy_tensor as _proxy_tensor
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_mode,
    get_proxy_slot,
    make_fx,
    track_tensor_tree,
)
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


def _to_symbolic_tensors(flat, shape_env, tracing_ctx):
    """Replace real tensors with symbolic fake tensors for make_fx tracing.

    All tensors marked dynamic share the same symbolic size per axis,
    so a single guard covers all dynamic inputs.
    """
    fake_mode = FakeTensorMode(shape_env=shape_env)
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


def _tensor_key(t):
    return (type(t), tuple(t.size()), tuple(t.stride()), t.dtype, t.device)


def _module_key(mod):
    return tuple(
        (name, _tensor_key(p))
        for name, p in chain(mod.named_parameters(), mod.named_buffers())
    )


def _cache_key(args):
    flat, tree_spec = pytree.tree_flatten(args)
    leaf_keys = []
    for a in flat:
        if isinstance(a, nn.Module):
            leaf_keys.append(_module_key(a))
        elif isinstance(a, torch.Tensor):
            leaf_keys.append(_tensor_key(a))
        else:
            leaf_keys.append(a)
    return (tree_spec, tuple(leaf_keys))


def _default_region_key(args):
    keys = []
    for a in args:
        if isinstance(a, torch.Tensor):
            keys.append((a.dim(), a.dtype, a.device))
        else:
            keys.append(a)
    return tuple(keys)


_trace_region_stats = {"calls": 0, "misses": 0, "hits": 0}


def _inline_cached_graph(cached_gm, args):
    """Copy cached graph nodes into the parent graph being traced."""
    mode = get_proxy_mode()
    tracer = mode.tracer
    graph = tracer.graph

    # Map cached graph's tensor placeholders to parent graph input nodes.
    # Non-tensor placeholders (e.g. booleans) are unused in the graph — their
    # values were inlined as constants during the original trace.
    val_map = {}
    placeholders = [n for n in cached_gm.graph.nodes if n.op == "placeholder"]
    for ph, arg in zip(placeholders, args):
        if isinstance(arg, torch.Tensor):
            val_map[ph] = get_proxy_slot(arg, tracer).proxy.node

    # Copy intermediate nodes into parent graph.
    for node in cached_gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        val_map[node] = graph.node_copy(node, lambda n: val_map[n])

    # Get output spec from cached graph.
    output_node = next(n for n in cached_gm.graph.nodes if n.op == "output")
    out_spec = output_node.args[0]

    # Create a fresh tensor from the cached node metadata and associate it
    # with the copied output proxy.  We use torch.empty rather than reusing the
    # cached tensor directly so each inlined call gets its own proxy slot
    # (important when the same region is called multiple times in one trace).
    # We also avoid torch.empty_like because the cached meta["val"] may be a
    # FakeTensor from a sub-tracer context that is incompatible with the
    # parent's FakeTensorMode.
    def _fresh_tensor(node):
        v = node.meta["val"]
        return torch.empty(v.shape, dtype=v.dtype, device=v.device)

    with disable_proxy_modes_tracing():
        if isinstance(out_spec, torch.fx.Node):
            proxy_out = torch.fx.Proxy(val_map[out_spec], tracer)
            out_tensor = _fresh_tensor(out_spec)
        elif isinstance(out_spec, (tuple, list)):
            proxy_out = type(out_spec)(
                torch.fx.Proxy(val_map[n], tracer) for n in out_spec
            )
            out_tensor = type(out_spec)(_fresh_tensor(n) for n in out_spec)
        else:
            return out_spec

    return track_tensor_tree(out_tensor, proxy_out, constant=None, tracer=tracer)


def trace_region(fn=None, *, cache_key=None):
    def decorator(fn):
        cache = {}

        @functools.wraps(fn)
        def wrapper(*args):
            if _proxy_tensor._CURRENT_MAKE_FX_TRACER is None:
                return fn(*args)

            _trace_region_stats["calls"] += 1

            if cache_key is not None:
                key = cache_key(*args)
            else:
                key = _default_region_key(args)

            if key not in cache:
                _trace_region_stats["misses"] += 1
                cache[key] = reenter_make_fx(fn)(*args)
            else:
                _trace_region_stats["hits"] += 1

            return _inline_cached_graph(cache[key], args)

        wrapper.cache = cache
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def nonstrict_compile(fn, backend=None, cache=True):
    if backend is None:
        from torch._inductor.compile_fx import compile_fx

        backend = compile_fx

    compiled_cache = {} if cache else None
    compiled_once = None
    is_module = isinstance(fn, nn.Module)

    def _trace_and_compile(call_fn, flat, structure):
        functional = _flatten_fn_with_structure(call_fn, structure)
        shape_env = ShapeEnv()
        tracing_ctx = TracingContext(FakeTensorMode(shape_env=shape_env))
        with tracing(tracing_ctx):
            fake_flat = _to_symbolic_tensors(flat, shape_env, tracing_ctx)
            gm = make_fx(functional, tracing_mode="symbolic")(*fake_flat)
            return backend(gm, flat)

    def wrapper(*args):
        nonlocal compiled_once
        # When fn is an nn.Module, prepend it so _flatten_args extracts
        # its params/buffers as explicit graph inputs.
        if is_module:
            call_fn = lambda mod, *a: mod(*a)  # noqa: E731
            all_args = (fn, *args)
        else:
            call_fn = fn
            all_args = args

        flat, structure = _flatten_args(all_args)
        if compiled_cache is not None:
            key = _cache_key(all_args)
            if key not in compiled_cache:
                compiled_cache[key] = _trace_and_compile(call_fn, flat, structure)
            return compiled_cache[key](*flat)
        else:
            if compiled_once is None:
                compiled_once = _trace_and_compile(call_fn, flat, structure)
            return compiled_once(*flat)

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

    # Test 6: cache hit — same shapes reuse compiled artifact
    trace_count = [0]

    def counting_backend(gm, example_inputs):
        trace_count[0] += 1
        return gm

    def add(x, y):
        return x + y

    compiled_add = nonstrict_compile(add, backend=counting_backend)
    compiled_add(torch.randn(3, 4), torch.randn(3, 4))
    compiled_add(torch.randn(3, 4), torch.randn(3, 4))
    assert trace_count[0] == 1, f"Expected 1 trace, got {trace_count[0]}"
    print("PASS: cache hit (same shapes)")

    # Test 7: cache miss on shape change
    trace_count[0] = 0
    compiled_add2 = nonstrict_compile(add, backend=counting_backend)
    compiled_add2(torch.randn(3, 4), torch.randn(3, 4))
    compiled_add2(torch.randn(5, 6), torch.randn(5, 6))
    assert trace_count[0] == 2, f"Expected 2 traces, got {trace_count[0]}"
    print("PASS: cache miss on shape change")

    # Test 8: cache miss on dtype change
    trace_count[0] = 0
    compiled_add3 = nonstrict_compile(add, backend=counting_backend)
    compiled_add3(torch.randn(3, 4), torch.randn(3, 4))
    compiled_add3(
        torch.randn(3, 4, dtype=torch.float64),
        torch.randn(3, 4, dtype=torch.float64),
    )
    assert trace_count[0] == 2, f"Expected 2 traces, got {trace_count[0]}"
    print("PASS: cache miss on dtype change")

    # Test 9: module cache — same input shape reuses, different shape retraces
    trace_count[0] = 0
    mod = nn.Linear(4, 3)
    compiled_mod2 = nonstrict_compile(mod, backend=counting_backend)
    compiled_mod2(torch.randn(2, 4))
    compiled_mod2(torch.randn(2, 4))
    assert trace_count[0] == 1, f"Expected 1 trace, got {trace_count[0]}"
    compiled_mod2(torch.randn(5, 4))
    assert trace_count[0] == 2, f"Expected 2 traces, got {trace_count[0]}"
    print("PASS: module cache")

    # Test 10: trace_region — eager passthrough
    @trace_region
    def attention(q, k, v):
        return q @ k.T @ v

    q, k, v = torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)
    result = attention(q, k, v)
    expected = q @ k.T @ v
    assert torch.allclose(result, expected)
    print("PASS: trace_region eager passthrough")

    # Test 11: trace_region — make_fx correctness
    @trace_region
    def region_add(x, y):
        return x + y

    def outer(x, y):
        return region_add(x, y) * 2

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    gm = make_fx(outer)(x, y)
    result = gm(x, y)
    assert torch.allclose(result, (x + y) * 2)
    print("PASS: trace_region make_fx correctness")

    # Test 12: trace_region — cache hit
    @trace_region
    def cached_fn(x, y):
        return x * y

    def outer2(x, y):
        return cached_fn(x, y) + 1

    gm = make_fx(outer2)(torch.randn(3, 4), torch.randn(3, 4))
    make_fx(outer2)(torch.randn(5, 6), torch.randn(5, 6))
    assert len(cached_fn.cache) == 1, (
        f"Expected 1 cache entry, got {len(cached_fn.cache)}"
    )
    print("PASS: trace_region cache hit")

    # Test 13: trace_region — cache miss on dtype change
    @trace_region
    def typed_fn(x):
        return x + 1

    def outer3(x):
        return typed_fn(x)

    make_fx(outer3)(torch.randn(3, 4))
    make_fx(outer3)(torch.randn(3, 4, dtype=torch.float64))
    assert len(typed_fn.cache) == 2, (
        f"Expected 2 cache entries, got {len(typed_fn.cache)}"
    )
    print("PASS: trace_region cache miss on dtype change")

    # Test 14: trace_region — custom cache key
    @trace_region(cache_key=lambda x, y: "always_same")
    def custom_cached(x, y):
        return x - y

    def outer4(x, y):
        return custom_cached(x, y)

    make_fx(outer4)(torch.randn(3, 4), torch.randn(3, 4))
    make_fx(outer4)(
        torch.randn(5, 6, dtype=torch.float64), torch.randn(5, 6, dtype=torch.float64)
    )
    assert len(custom_cached.cache) == 1, (
        f"Expected 1 cache entry, got {len(custom_cached.cache)}"
    )
    print("PASS: trace_region custom cache key")

    # Test 15: trace_region — multiple calls with boolean branching
    # outer5 calls branching_fn 4 times: 2x True, 2x False.
    # First make_fx: 4 calls, 2 misses (True, False), 2 hits.
    # Second make_fx: 4 calls, 0 misses, 4 hits (both keys already cached).
    _trace_region_stats.update(calls=0, misses=0, hits=0)

    @trace_region
    def branching_fn(x, flag):
        if flag:
            return x * 2
        else:
            return x + 1

    def outer5(x):
        a = branching_fn(x, True)
        b = branching_fn(x, True)
        c = branching_fn(x, False)
        d = branching_fn(x, False)
        return a + b + c + d

    x = torch.randn(3, 4)
    gm = make_fx(outer5)(x)
    assert _trace_region_stats == {"calls": 4, "misses": 2, "hits": 2}, (
        _trace_region_stats
    )
    result = gm(x)
    expected = (x * 2) + (x * 2) + (x + 1) + (x + 1)
    assert torch.allclose(result, expected)
    assert len(branching_fn.cache) == 2, (
        f"Expected 2 cache entries, got {len(branching_fn.cache)}"
    )

    # Second make_fx reuses the cache — all hits, no misses.
    _trace_region_stats.update(calls=0, misses=0, hits=0)
    make_fx(outer5)(torch.randn(3, 4))
    assert _trace_region_stats == {"calls": 4, "misses": 0, "hits": 4}, (
        _trace_region_stats
    )
    print("PASS: trace_region multiple calls with boolean branching")
