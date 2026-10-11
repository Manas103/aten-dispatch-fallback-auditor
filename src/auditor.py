"""Traces every ATen operator call in a forward+backward step through a
TorchDispatchMode shim, attributes wall time per call, and classifies
each call as either a dedicated per-backend kernel or a fallback onto a
Composite (decomposed) implementation.

Fallback classification: PyTorch's dispatcher resolves `aten::foo` for a
given backend (here, CPU) by checking, in order, whether that backend
has a directly registered kernel. If it doesn't, dispatch falls through
to a CompositeImplicitAutograd or CompositeExplicitAutograd kernel, which
is implemented *in terms of other aten ops* -- so a "fallback" call
doesn't do its own work, it decomposes into further dispatcher calls,
which this same TorchDispatchMode also captures. `torch._C._dispatch_has_
kernel_for_dispatch_key` is the real, internal PyTorch API this project
uses to ask the dispatcher that question directly, rather than guessing
from an op's name.
"""
import time
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_KERNEL_KEY_CACHE: dict[str, bool] = {}


def op_full_name(func) -> str:
    """The overload-qualified schema name the dispatcher introspection API
    actually needs, e.g. 'aten::add.Tensor', not 'aten::add' (which is not
    itself a registered name when an op has more than one overload) and
    not str(func)'s dot-separated display form ('aten.add.Tensor')."""
    if not hasattr(func, "_schema"):
        return str(func)
    base = func._schema.name
    overload = getattr(func, "_overloadname", "")
    return base if overload in ("", "default") else f"{base}.{overload}"


def is_fallback(func) -> bool:
    """True if `func` has no directly registered CPU kernel and resolves
    through a Composite (decomposing) kernel instead."""
    name = op_full_name(func)
    if name in _KERNEL_KEY_CACHE:
        return _KERNEL_KEY_CACHE[name]
    try:
        has_cpu = torch._C._dispatch_has_kernel_for_dispatch_key(name, "CPU")
        has_composite = torch._C._dispatch_has_kernel_for_dispatch_key(
            name, "CompositeImplicitAutograd"
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(
            name, "CompositeExplicitAutograd"
        )
        result = (not has_cpu) and has_composite
    except RuntimeError:
        # Some internal/alias ops aren't registered under a plain schema
        # name the dispatcher introspection API recognizes; treated as
        # "not a fallback" rather than raising, since they are not real
        # operator-dispatch-selection events.
        result = False
    _KERNEL_KEY_CACHE[name] = result
    return result


class DispatchAuditor(TorchDispatchMode):
    """Records (op name, dtype, wall time, fallback?) for every ATen call
    made while this mode is active."""

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        t0 = time.perf_counter_ns()
        result = func(*args, **kwargs)
        t1 = time.perf_counter_ns()

        dtype = None
        for a in args:
            if isinstance(a, torch.Tensor):
                dtype = str(a.dtype)
                break

        self.calls.append({
            "op": op_full_name(func),
            "dtype": dtype,
            "elapsed_ns": t1 - t0,
            "fallback": is_fallback(func),
        })
        return result

    def summary(self) -> dict:
        total_calls = len(self.calls)
        total_ns = sum(c["elapsed_ns"] for c in self.calls)
        fallback_calls = [c for c in self.calls if c["fallback"]]
        fallback_ns = sum(c["elapsed_ns"] for c in fallback_calls)

        by_op = defaultdict(lambda: {"calls": 0, "elapsed_ns": 0, "fallback": False})
        for c in self.calls:
            e = by_op[c["op"]]
            e["calls"] += 1
            e["elapsed_ns"] += c["elapsed_ns"]
            e["fallback"] = c["fallback"]

        ranked = sorted(
            ({"op": op, **stats} for op, stats in by_op.items()),
            key=lambda x: x["elapsed_ns"],
            reverse=True,
        )

        return {
            "total_calls": total_calls,
            "total_ns": total_ns,
            "fallback_calls": len(fallback_calls),
            "fallback_ns": fallback_ns,
            "fallback_pct_of_time": (fallback_ns / total_ns * 100.0) if total_ns else 0.0,
            "by_op_ranked_by_time": ranked,
        }
