import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from auditor import DispatchAuditor, is_fallback, op_full_name  # noqa: E402


def test_op_full_name_includes_overload():
    op = torch.ops.aten.add.Tensor
    assert op_full_name(op) == "aten::add.Tensor"


def test_op_full_name_omits_default_overload():
    op = torch.ops.aten.view.default
    assert op_full_name(op) == "aten::view"


def test_is_fallback_true_for_a_known_composite_op():
    # aten::embedding has no direct CPU kernel, only a
    # CompositeExplicitAutograd one (verified directly against the live
    # dispatcher, not hardcoded -- this is the dispatcher's own answer on
    # whatever PyTorch build is running the test).
    assert is_fallback(torch.ops.aten.embedding.default) is True


def test_is_fallback_false_for_a_known_direct_kernel():
    assert is_fallback(torch.ops.aten.add.Tensor) is False


def test_dispatch_auditor_records_real_calls():
    aud = DispatchAuditor()
    with aud:
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        _ = a + b
    ops = [c["op"] for c in aud.calls]
    assert "aten::add.Tensor" in ops
    assert all(c["elapsed_ns"] >= 0 for c in aud.calls)


def test_dispatch_auditor_summary_ranks_by_time():
    aud = DispatchAuditor()
    with aud:
        x = torch.randn(64, 64)
        y = torch.randn(64, 64)
        _ = x @ y
        _ = x + y
    summary = aud.summary()
    times = [r["elapsed_ns"] for r in summary["by_op_ranked_by_time"]]
    assert times == sorted(times, reverse=True)


def test_dispatch_auditor_counts_a_fallback_op():
    aud = DispatchAuditor()
    with aud:
        emb = torch.nn.Embedding(10, 4)
        idx = torch.tensor([1, 2, 3])
        _ = emb(idx)
    summary = aud.summary()
    assert summary["fallback_calls"] >= 1
