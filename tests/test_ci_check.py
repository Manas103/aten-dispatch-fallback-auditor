import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import ci_check  # noqa: E402


def _fake_summary(ranked):
    total_ns = sum(r["elapsed_ns"] for r in ranked)
    return {
        "total_calls": len(ranked),
        "total_ns": total_ns,
        "fallback_calls": sum(1 for r in ranked if r["fallback"]),
        "fallback_ns": sum(r["elapsed_ns"] for r in ranked if r["fallback"]),
        "fallback_pct_of_time": 0.0,
        "by_op_ranked_by_time": ranked,
    }


class _FakeAuditor:
    def __init__(self, ranked):
        self._ranked = ranked

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def summary(self):
        return _fake_summary(self._ranked)


def test_ci_check_fails_when_fallback_in_top_n():
    ranked = [
        {"op": "aten::addmm", "calls": 1, "elapsed_ns": 100, "fallback": False},
        {"op": "aten::some_fallback", "calls": 1, "elapsed_ns": 90, "fallback": True},
    ]
    with patch.object(ci_check, "DispatchAuditor", lambda: _FakeAuditor(ranked)), \
         patch.object(ci_check, "build_step_inputs", lambda: (None, None, None)), \
         patch.object(ci_check, "run_forward_backward_step", lambda *a: None):
        passed, detail = ci_check.run_ci_check(top_n=2)
    assert passed is False
    assert [r["op"] for r in detail["offenders"]] == ["aten::some_fallback"]


def test_ci_check_passes_when_fallback_outside_top_n():
    ranked = [
        {"op": "aten::addmm", "calls": 1, "elapsed_ns": 100, "fallback": False},
        {"op": "aten::mm", "calls": 1, "elapsed_ns": 90, "fallback": False},
        {"op": "aten::some_fallback", "calls": 1, "elapsed_ns": 1, "fallback": True},
    ]
    with patch.object(ci_check, "DispatchAuditor", lambda: _FakeAuditor(ranked)), \
         patch.object(ci_check, "build_step_inputs", lambda: (None, None, None)), \
         patch.object(ci_check, "run_forward_backward_step", lambda *a: None):
        passed, detail = ci_check.run_ci_check(top_n=2)
    assert passed is True
    assert detail["offenders"] == []


def test_ci_check_passes_with_no_fallbacks_at_all():
    ranked = [{"op": "aten::addmm", "calls": 1, "elapsed_ns": 100, "fallback": False}]
    with patch.object(ci_check, "DispatchAuditor", lambda: _FakeAuditor(ranked)), \
         patch.object(ci_check, "build_step_inputs", lambda: (None, None, None)), \
         patch.object(ci_check, "run_forward_backward_step", lambda *a: None):
        passed, _ = ci_check.run_ci_check(top_n=10)
    assert passed is True
