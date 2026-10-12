import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from accuracy_gate import gate_decision  # noqa: E402


def test_passes_when_drop_within_budget():
    drop_pct, blocked = gate_decision(acc_fp32=0.90, acc_int8=0.895, budget_pct=1.0)
    assert round(drop_pct, 2) == 0.5
    assert blocked is False


def test_blocks_when_drop_exceeds_budget():
    drop_pct, blocked = gate_decision(acc_fp32=0.90, acc_int8=0.87, budget_pct=1.0)
    assert round(drop_pct, 2) == 3.0
    assert blocked is True


def test_just_under_budget_is_not_blocked():
    drop_pct, blocked = gate_decision(acc_fp32=0.90, acc_int8=0.8905, budget_pct=1.0)
    assert round(drop_pct, 2) == 0.95
    assert blocked is False


def test_int8_better_than_fp32_never_blocks():
    drop_pct, blocked = gate_decision(acc_fp32=0.80, acc_int8=0.85, budget_pct=1.0)
    assert drop_pct < 0
    assert blocked is False
