"""CI gate: traces one forward+backward step of the model in model.py
and fails the build (exit 1) if any fallback operator ranks in the top
10 operators by cumulative wall time. This is the guardrail a real
dispatch-coverage regression would need: a newly-added op (or a
PyTorch upgrade that drops a backend-specific kernel for an existing
op) silently falling back to a Composite decomposition doesn't crash
anything, it just gets slower -- the only way to catch that before a
user does is to watch the dispatch trace itself.
"""
import sys

from auditor import DispatchAuditor
from model import build_step_inputs, run_forward_backward_step

TOP_N = 10


def run_ci_check(top_n: int = TOP_N) -> tuple[bool, dict]:
    model, x, labels = build_step_inputs()
    auditor = DispatchAuditor()
    with auditor:
        run_forward_backward_step(model, x, labels)

    summary = auditor.summary()
    top = summary["by_op_ranked_by_time"][:top_n]
    offenders = [r for r in top if r["fallback"]]
    passed = len(offenders) == 0
    return passed, {"summary": summary, "top_n": top, "offenders": offenders}


def main():
    passed, detail = run_ci_check()
    summary = detail["summary"]
    print(f"traced {summary['total_calls']} ATen calls, "
          f"{summary['fallback_calls']} fell back "
          f"({summary['fallback_pct_of_time']:.2f}% of step time)")
    print(f"\ntop {TOP_N} operators by cumulative time:")
    for r in detail["top_n"]:
        flag = " FALLBACK" if r["fallback"] else ""
        print(f"  {r['op']:35s} calls={r['calls']:3d} time_ns={r['elapsed_ns']:10d}{flag}")

    if not passed:
        print(f"\nFAIL: {len(detail['offenders'])} fallback operator(s) in the top {TOP_N} by time:")
        for r in detail["offenders"]:
            print(f"  {r['op']}")
        sys.exit(1)

    print(f"\nPASS: no fallback operator in the top {TOP_N} by time")
    sys.exit(0)


if __name__ == "__main__":
    main()
