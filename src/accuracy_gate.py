"""The accuracy gate: evaluates top-1 accuracy of the FP32 OpenVINO IR
and the NNCF INT8 IR on the held-out validation set
(train_classifier.py's artifacts/val_set.pt, which the INT8 calibration
set in quantize_int8.py never saw), and fails (exit 1) if the drop
exceeds a 1 percentage-point budget. This is a gate, not a report: the
exit code is the claim, the printed numbers are how to check it by hand.
"""
import sys

import numpy as np
import openvino as ov
import torch

ARTIFACT_DIR = "artifacts"
BUDGET_PCT = 1.0


def gate_decision(acc_fp32: float, acc_int8: float, budget_pct: float = BUDGET_PCT) -> tuple[float, bool]:
    """Pure decision logic, independent of any real measurement, so the
    gate's branch (block vs pass) is tested the same way
    tests/test_ci_check.py tests ci_check.py's branch in the sibling
    ATen auditor: against synthetic accuracy pairs, not this run's
    actual noisy numbers."""
    drop_pct = (acc_fp32 - acc_int8) * 100.0
    return drop_pct, drop_pct > budget_pct


def top1_accuracy(compiled_model, token_ids: torch.Tensor, labels: torch.Tensor) -> float:
    output_port = compiled_model.output(0)
    correct = 0
    for i in range(token_ids.shape[0]):
        inp = token_ids[i : i + 1].numpy()
        logits = compiled_model(inp)[output_port]
        pred = int(np.argmax(logits, axis=-1)[0])
        correct += int(pred == labels[i].item())
    return correct / token_ids.shape[0]


def main():
    val = torch.load(f"{ARTIFACT_DIR}/val_set.pt")
    token_ids, labels = val["token_ids"], val["labels"]

    core = ov.Core()
    fp32 = core.compile_model(f"{ARTIFACT_DIR}/model_fp32.xml", "CPU")
    int8 = core.compile_model(f"{ARTIFACT_DIR}/model_int8.xml", "CPU")

    acc_fp32 = top1_accuracy(fp32, token_ids, labels)
    acc_int8 = top1_accuracy(int8, token_ids, labels)
    drop_pct, blocked = gate_decision(acc_fp32, acc_int8)

    print(f"FP32 top-1 accuracy: {acc_fp32:.4f} ({token_ids.shape[0]} examples)")
    print(f"INT8 top-1 accuracy: {acc_int8:.4f}")
    print(f"drop: {drop_pct:.2f} percentage points (budget: {BUDGET_PCT:.1f})")

    if blocked:
        print(f"BLOCKED: accuracy drop {drop_pct:.2f}pp exceeds the {BUDGET_PCT:.1f}pp budget")
        sys.exit(1)
    print("PASSED: within budget")
    sys.exit(0)


if __name__ == "__main__":
    main()
