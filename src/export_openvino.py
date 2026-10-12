"""Exports the trained TinyEncoder (train_classifier.py's
artifacts/model_fp32.pt) to an OpenVINO IR model at a fixed batch/seq
shape, via torch.jit.trace -> openvino.convert_model. This FP32 IR is
the baseline both the INT8 leg (quantize_int8.py) and the throughput and
accuracy comparisons measure against.
"""
import os

import openvino as ov
import torch

from model import BATCH, SEQ_LEN, TinyEncoder
from task import N_CLASSES

ARTIFACT_DIR = "artifacts"


def load_trained_model() -> torch.nn.Module:
    model = TinyEncoder(n_classes=N_CLASSES)
    model.load_state_dict(torch.load(f"{ARTIFACT_DIR}/model_fp32.pt"))
    model.eval()
    return model


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model = load_trained_model()
    example = torch.randint(0, 1000, (BATCH, SEQ_LEN))

    traced = torch.jit.trace(model, example)
    ov_model = ov.convert_model(traced, example_input=example, input=([-1, SEQ_LEN], ov.Type.i64))
    ov.save_model(ov_model, f"{ARTIFACT_DIR}/model_fp32.xml")
    print(f"saved {ARTIFACT_DIR}/model_fp32.xml (+ .bin)")


if __name__ == "__main__":
    main()
