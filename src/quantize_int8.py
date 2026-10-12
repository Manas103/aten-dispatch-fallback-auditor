"""NNCF post-training quantization of the FP32 OpenVINO IR
(export_openvino.py's artifacts/model_fp32.xml) into an INT8 IR,
calibrated on 300 held-out-from-training synthetic examples (a third
seed, disjoint from both the training set and the accuracy-gate
validation set, so the number the accuracy gate reports is not measured
on data the quantizer itself saw).
"""
import os

import nncf
import openvino as ov
import torch

from task import make_dataset

ARTIFACT_DIR = "artifacts"
CALIB_SEED = 9999
N_CALIB = 300


def main():
    fp32_model = ov.Core().read_model(f"{ARTIFACT_DIR}/model_fp32.xml")

    calib_ids, _ = make_dataset(N_CALIB, CALIB_SEED)

    def transform_fn(data_item):
        return data_item.unsqueeze(0).numpy()

    calibration_dataset = nncf.Dataset(list(calib_ids), transform_fn)

    int8_model = nncf.quantize(fp32_model, calibration_dataset, subset_size=N_CALIB)

    ov.save_model(int8_model, f"{ARTIFACT_DIR}/model_int8.xml")
    print(f"saved {ARTIFACT_DIR}/model_int8.xml (+ .bin)")


if __name__ == "__main__":
    main()
