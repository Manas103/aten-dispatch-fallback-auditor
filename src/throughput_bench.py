"""Measures single-stream CPU inference throughput for the FP32 and
NNCF INT8 OpenVINO IRs on the same batch-4 input, reports the speedup
ratio. Single-threaded by design (`ov::inference_num_threads(1)`): this
machine is shared and in active use, and a multi-threaded throughput
number would mix "INT8 is a faster kernel" with "this run grabbed more
cores", which is not the quantity the claim is about.
"""
import statistics
import time

import numpy as np
import openvino as ov
import torch

ARTIFACT_DIR = "artifacts"
N_WARMUP = 20
N_TIMED = 200


def bench(ir_path: str) -> list[float]:
    core = ov.Core()
    core.set_property("CPU", {"INFERENCE_NUM_THREADS": 1})
    model = core.compile_model(ir_path, "CPU")
    tok = torch.randint(0, 1000, (4, 16)).numpy()
    req = model.create_infer_request()

    for _ in range(N_WARMUP):
        req.infer([tok])

    latencies_s = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        req.infer([tok])
        latencies_s.append(time.perf_counter() - t0)
    return latencies_s


def main():
    fp32_lat = bench(f"{ARTIFACT_DIR}/model_fp32.xml")
    int8_lat = bench(f"{ARTIFACT_DIR}/model_int8.xml")

    fp32_med = statistics.median(fp32_lat)
    int8_med = statistics.median(int8_lat)
    fp32_throughput = 1.0 / fp32_med
    int8_throughput = 1.0 / int8_med
    speedup = fp32_med / int8_med

    print(f"FP32: median latency {fp32_med * 1e6:.1f}us, {fp32_throughput:.1f} infers/sec")
    print(f"INT8: median latency {int8_med * 1e6:.1f}us, {int8_throughput:.1f} infers/sec")
    print(f"speedup (INT8 throughput / FP32 throughput): {speedup:.2f}x")
    print(f"FP32 p10-p90 latency: {np.percentile(fp32_lat, 10)*1e6:.1f}-{np.percentile(fp32_lat, 90)*1e6:.1f}us")
    print(f"INT8 p10-p90 latency: {np.percentile(int8_lat, 10)*1e6:.1f}-{np.percentile(int8_lat, 90)*1e6:.1f}us")


if __name__ == "__main__":
    main()
