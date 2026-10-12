"""Attributes every oneDNN primitive execution during one OpenVINO CPU
inference to the model layer that produced it, from `ONEDNN_VERBOSE=1`
output. oneDNN only tells you what kernel and ISA it chose
(`inner_product,brgemm:avx512_core_bf16,...,mb4ic128oc384,...`), never
which named layer the primitive belongs to; this project closes that gap
itself, by matching the batch/input-channel/output-channel field
(`mb<B>ic<I>oc<O>`) against the known (in_features, out_features) of
each `nn.Linear` in TinyEncoder, since the four linears in one encoder
block all have distinct (ic, oc) pairs here and so are matched
unambiguously; the two attention matmuls have no fixed ic/oc shape and
are instead attributed by execution order (the scores matmul always
precedes softmax, which always precedes the attention-output matmul, in
this model's forward pass).

Runs inference in a child process (`subprocess`, not `os.environ` in
this process) because oneDNN reads `ONEDNN_VERBOSE` once at its own
startup, before this script's imports would have a chance to set it.
"""
import re
import subprocess
import sys

from model import D_MODEL, MLP_HIDDEN

LAYER_BY_IC_OC = {
    (D_MODEL, 3 * D_MODEL): "wqkv (combined QKV projection)",
    (D_MODEL, MLP_HIDDEN): "fc1 (MLP up-projection)",
    (MLP_HIDDEN, D_MODEL): "fc2 (MLP down-projection)",
    (D_MODEL, 10): "head (classification head)",
}

_INFER_SNIPPET = """
import sys
sys.path.insert(0, ".")
import openvino as ov, torch
core = ov.Core()
m = core.compile_model(sys.argv[1], "CPU")
tok = torch.randint(0, 1000, (4, 16)).numpy()
req = m.create_infer_request()
req.infer([tok])
"""


def run_with_verbose(ir_path: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _INFER_SNIPPET, ir_path],
        env={"ONEDNN_VERBOSE": "1", **__import__("os").environ},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout + result.stderr


PRIMITIVE_LINE = re.compile(
    r"^onednn_verbose,v1,primitive,exec,cpu,(?P<prim>[^,]+),(?P<impl>[^,]+),"
    r"(?P<prop>[^,]*),(?P<mds>[^,]*),(?P<attrs>[^,]*),(?P<aux>[^,]*),"
    r"(?P<shape>[^,]*),(?P<time>[^,]*)$"
)
SHAPE_FIELDS = re.compile(r"mb(\d+)ic(\d+)oc(\d+)")


def attribute(verbose_text: str) -> list[dict]:
    rows = []
    matmul_seen = 0
    for line in verbose_text.splitlines():
        m = PRIMITIVE_LINE.match(line)
        if not m:
            continue
        prim = m.group("prim")
        impl = m.group("impl")
        shape = m.group("shape")

        layer = None
        shape_m = SHAPE_FIELDS.search(shape)
        if prim == "inner_product" and shape_m:
            ic, oc = int(shape_m.group(2)), int(shape_m.group(3))
            layer = LAYER_BY_IC_OC.get((ic, oc))
        elif prim == "matmul":
            matmul_seen += 1
            layer = "attn scores (q @ k^T)" if matmul_seen == 1 else "attn output (probs @ v)"
        elif prim == "softmax":
            layer = "attn softmax"
        elif prim in ("layer_normalization", "reduction"):
            layer = "layer norm (ln1/ln2/ln_f, not individually distinguishable)"
        elif prim == "eltwise":
            layer = "gelu / relu activation"
        elif prim == "reorder":
            layer = "(layout conversion, not a model layer)"

        rows.append({
            "primitive": prim,
            "implementation": impl,
            "shape": shape,
            "layer": layer or "(unattributed)",
        })
    return rows


def main():
    for label, ir_path in (("FP32", "artifacts/model_fp32.xml"), ("INT8", "artifacts/model_int8.xml")):
        print(f"=== {label}: {ir_path} ===")
        verbose_text = run_with_verbose(ir_path)
        rows = attribute(verbose_text)
        for r in rows:
            print(f"  {r['layer']:45s} {r['primitive']:18s} {r['implementation']:28s} {r['shape']}")
        print()


if __name__ == "__main__":
    main()
