import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from onednn_attribution import attribute  # noqa: E402

FP32_VERBOSE = """onednn_verbose,v1,info,oneDNN v3.10.2 (commit deadbeef)
onednn_verbose,v1,info,cpu,runtime:threadpool,nthr:8
onednn_verbose,v1,primitive,exec,cpu,reorder,jit:uni,undef,src:f32 dst:bf16,,,384x128,0.1
onednn_verbose,v1,primitive,exec,cpu,inner_product,brgemm:avx512_core_bf16,forward_inference,src:bf16 wei:bf16,attr,,mb64ic128oc384,0.6
onednn_verbose,v1,primitive,exec,cpu,inner_product,brgemm:avx512_core_bf16,forward_inference,src:bf16 wei:bf16,attr,,mb64ic128oc512,0.5
onednn_verbose,v1,primitive,exec,cpu,inner_product,brgemm:avx512_core_bf16,forward_inference,src:bf16 wei:bf16,attr,,mb64ic512oc128,0.4
onednn_verbose,v1,primitive,exec,cpu,inner_product,brgemm:avx512_core_bf16,forward_inference,src:bf16 wei:bf16,attr,,mb4ic128oc10,0.1
"""


def test_inner_product_attributed_by_ic_oc_shape():
    rows = attribute(FP32_VERBOSE)
    ip_rows = [r for r in rows if r["primitive"] == "inner_product"]
    assert [r["layer"] for r in ip_rows] == [
        "wqkv (combined QKV projection)",
        "fc1 (MLP up-projection)",
        "fc2 (MLP down-projection)",
        "head (classification head)",
    ]


def test_reorder_is_not_attributed_to_a_model_layer():
    rows = attribute(FP32_VERBOSE)
    reorder_rows = [r for r in rows if r["primitive"] == "reorder"]
    assert len(reorder_rows) == 1
    assert reorder_rows[0]["layer"] == "(layout conversion, not a model layer)"


def test_isa_is_captured_from_the_implementation_field():
    rows = attribute(FP32_VERBOSE)
    assert all(
        r["implementation"] == "brgemm:avx512_core_bf16"
        for r in rows
        if r["primitive"] == "inner_product"
    )


def test_matmul_lines_attributed_in_execution_order():
    text = (
        "onednn_verbose,v1,primitive,exec,cpu,matmul,brgemm:avx512_core,undef,a,b,,4x16x16,0.1\n"
        "onednn_verbose,v1,primitive,exec,cpu,softmax,jit:avx512,undef,a,b,,4x16x16,0.1\n"
        "onednn_verbose,v1,primitive,exec,cpu,matmul,brgemm:avx512_core,undef,a,b,,4x16x128,0.1\n"
    )
    rows = attribute(text)
    assert rows[0]["layer"] == "attn scores (q @ k^T)"
    assert rows[1]["layer"] == "attn softmax"
    assert rows[2]["layer"] == "attn output (probs @ v)"


def test_lines_not_matching_the_primitive_schema_are_ignored():
    rows = attribute("onednn_verbose,v1,info,cpu,runtime:threadpool,nthr:8\ngarbage line\n")
    assert rows == []
