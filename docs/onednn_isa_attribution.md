# oneDNN ISA attribution: what actually dispatches through oneDNN, and what doesn't

## The method

`src/onednn_attribution.py` runs one OpenVINO CPU inference with
`ONEDNN_VERBOSE=1` set in a child process (oneDNN reads this once at its
own startup, so it has to be set before the interpreter that loads
OpenVINO starts, not from inside this project's own process after
`openvino` is already imported), captures every
`onednn_verbose,v1,primitive,exec,...` line, and attributes each one to
a named model layer. oneDNN's verbose output never names the layer
itself, only the primitive kind, the chosen kernel implementation
string (which encodes the ISA, e.g. `brgemm:avx512_core_bf16`), and a
packed shape field (`mb<batch>ic<in_channels>oc<out_channels>` for
`inner_product`). Since `TinyEncoder`'s four linears
(`wqkv`: 128->384, `fc1`: 128->512, `fc2`: 512->128, `head`: 128->10)
all have distinct `(ic, oc)` pairs, that shape field alone identifies
the layer unambiguously; the two attention matmuls have no fixed shape
to key on and are instead attributed by execution order (scores matmul,
then softmax, then the output matmul, always in that order in this
model's forward pass).

## Finding 1: only the four `nn.Linear` layers dispatch through oneDNN at all

A full trace of this model's forward pass (`docs/onednn_attribution_output.txt`)
contains exactly two oneDNN primitive kinds: `inner_product` (4 calls,
one per linear) and `reorder` (layout conversions around them). There is
no `matmul`, `softmax`, `layer_normalization`, or `eltwise` primitive
anywhere in the trace, for either the FP32 or the INT8 model, confirmed
by grepping the raw verbose output for every distinct primitive name
(`awk -F, '{print $6}' | sort | uniq -c`) rather than trusting this
script's own filter to be exhaustive. OpenVINO's CPU plugin evidently
executes the attention matmuls, the softmax, and all three
`LayerNorm`s through its own native (non-oneDNN) JIT-generated
executors at this model's tiny size (batch 4, seq 16, d_model 128),
reserving oneDNN primitives for the GEMM-shaped linears. This means "each
layer attributed to the oneDNN kernel ISA selected" is honestly only
ever answerable for 4 of the model's roughly 8 distinct layer types; the
other layers are real, they execute, but not through oneDNN, so there is
no oneDNN ISA to attribute them to. The project states this rather than
silently only ever showing the four linears and implying full coverage.

## Finding 2: the "FP32" IR silently runs its linears in bf16

Every `inner_product` primitive in the FP32 model's trace reports
`brgemm:avx512_core_bf16`, not an `f32` kernel, and the `reorder`
primitives immediately before each one convert `f32` activations and
weights to `bf16`. OpenVINO's CPU plugin defaults to running FP32 IR
through automatic bf16 kernels on hardware with AVX-512 BF16 support
(this machine's) unless that behavior is explicitly disabled
(`ov::hint::inference_precision(ov::element::f32)`). The "FP32" baseline
this project's throughput and ISA numbers are measured against is
therefore, honestly, an auto-bf16 baseline on this hardware, not a true
f32 compute path; this is disclosed here rather than assumed away, and
it is also the reason the FP32-to-INT8 throughput gap (Measured results
in the README) is smaller than a true-f32-to-int8 comparison would show,
since bf16 is already most of the way to INT8's throughput advantage on
this CPU.

## Finding 3: INT8's linears select VNNI kernels, as expected

The INT8 model's four `inner_product` primitives all report
`brgemm:avx512_core_vnni`, the DL Boost VNNI-accelerated INT8 GEMM
kernel, which is the expected and correct kernel selection for an
NNCF-quantized OpenVINO IR on this CPU.
