# ATen Dispatch and Fallback Auditor for a PyTorch Model Step, Extended with an OpenVINO INT8 Leg

Traces every ATen operator call in one real forward-and-backward step of
a small transformer encoder through a `TorchDispatchMode` shim, attributes
wall time per operator, and classifies each call as either a dedicated
backend kernel or a fallback onto a Composite (decomposing) kernel, using
PyTorch's own dispatcher introspection API to ask the question directly
rather than guessing from an op's name. Extended with a second leg that
takes the same model architecture, actually trains it, exports it to an
OpenVINO IR, NNCF-quantizes it to INT8, gates the result on a top-1
accuracy budget, benchmarks FP32 against INT8 throughput, and attributes
every oneDNN primitive the OpenVINO CPU plugin executes to the kernel
ISA it selected. Every number below is measured on this machine (the
ATen leg on PyTorch 2.12 CPU, the OpenVINO leg on OpenVINO 2026.2 and
NNCF 2.19), not estimated or assumed from documentation.

**Honest framing, up front:**
- **CPU only, by design, not as a limitation being worked around.** Both
  legs audit mechanisms (dispatch fallback; INT8 kernel selection) that
  are visible identically on CPU, and no AMD or NVIDIA GPU is needed,
  used, or claimed anywhere here.
- **ATen leg, measured: 133 ATen calls traced, 59 fell back, consuming
  roughly 8-20% of step time across repeated runs** (noisy at this
  model's tiny size; see Measured results). Not the same numbers as
  this project's own original target (112 / 14 / 38%) -- reported as
  measured, not adjusted to match.
- **A real, non-obvious finding drove the final model design**: two of
  this project's three originally-planned fallback ops
  (`aten::linear`, `aten::layer_norm`) never once appear in any trace on
  this PyTorch build, because their Python-level call sites bypass the
  dispatcher-level composite op entirely. See Findings and
  `docs/op_classification.md`.
- **OpenVINO leg, measured: the same architecture, actually trained to
  98.6% top-1 on a synthetic task, quantized to INT8, passes the 1%
  accuracy-budget gate (0.10 percentage point drop measured) and
  measures 1.2x-1.6x throughput across every configuration tried, not
  the 2.6x this leg's own original target.** The gate mechanism blocks
  correctly when tested against synthetic accuracy pairs
  (`tests/test_accuracy_gate.py`); it does not fire on this model, and
  that is reported honestly rather than forced. See Findings and
  `docs/onednn_isa_attribution.md`.
- **A second real, non-obvious finding**: only 4 of this model's roughly
  8 layer types (the four `nn.Linear`s) ever dispatch through a oneDNN
  primitive at all; the attention matmuls, softmax, and all three
  `LayerNorm`s run through OpenVINO's own native CPU executors, invisible
  to `ONEDNN_VERBOSE`. A third finding: the "FP32" IR silently runs its
  linears through automatic `bf16` kernels on this AVX-512 BF16-capable
  CPU, not true `f32` kernels, which is also most of the reason INT8's
  throughput edge over "FP32" here is smaller than a true f32-to-int8
  comparison would show.

## Why this exists

A model can run correctly on a GPU backend and still be slow for a
reason no benchmark isolates: some operator silently fell off the
backend's optimized kernel path onto a generic, decomposing
implementation, and nothing about the program's output changes, only its
speed. The real, AMD-specific version of this problem is a model running
on an AMD GPU where some ATen op has no ROCm-backed kernel and falls back
to a generic path; this project builds the auditing mechanism for that
problem and demonstrates it on CPU, where the same dispatcher mechanism
is visible without needing the hardware the mechanism is really about.

A second, related question a dispatch trace alone can't answer: once a
model is quantized for a CPU inference runtime, which specific kernel
and instruction set actually ran for each layer, and did the cheaper
numeric format cost any accuracy worth gating on? The OpenVINO leg
(`src/export_openvino.py` onward) puts the same "ask the dispatcher, not
the name" discipline to that question on Intel's own CPU inference
stack: oneDNN's verbose log is asked directly which kernel and ISA it
picked, rather than assuming AVX-512 or INT8-VNNI from the IR's declared
precision alone.

## Architecture

```
src/model.py      TinyEncoder: token+positional embedding, one encoder
                  block (combined-QKV self-attention + GELU MLP), a
                  classification head -- a small, real forward+backward
                  step, not a synthetic op-call generator
src/auditor.py    DispatchAuditor(TorchDispatchMode): records (op, dtype,
                  elapsed_ns, fallback?) for every ATen call; is_fallback()
                  queries torch._C._dispatch_has_kernel_for_dispatch_key
                  directly, never a hardcoded op list
src/ci_check.py   runs one step, ranks operators by cumulative time, fails
                  (exit 1) if any fallback op is in the top 10

tests/test_auditor.py    unit tests against the live dispatcher (not
                          hardcoded expectations) plus a real-tensor
                          smoke test of the auditor itself
tests/test_ci_check.py   the CI gate's pass/fail logic, exercised against
                          synthetic ranked-operator lists so its branches
                          are tested independent of this run's actual
                          timing noise

src/task.py              the synthetic first-token-mod-10 classification
                          task (see Findings for why a sum-mod-10 task
                          was tried first and discarded)
src/train_classifier.py  actually trains TinyEncoder to a real accuracy
                          on the task above; saves artifacts/model_fp32.pt
                          and artifacts/val_set.pt
src/export_openvino.py   torch.jit.trace -> openvino.convert_model, fixed
                          seq_len, dynamic batch; saves artifacts/model_fp32.xml
src/quantize_int8.py     NNCF post-training quantization of the FP32 IR,
                          calibrated on a third, disjoint seed; saves
                          artifacts/model_int8.xml
src/accuracy_gate.py     FP32 vs INT8 top-1 accuracy on the held-out val
                          set; `gate_decision()` is the pure, testable
                          block/pass logic, blocks (exit 1) past a 1pp budget
src/throughput_bench.py  FP32 vs INT8 CPU inference throughput, single
                          thread, median over 200 timed calls after warmup
src/onednn_attribution.py   runs inference with ONEDNN_VERBOSE=1 in a
                          child process, attributes every oneDNN primitive
                          to a named model layer by shape (linears) or
                          execution order (attention matmuls)

tests/test_task.py             dataset determinism and the label rule
tests/test_accuracy_gate.py    gate_decision()'s block/pass branches against
                                synthetic accuracy pairs, independent of this
                                run's actual measured accuracy
tests/test_onednn_attribution.py   attribute()'s shape- and order-based
                                matching against synthetic verbose-log text,
                                independent of this run's actual CPU and ISA

docs/op_classification.md        the finding that reshaped this model's design
docs/onednn_isa_attribution.md   what oneDNN does and doesn't see, and the
                                  silent FP32-runs-as-bf16 finding
docs/ci_check_output.txt         raw CI check output backing every ATen number
docs/test_output.txt             raw pytest output (22 tests, both legs)
docs/openvino_train_output.txt   raw training log (per-epoch loss and val accuracy)
docs/openvino_accuracy_gate_output.txt   raw accuracy-gate output and exit code
docs/openvino_throughput_output.txt      raw throughput-benchmark output (canonical run)
docs/openvino_throughput_attempts.txt    all 3 genuine throughput-measurement attempts
docs/onednn_attribution_output.txt       raw per-layer oneDNN ISA attribution, FP32 and INT8
```

### Why `is_fallback` needs the overload-qualified name

`torch._C._dispatch_has_kernel_for_dispatch_key` expects
`"aten::add.Tensor"`, not `"aten::add"` (not a valid standalone name once
an op has more than one overload) and not `str(func)`'s dot-separated
display form `"aten.add.Tensor"`. `auditor.op_full_name` builds this from
`func._schema.name` plus `func._overloadname`, dropping the suffix only
for the (implicit) `default` overload. Getting this wrong doesn't raise
an error, it just silently returns `False` for every query -- the single
bug that produced this project's very first (wrong) measurement; see
Findings.

## Validation

- **Unit tests against the live dispatcher, not a hardcoded answer key**
  (`tests/test_auditor.py`): `is_fallback` is checked against
  `aten::embedding` (true) and `aten::add.Tensor` (false) by calling the
  same dispatcher API the auditor itself calls, so the test tracks
  whatever PyTorch build actually runs it rather than a frozen
  assumption.
- **The CI gate's branch logic is tested independent of real timing
  noise** (`tests/test_ci_check.py`): synthetic ranked-operator lists
  with a fallback forced into or out of the top N, so "does the gate
  fail when it should" is proven without depending on this run's actual
  microsecond-scale measurements.
- **10 pytest checks total for the ATen leg**; **22 total across both
  legs** (`docs/test_output.txt`).
- **The OpenVINO accuracy gate's own branch logic is tested independent
  of this run's real, noisy accuracy numbers**
  (`tests/test_accuracy_gate.py`), the same discipline
  `tests/test_ci_check.py` already applies to the ATen leg's gate:
  synthetic FP32/INT8 accuracy pairs that should and shouldn't cross a
  1pp budget, checked against the pure `gate_decision()` function rather
  than a live OpenVINO run.
- **The oneDNN attribution's shape- and order-matching logic is tested
  against synthetic verbose-log text** (`tests/test_onednn_attribution.py`),
  not this run's actual CPU and ISA, so the matching rule itself is
  checked independent of what chip happens to run the test.
- **The dataset is checked for determinism and for the label rule it
  claims** (`tests/test_task.py`): same seed gives identical data,
  different seeds don't, and labels are exactly `token_ids[:, 0] % 10`.

## Findings

**The first real measurement was zero fallbacks, and the bug was a
string-format mismatch, not a logic error.** `is_fallback`'s first
version queried the dispatcher with `str(func)` (PyTorch's dot-separated
display form, e.g. `"aten.add.Tensor"`). `_dispatch_has_kernel_for_
dispatch_key` doesn't recognize that format; it raised internally, which
the original code caught and silently treated as "not a fallback" --
every single query returned `False`, so the first full trace (366 calls)
reported 0 fallbacks. The fix was building the proper
`"aten::op.overload"` schema name from `func._schema.name` and
`func._overloadname` directly (see Architecture). Re-run after the fix,
the same trace reported 180 fallback calls.

**Two of the three operators this project originally planned to exercise
never dispatch at all on this PyTorch build.** `aten::linear` and
`aten::layer_norm` are both confirmed, directly against the dispatcher,
to have no `CPU` kernel and only a `CompositeImplicitAutograd` one --
textbook fallbacks. Neither ever appears in any trace of any model built
here, because `torch.nn.functional.linear`/`layer_norm`'s Python-level
implementations call the backend-specific primitive (`addmm`,
`native_layer_norm`) directly, without the dispatcher ever seeing a call
to the generic op first. "An op is Composite" and "an op will show up as
a fallback in a real trace" are not the same claim; closing that gap is
what `docs/op_classification.md` is for, and it's why the final model
adds a real embedding layer (`aten::embedding`, confirmed by trace to
both dispatch and fall back) instead of leaning on `linear`/`layer_norm`.

**The CI check fails on this model's own step, and the reason it fails
is a real, fixable inefficiency the check itself surfaced.**
`aten::arange` and `aten::unsqueeze` both land in the top 10 by time in
every run (`docs/ci_check_output.txt`). Both exist only because
`TinyEncoder.forward` recomputes position indices
(`torch.arange(T).unsqueeze(0)`) from scratch on every call instead of
caching them -- a real performance bug, not a necessary cost, and
exactly the kind of regression this CI gate exists to catch before
someone else notices the model got slower for no visible reason.

**A first attempt at a training task for the OpenVINO leg was
unlearnable, and the symptom (train loss falling while val accuracy sat
at chance) was the giveaway, not an error.** `src/task.py`'s first
version labeled each sequence `sum(token_ids) % 10`. Over 12 epochs,
training loss fell from 2.37 to 1.16, but validation accuracy never
moved off chance (about 0.09, expected for 10 classes); the model was
memorizing specific training sums rather than learning the mod-10 rule,
because a 1-layer attention block has no mechanism for exact multi-value
modular arithmetic. Switching the label to `token_ids[:, 0] % 10` (still
routed through the full encoder, since the classification head reads
position 0 only after attention and residual connections have mixed in
every other position) trained to 98.6% val accuracy in the same 12
epochs (`docs/openvino_train_output.txt`). This is the same category of
finding the ATen leg already made with `aten::linear`/`aten::layer_norm`:
what looks learnable or dispatchable by construction can simply not be,
and the fix is to measure first, not to assume.

**The INT8 accuracy gate is a real, tested mechanism that does not fire
on this model, and three genuine attempts to make it fire (not to force
it) all landed at the same answer.** The claim this leg was built
against was "blocked for breaching a 1% top-1 budget". Measured: FP32
98.63%, INT8 (300-example NNCF calibration, defaults) 98.54%, a 0.10
percentage-point drop, comfortably inside budget
(`docs/openvino_accuracy_gate_output.txt`). Two further genuine attempts
to find a configuration that actually breaches the budget -- a 5-example
calibration set, and a 1-example calibration set with bias correction
disabled and a MIXED quantization preset -- both still measured 98.54%
(0.10pp drop). The honest read is that this particular task and model
are simply robust to INT8 PTQ at this scale; the gate's block branch is
real and tested (`tests/test_accuracy_gate.py`) and would fire on a
model that actually regressed, but it is not demonstrated firing live
here, unlike the ATen leg's `ci_check.py`, which does fail live on its
own model. That asymmetry is stated rather than hidden.

**Measuring "FP32 vs INT8 throughput" surfaced that this build's "FP32"
path isn't running in f32 at all.** `docs/onednn_isa_attribution.md`
has the full finding: every linear in the nominally-FP32 IR executes
through a `brgemm:avx512_core_bf16` oneDNN kernel, because OpenVINO's
CPU plugin defaults to automatic bf16 inference on AVX-512 BF16
hardware. The 2.6x throughput target implicitly assumed a true f32
baseline; measured against this CPU's actual auto-bf16 FP32 path, INT8
(`brgemm:avx512_core_vnni`) is already starting from a smaller gap.

## Measured results

Machine: Windows 11, Python 3.12.10, PyTorch 2.12.0+cpu, no GPU of any
kind involved. `docs/ci_check_output.txt` is one representative run;
`fallback_pct_of_time` was also sampled over 5 repeated runs to report
honestly as a range (noisy on this shared machine at this model's small
size) rather than a single possibly-unrepresentative number.

**Claim: "112 ATen operator calls traced in one forward and backward
step."** Measured **133**. **Not met exactly**, same order of
magnitude; the model was redesigned three times during this project
(2-layer multi-head: 366 calls; 1-layer single-head: 173; 1-layer
combined-QKV projection: 124, before adding a real embedding layer
brought it to 133) to approach this figure without fabricating ops that
wouldn't actually appear in a real step -- see Findings for why
`linear`/`layer_norm` couldn't be used to get there.

**Claim: "14 fell back for 38% of step time."** Measured **59 fallback
calls, roughly 8-20% of step time** across 5 repeated runs (7.7% to
19.6%; `docs/ci_check_output.txt` shows one run at 18.15%). **Not met**
on either number. Root cause, not just a gap: most of this trace's
fallback calls (`detach`, `t`, `clone`, `expand`, `_unsafe_view`) are
near-free autograd view/metadata operations, not expensive generic
compute -- real fallbacks by the dispatcher's own definition, but cheap
ones, which is exactly why the call-count miss (59 vs. 14) doesn't
translate into a proportionally large time-fraction miss.

**Claim: "CI check fails the build when a fallback enters the top 10 by
time."** **Met, and demonstrated rather than just implemented**: on this
project's own model, it currently does fail
(`docs/ci_check_output.txt`: `aten::arange` and `aten::embedding`, and in
some runs `aten::unsqueeze`, rank in the top 10). The mechanism is
real and currently firing, not a hypothetical.

### OpenVINO/NNCF leg

Machine: Windows 11, 8-core/16-thread AMD Ryzen 7 7800X3D (AVX-512,
including AVX-512 BF16 and VNNI), Python 3.12.10, OpenVINO 2026.2.0,
NNCF 2.19.0, PyTorch 2.12.0+cpu. Single-threaded (`INFERENCE_NUM_THREADS=1`)
for the throughput numbers below, since this is a shared, in-use
machine and a multi-threaded number would partly measure core
availability rather than the kernel.

**Claim: "same model at FP32 and NNCF-calibrated INT8."** **Met.** The
trained `TinyEncoder` (98.63% val accuracy, `docs/openvino_train_output.txt`)
is exported once to an OpenVINO IR and NNCF-quantized once to INT8 from
that same IR; both are real, loadable, runnable models
(`artifacts/model_fp32.xml`, `artifacts/model_int8.xml`, regenerated by
`export_openvino.py` and `quantize_int8.py`, not committed, see
`.gitignore`).

**Claim: "each layer attributed to the oneDNN kernel ISA selected from
verbose logs."** **Partially met, and the partial part is the finding.**
The four `nn.Linear` layers are attributed exactly, by shape, to
`brgemm:avx512_core_bf16` (FP32 IR) or `brgemm:avx512_core_vnni` (INT8
IR) (`docs/onednn_attribution_output.txt`). The attention matmuls,
softmax, and the three `LayerNorm`s never appear in oneDNN's verbose
output at all on this build, confirmed by listing every distinct
primitive kind oneDNN logged (just `inner_product` and `reorder`); see
`docs/onednn_isa_attribution.md` for the full finding. "Every layer" was
the claim; four of roughly eight is what is honestly attributable
through this mechanism.

**Claim: "INT8 at 2.6x throughput."** Measured **1.18x-1.58x across
three genuine attempts** (single-stream batch-4 matching the resume's
implicit workload: 1.18x-1.40x across repeated runs; batch sizes 4
through 1024 single-stream: 1.28x-1.58x; a 6-thread THROUGHPUT-hint,
multi-request configuration: 1.32x) (`docs/openvino_throughput_output.txt`,
`docs/openvino_throughput_attempts.txt`). **Not met.** Root cause, not
just a gap: the "FP32" baseline already runs through automatic bf16
kernels on this CPU (Findings, `docs/onednn_isa_attribution.md`), so
INT8's measured advantage here is bf16-to-int8, not f32-to-int8, and
this model's linears are small enough that per-call framework and
reorder overhead is a real fraction of total time at every batch size
tried.

**Claim: "blocked for breaching a 1% top-1 budget."** Measured drop:
**0.10 percentage points, FP32 98.63% vs INT8 98.54%**
(`docs/openvino_accuracy_gate_output.txt`), well inside the 1pp budget.
**Not met**: the gate passes (exit 0) rather than blocking. See
Findings for the three genuine attempts made to find a configuration
that actually breaches the budget, all of which measured the same
0.10pp drop. The gate's block branch is real and unit-tested
(`tests/test_accuracy_gate.py`), demonstrated against synthetic
accuracy pairs rather than live on this model, unlike the ATen leg's
`ci_check.py`, which is demonstrated failing live.

## Building and running

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cd src
python ci_check.py       # traces one step, prints the top-10 table, exits 1 on a fallback regression

cd ..
python -m pytest tests -q
```

No GPU, no WSL, no ROCm: everything here runs on plain CPU PyTorch.

### OpenVINO/NNCF leg

```bash
pip install -r requirements.txt   # adds openvino, nncf, numpy

cd src
python train_classifier.py     # trains TinyEncoder, saves artifacts/model_fp32.pt + val_set.pt
python export_openvino.py      # torch -> OpenVINO IR, artifacts/model_fp32.xml
python quantize_int8.py        # NNCF PTQ, artifacts/model_int8.xml
python onednn_attribution.py   # per-layer oneDNN ISA attribution, both IRs
python accuracy_gate.py        # FP32 vs INT8 top-1, exits 1 past the 1pp budget
python throughput_bench.py     # FP32 vs INT8 single-stream throughput
```

`artifacts/` is regenerated by the four scripts above, in that order;
nothing under it is committed (see `.gitignore`).

## Layout

See Architecture above for the full file tree and what each file does;
`docs/op_classification.md` and `docs/onednn_isa_attribution.md` are the
two docs worth reading before the code, since between them they explain
why the model looks the way it does and what the OpenVINO CPU plugin
actually does and doesn't run through oneDNN.

## Known limitations

- CPU only. This audits the dispatch-fallback *mechanism*, which is
  identical in kind on any backend, but the specific set of ops that
  fall back, and their relative cost, would differ on a real ROCm build
  -- this project makes no claim about which ops fall back on AMD
  hardware specifically, only about how to detect and gate on fallbacks
  in general.
- The model is intentionally tiny (d_model=128, one encoder block, batch
  4, seq_len 16) so the whole trace stays legible; at this scale,
  Python-level dispatch overhead on cheap ops is a larger fraction of
  total time than it would be on a real workload, which is part of why
  the measured fallback time-fraction (8-20%) undershoots the original
  38% target -- see Measured results.
- `is_fallback`'s classification (no direct `CPU` kernel, only a
  Composite one) is a real, inspectable signal, but it isn't the only
  possible definition of "dispatch fallback" a different project might
  use; `docs/op_classification.md` states the exact rule so it can be
  checked rather than assumed.
- `aten::linear` and `aten::layer_norm`'s unreachability from this
  model's Python-level call sites is specific to this PyTorch build
  (2.12.0+cpu); a different PyTorch version's internals could route
  differently, which is itself a reason to query the live dispatcher
  (as this project does) rather than hardcode a fallback op list.
- The OpenVINO leg's accuracy gate is real and unit-tested but never
  observed blocking on a real model; the claim it was built against
  (a breach of a 1% budget) was not reproduced after three genuine
  attempts, see Measured results.
- "FP32" on this machine's OpenVINO CPU plugin silently means
  auto-bf16, not true `f32` kernels (`docs/onednn_isa_attribution.md`);
  every FP32-vs-INT8 comparison in this README is therefore honestly a
  bf16-vs-int8 comparison on this hardware, stated rather than hidden.
- The oneDNN attribution only covers the 4 `nn.Linear` layers; the
  attention matmuls, softmax, and LayerNorms run through OpenVINO's own
  native CPU executors and never reach oneDNN's verbose log on this
  build, so "every layer attributed to a oneDNN kernel" could not be
  made true by trying harder, only by claiming less.
- The classification task (`token_ids[:, 0] % 10`) is synthetic and
  deliberately easy (98.6% val accuracy in 12 epochs); it is a vehicle
  for a real FP32-vs-INT8 accuracy comparison on a model that actually
  learned something, not a claim about quantization robustness on a
  harder or more realistic task.
