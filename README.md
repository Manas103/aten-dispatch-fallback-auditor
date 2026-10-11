# ATen Dispatch and Fallback Auditor for a PyTorch Model Step

Traces every ATen operator call in one real forward-and-backward step of
a small transformer encoder through a `TorchDispatchMode` shim, attributes
wall time per operator, and classifies each call as either a dedicated
backend kernel or a fallback onto a Composite (decomposing) kernel, using
PyTorch's own dispatcher introspection API to ask the question directly
rather than guessing from an op's name. Every number below is measured on
this machine's CPU build of PyTorch 2.12, not estimated or assumed from
documentation.

**Honest framing, up front:**
- **CPU only, by design, not as a limitation being worked around.** This
  project audits *dispatch fallback*, the mechanism a backend without a
  registered kernel for some op falls back to a generic decomposition --
  that mechanism and its cost are visible identically on CPU, and no
  AMD or NVIDIA GPU is needed, used, or claimed anywhere here.
- **Measured: 133 ATen calls traced, 59 fell back, consuming roughly
  8-20% of step time across repeated runs** (noisy at this model's tiny
  size; see Measured results). Not the same numbers as this project's
  own original target (112 / 14 / 38%) -- reported as measured, not
  adjusted to match.
- **A real, non-obvious finding drove the final model design**: two of
  this project's three originally-planned fallback ops
  (`aten::linear`, `aten::layer_norm`) never once appear in any trace on
  this PyTorch build, because their Python-level call sites bypass the
  dispatcher-level composite op entirely. See Findings and
  `docs/op_classification.md`.

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

docs/op_classification.md   the finding that reshaped this model's design
docs/ci_check_output.txt    raw CI check output backing every number below
docs/test_output.txt        raw pytest output
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
- **10 pytest checks total** (`docs/test_output.txt`).

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

## Layout

See Architecture above for the full file tree and what each file does;
`docs/op_classification.md` is the one doc worth reading before the
code, since it explains why the model looks the way it does.

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
