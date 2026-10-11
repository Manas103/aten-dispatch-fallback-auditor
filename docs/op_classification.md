# Op classification: what counts as a "fallback" here, and what doesn't

## The classification rule

`src/auditor.py`'s `is_fallback(func)` asks PyTorch's own dispatcher,
via `torch._C._dispatch_has_kernel_for_dispatch_key`, two questions about
the op's overload-qualified schema name (e.g. `aten::embedding`, not the
bare `aten::embedding` without an overload, and not `str(func)`'s dot
form `aten.embedding.default`):

1. Does the `CPU` dispatch key have a directly registered kernel?
2. Does `CompositeImplicitAutograd` or `CompositeExplicitAutograd` have one?

An op is a fallback if (1) is no and (2) is yes: no backend-specific
implementation exists, so the dispatcher resolves it to a kernel that is
itself implemented in terms of other ATen ops. Those other ops are
dispatched too, and this same `TorchDispatchMode` captures them as
separate calls in the trace -- a fallback call shows up as itself *plus*
whatever it decomposed into.

## A finding that shaped the whole project: many "composite" ops never
## get dispatched at all on this PyTorch build

The first design assumption was that `nn.Linear` and `nn.LayerNorm` --
both genuinely `CompositeImplicitAutograd`-only, no direct `CPU` kernel,
confirmed directly against the dispatcher -- would show up in a trace as
fallbacks decomposing into `addmm`/`native_layer_norm`. They never do.
`F.linear` and `F.layer_norm`'s Python-level implementations call the
backend-specific primitive (`torch.addmm`, `torch.native_layer_norm`)
directly, never going through a dispatcher call to the generic
`aten::linear`/`aten::layer_norm` op at all. The same is true of
`F.cross_entropy`, which composes `log_softmax` + `nll_loss` in Python
rather than dispatching `aten::cross_entropy_loss`.

This matters because it means "an op is Composite" is not the same claim
as "an op shows up as a fallback in a real trace" -- the Python-level
call site has to actually reach the dispatcher as that op for this
project's method to see it. The model in `src/model.py` was redesigned
around this finding (see Findings in the README): it keeps `aten::
embedding` (confirmed, by trace, to actually dispatch and actually
fall back) and drops the assumption that `nn.Linear`/`nn.LayerNorm`
would contribute any fallback signal.

## What this trace's fallback ops actually are

Ranked by call count in the measured step (`docs/ci_check_output.txt`):
`aten::detach`, `aten::t`, `aten::view`-adjacent aliasing ops
(`aten::_unsafe_view`, `aten::expand`), `aten::clone`,
`aten::embedding`, `aten::arange`, `aten::unsqueeze`, `aten::sum`,
`aten::ones_like`, `aten::select.int`, `aten::split.Tensor`. Most of
these are near-free view/metadata operations autograd inserts around
every `nn.Linear`'s backward pass (a weight transpose, a `detach` on
saved tensors) -- real fallbacks by the dispatcher's own definition, but
not expensive ones. `aten::embedding`, `aten::arange` and
`aten::unsqueeze` are the step's only fallbacks that do real,
non-trivial work, and notably `arange`+`unsqueeze` exist in this model
only because position indices are recomputed from scratch every forward
call (`torch.arange(T).unsqueeze(0)` in `TinyEncoder.forward`) --
itself a real, fixable inefficiency the CI check's own output flagged
(see README Findings).
