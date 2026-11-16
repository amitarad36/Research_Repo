# Two-update study: runtime verified, trained artifacts not located

31 October 2026, cycle 16. Resumed the 30 October evening heartbeat after
its initial read-only inventory. The original proposal/state/log hashes
still matched cycle 15. Scope: local artifact/runtime feasibility and one
data-free conformance fixture; no trained-model performance study.

## Inventory and concrete limitation

Searched these three bounded roots with rg --files --hidden, excluding
.git/node_modules, for checkpoint/archive/table extensions, Boston/Housing
filenames and Python environment metadata:

* C:\Users\amita\source\repos\Detach Proposal Presentation: 106 NPZ files.
* C:\Users\amita\source\repos\Project with Assaf: five checkpoint files in
  style_transfer/saved_models and DDNM/exp/logs/celeba; no tabular candidate.
  The style-transfer README identifies the four matching named styles.
* This ChatGPT project's local mirror: 106 NPZ files; synced sources read-only.

Full paths, sizes, patterns and scope are in inventory_results.json. No
matching Boston Housing file, tabular checkpoint or separate preprocessing
bundle was located in those roots. This is NOT a machine-wide absence claim.
The five known initial models were inspected without loading labels: each
contains 114 parameters and is the documented tanh/two-output surrogate.
The inspected official architecture has 1291 parameters, ReLU widths
30/20/10, 13 inputs, one output and auxiliary value 50. Tensor shape is only
an exclusion check, not sufficient provenance. Unrelated .pth/.ckpt files
were not deserialized or repurposed. No source directory was modified.

## Runtime and data-free conformance

The bundled Python has NumPy but no Torch/JAX/TensorFlow module.
The existing C:\Users\amita\anaconda3\python.exe successfully imports
PyTorch 2.9.0+cpu. The registered final-dl-env has a Torch module spec but
was not imported or tested; linearizer has no Torch spec. No install or
environment change was made. Registry: C:\Users\amita\.conda\environments.txt.

conformance.py constructs the source-sized Linear/ReLU architecture in CPU
float64 with deterministic assigned coefficients and two invented input
rows. These are fixtures, not trained weights or research data. It checks
all four declared graphs over two steps, starting from cloned nonzero
Adam step/m/v history. It also performs one negative-control step with
altered history: nine tiny fixture optimizer steps total. Reference
parameters have requires_grad=False for the fixture while the reference
input remains differentiable; this suppresses unused parameter gradients,
unlike the notebook, without removing the tested input path.

All checks passed in 8.203125 CPU seconds including import, below the
10-second cap. Formula-vs-autograd maximum error, cached-forward deviation,
and matched first-gradient/parameter discrepancies were all zero on this
fixture. Cache values remained unchanged. load_state_dict reset weights
without resetting Adam step/m/v. The different-history negative control
changed the first update by 3.1565392526883285e-7 in max parameter norm,
above the predeclared 1e-12 tolerance. It confirms the confound; it does not
estimate its prevalence. Float32, trained-data behavior, activation/sign
events and effect sizes remain untested. No labels/checkpoint/network
download, training, benchmark or GPU work occurred.

## Preprocessing provenance matters

Reused the hash-pinned notebook from cycle 9 (commit
100524001d890efc08911f3cba8f45f51f4aa7d1; notebook SHA256
6d583591e7433547c3b8fbb85f2177e51a933596f21709bc30b251be89c1e227).
Cell 1 names bostonHousing.txt and its upstream download URL. Cell 2,
lines 9-26, reads features/targets and computes both means and standard
deviations on the full dataset. Cell 4, lines 46-50, THEN takes the first
90% of rows for training and the remainder for testing. The model sees
normalized targets and has default auxiliary input 50, so its target scale
is essential to reproducing its forward graph. These statistics must be
paired with any checkpoint; do not recompute different statistics silently.

The source recipe therefore uses held-out input/target statistics before
the split. That is a declared information-budget difference from strictly
training-only preprocessing; it cannot be described as evaluator-only
held-out-label access. No such statistics were loaded here. A newly trained
study should fit statistics on training rows only and explicitly declare
that deviation from the notebook, rather than inherit test-label access
or pretend to reproduce its exact weights/results.

Cell 12 trains an in-memory model for 30 epochs. The saved code extraction
contains no torch.save/torch.load checkpoint persistence; load_state_dict
in cell 13 is an in-memory reset. This observation is limited to the
inspected notebook, not the entire official repository.

## Decision and minimum next step

The CPU harness is feasible. The trained-model study is not ready because
no compatible trained artifact plus input/split/scaling provenance was
located in the bounded inventory. A data-free fixture cannot substitute.
Minimum input bundle: source/model revision, state_dict and dtype, raw-data
version/hash, row order/split indices, feature and target scaling, training
seed/settings and evaluation information budget. The initial adaptation
Adam state is fresh at the ttt-call boundary; later per-batch snapshots are
generated only by the declared baseline continuation.

HARNESS_PLAN.md fixes the intended controls/state handling and bounded
study design without claiming compatible artifacts exist. NEXT_ARTIFACT_PROTOCOL.md
resolves the remaining provenance through a bounded primary-repository
artifact check; if unavailable, it prepares a small, separately registered
local reproduction using the working runtime. Do not repeatedly re-inventory
the same directories or turn this prerequisite failure into a scientific
rejection of the mechanism. All earlier negative scientific gates remain.
