# Next bounded task: make the two-update study executable or record its blocker

Read the cycle-15 graph result and pinned cell 13. Inventory only relevant
local research/checkpoint locations and installed CPU autodiff runtimes.
Do not install packages, retrieve paid resources, train, use evaluation
labels or launch a benchmark. Distinguish the existing tanh checkpoints
from the 13-input, one-output, 30/20/10 ReLU/k50 notebook architecture.
Check model and preprocessing provenance; a compatible tensor shape alone
is not evidence of an appropriate pretrained checkpoint.

Prepare a reviewable CPU harness plan implementing the four declared
graphs, two steps at lr=1e-4, frozen reference and per-batch weight reset.
Explicitly clone Adam step/m/v state and allocate N*d cache in each arm.
Primary intervention forks each batch from a common snapshot; it does not
silently replace the source's carried optimizer with reset moments.
Choose the one source batch-size setting to test before viewing outputs.
No-data fixture checks may validate gradients/reset/cache/return-output
semantics within 10 CPU seconds if the runtime is already installed; they
are not performance evidence. Preserve failures rather than installing or
changing runtimes to conceal them.

Deliver either (a) compatible provenance and a fixed prospective protocol
with batch/update budget, dtype tolerance, no-adaptation and four-graph
controls, nontrivial-effect threshold and falsification criteria, ready
for a separate run; or (b) the exact missing checkpoint/data/runtime and
the minimum feasible next step, without inventing results or substituting
the tanh study. No additional literature expansion or bound tightening.
