# Conditional prospective two-update harness

Saved before trained outputs were viewed. NOT executed, and not yet a
final ready-to-run data protocol: checkpoint and data provenance must be
resolved. Select batch size 2 (one existing source setting), two Adam steps,
lr=1e-4, fixed reference, no EMA. Use CPU float64 for the mechanistic study;
this is a documented precision/device change from the source, not numerical
reproduction of its reported results. Preserve Adam defaults explicitly
(betas .9/.999, eps 1e-8, no weight decay, amsgrad=False, foreach=False).

Take the first 20 consecutive full batches of the declared held-out order
(40 inputs); do not choose batches by observed events. Use unmasked inputs
(source p=0 setting); reject missing/insufficient inputs rather than change
batch size after results. Baselines: no adaptation and all four
current/cached x identity/zero input-derivative variants. Allocate two
cached scalars in each adaptation arm. Only cached arms use them in their
forward reference input. No evaluator labels enter updates, batch selection,
cache, mask/sign-event selection or model selection.

Create a canonical current-connected Adam at the ttt-call boundary with
fresh state. For each batch reset its student weights to the fixed reference,
retain its optimizer history, and copy that full snapshot to all four arms.
Each arm resets weights, then computes cache c=q0(x). Run two steps and
discard the three noncanonical histories after collecting results. Advance
only canonical current-connected history to the next batch. This provides
matched-state forks while preserving a declared source-style history; it
is NOT four independently carried streams. No-adaptation retains original
weights and consumes no optimizer updates. Primary total: 160 fixture-free
adaptation updates (20 batches x 4 arms x 2), with canonical reuse, not an
additional fifth adaptation run. Hard CPU computation cap 60 seconds after
artifact loading; incomplete work is not a primary pass.

Before each update record student gradients, loss signs, reference masks
and input Jacobians. Verify first-update equality within each derivative
pair. At the connected pair's common second evaluation compute the exact
sign/Jacobian decomposition using the SAME shared state. Preserve complete
optimizer moments/counters and report whether gradient differences survive
the Adam transition. Return/evaluate student outputs, not the reference
output confused by notebook variable names.

Correctness tolerance for float64: max absolute error <=1e-10 times
max(1, max absolute values compared). Equality or decomposition failure
beyond this threshold invalidates the harness result. Cache-forward values
must equal the stored cache exactly for finite inputs. Log zero and near-zero
residuals/preactivations (absolute <=1e-10) rather than dropping batches.

Exploratory nontrivial-effect criterion, fixed before outcomes: in >=5/20
batches, the connected pair's second-gradient difference must exceed
1e-4*max(||g_current||_2,||g_cached||_2,1e-8), and its post-update student
output RMS difference must exceed 1e-6*max(RMS(q0),1). Also report every
batch's raw magnitude, including zeros. This threshold declares a measurable
mechanism in this small study, not useful accuracy or statistical significance.
If no measurable effect or count criterion fails, reject the effect for
this setting without changing steps, masks, batch order or thresholds.

The identity requires sign/Jacobian differences before gradient separation;
differences in masks alone may cancel. Report the two decomposition terms,
their sum and cancellations; do not attribute additive risk effects to them.
Primary endpoint is unlabeled output/gradient separation. Any future labeled
risk evaluation must be separately declared after this protocol and must
not influence this gate. Training-derived target normalization is model
metadata; any full-dataset target statistics in an inherited checkpoint
must be disclosed as extra information. This study alone cannot establish
continual adaptation, retention, recoverability, novelty or a useful method.
