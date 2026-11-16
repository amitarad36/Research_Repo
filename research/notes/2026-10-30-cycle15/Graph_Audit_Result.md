# Forward input values and backward paths: a two-update relevance gate

30 October 2026, cycle 15. Completed the cycle-14 static/analytic protocol.
Result: the earlier stopped/cached diagnostic does not by itself isolate the
effect in the official connected-input graph. A four-graph comparison gives
an exact discrete equality criterion and a narrowly testable two-update
question. No trained-model experiment, performance result or novelty claim.

## Primary implementation evidence

Reused and hash-verified the official notebook at revision
100524001d890efc08911f3cba8f45f51f4aa7d1:
[tabular/vanilla_model.ipynb](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb).
The retained notebook SHA256 is
6d583591e7433547c3b8fbb85f2177e51a933596f21709bc30b251be89c1e227;
numbered extraction SHA256
ead7cade66b9df7fa7e90d4c66d986b75cd847d23473d4ea86f80e3737ee46d2.
The exact source_manifest.json is embedded in graph_verification.json;
original files remain in ../2026-10-26-cycle9. No downloaded code executed.

Zero-based cell 5 lines 82-109 defines Linear/ReLU layers of widths
30/20/10 and a linear output. Cell 12 lines 5-9 selects 13 inputs, one
output and default auxiliary input k=50. Thus q_theta(x)=f_theta(x,50)
here, unlike the zero-input/tanh/two-output surrogate.
Cell 13 lines 34-38 passes q into the reference without detaching and
takes mean absolute discrepancy. Lines 41-58 reset student weights per
batch, perform updates and return both outputs. Lines 61-64 construct
student-only Adam once per ttt call; lines 90-95 reuse it across batches.
Line 103 calls exactly TWO updates, learning rate 1e-4. The reported
y_hat_2 is the student output despite confusing nested variable names.
No EMA appears in this cell. Reference parameters are not stepped;
eval() is not a gradient detach, and unused reference gradients may still
be accumulated. The listed model has no dropout or running-statistic layer.
This is a static code interpretation, not a reproduction of notebook results.

## Four precisely specified graphs

For each sample let q=q_theta(x), J=D_theta q, cached c=q_theta0(x),
T_phi(x,z) the reference, A_v=D_z T_phi(x,z_v), r_v=T_phi(x,z_v)-q.
Treat c as constant, and hold phi fixed during the student backward pass.
Let v select current or cached FORWARD input and b in {0,1} select the
input derivative with respect to q. A single loss backpropagates as

    U_(v,b) = (1/N) sum J^T (I-b*A_v)^T r_v        [squared]
    U_(v,b) = (1/(N*d)) sum J^T (I-b*A_v)^T sign(r_v) [absolute],

where U is the negative student gradient. Squared normalization is
half the mean per-example squared vector norm, matching the previous
surrogate. Absolute normalization is mean over all N*d entries, matching
the inspected cell. Use the specified selected derivative at zero; the
strict-region statements below exclude zero residuals/activations.

| Forward value | Input derivative | Expression | Interpretation |
|---|---|---|---|
| current q | I | q | audited connected-input path |
| current q | 0 | stop_gradient(q) | earlier stopped comparison |
| cached c | 0 | c | ordinary cached-input comparison |
| cached c | I | c + (q-stop_gradient(q)) | cached-value straight-through diagnostic |

Parenthesize q-stop_gradient(q) before adding c; in ordinary finite
floating-point evaluation it cancels identical q values before addition.
This avoids the avoidable cancellation error of (c+q)-q. Nonfinite inputs
are outside the test. The straight-through graph deliberately changes the
backward rule; it is not the ordinary derivative of its forward loss or
an official method. Keeping the identity input derivative does not hold
the numerical A_v fixed: caching can change which reference ReLUs are on.
Stopping the entire reference output and stopping its input give the same
student direction here, but need not give the same reference .grad buffer.

## Matched-start and optimizer-state equality

At c=q0, the current/cached pair has equal r and A for each fixed b.
Thus current-connected equals cached-connected initially; current-stopped
equals cached-stopped initially. The connected and stopped groups need
not agree: their difference contains J^T A^T w.

Equal gradients give equal first Adam moments, second moments, bias
corrections and updates if parameters, optimizer step count, all moment
buffers and hyperparameters are identical. This follows directly by
substitution into the deterministic Adam recurrence. Equal weights alone
are insufficient. In the inspected code, weights reset but optimizer state
persists, so separately evolved arms can begin a later batch with different
optimizer states. Fork a COMMON snapshot for a controlled batch comparison;
do not claim first-update equality automatically across independent streams.

## A useful discrete theorem for absolute loss

At any shared student/reference state, shared deterministic inputs and
shared optimizer state, suppose sign(r_current)=sign(r_cached) samplewise.
For b=0, directions agree without an assumption on A. For b=1 they also
agree whenever A_current=A_cached. In a fixed feed-forward ReLU reference,
identical activation masks at the two inputs are sufficient for that
Jacobian equality. They are not necessary, since different masks can yield
the same Jacobian. Apply the displayed formula to prove gradient equality;
the optimizer's deterministic transition gives equal next states.
Induction gives equality for as long as these conditions hold at each
shared iterate. Student masks need not stay constant over time: both arms
have the same student and J at every shared state.

Consequently, a first gradient difference in the connected absolute-loss
comparison requires a residual-sign difference OR a reference-input
Jacobian difference at the shared state. Different ReLU masks are a
detectable sufficient warning, not proof that A changed or the update
differs. Mismatches can cancel after multiplication by J and aggregation.
The result concerns discrete evaluations; it does not locate continuous
crossing times. It does not require the failed continuous-time certificate.

At a common state the exact connected-field difference decomposes as

    Delta U = 1/(N*d) sum J^T [
        (I-A_current)^T (s_current-s_cached)
        + (A_cached-A_current)^T s_cached ].

This is an algebraic decomposition of the gradient difference, not an
additive decomposition of nonlinear Adam updates or risk. Calling the
first term a sign contribution and the second a Jacobian contribution
requires this declared ordering; other decompositions are possible.

For squared loss, same signs and ReLU masks do NOT give this equality:
residual magnitudes still matter. The previous smooth squared mechanism
therefore cannot be presumed to describe this absolute-loss setting.

## Exact checks and counterexamples

verify_graphs.py verifies 96 local derivatives of the four specified
graphs using exact rational symmetric differences on strictly affine
regions, 24 matched-gradient/Adam-moment identities, and six vector
connected-absolute decompositions (N=2,d=2 normalization). These are
algebraic fixture checks, not PyTorch autograd or trained-network tests.

Three scalar witnesses with q_theta=theta, N=d=1:

* T(z)=2 ReLU(z)+1, c=1,q=1/2: same masks, positive residuals;
  connected absolute directions both -1, squared directions -3/2 and -5/2.
* Same T,c, q=-1/2: positive residuals but changed reference gate;
  connected absolute directions +1 and -1.
* T(z)=z/2+1, c=0,q=3/2: identical Jacobians but opposite residual signs;
  connected absolute directions +1/2 and -1/2.

All exact checks pass in .015625 CPU seconds, below the 10-second cap.
No labels, neural trajectories, remote execution or additional paper read.

## Decision: one prospective causal question, not a paper result

Retain the input intervention only as a conditional, implementation-relevant
diagnostic. Question: **in the pinned two-update absolute-loss regression
graph, how often does caching the forward input change the SECOND update
when the identity input-backward path and initial optimizer state are held
fixed, and is the gradient difference accounted for by the sign/Jacobian
decomposition?** Because the first updates agree, this isolates a common-state
comparison before divergent parameter/optimizer histories complicate it.
It does not assert that either arm is better. At most two steps per arm,
fixed lr=1e-4 and no EMA, to match this selected call.

Prospective controls: no adaptation, current-connected, cached-connected,
current-stopped, cached-stopped. Use the same batch order, copied optimizer
snapshot, dtype and cached N*d scalars allocated for all adaptation arms;
only cached arms consume their values. Cache is created AFTER the per-batch
weight reset and expires at that batch's end. No cached evaluator labels.
For primary isolation fork one common state per batch; independent carried
optimizer histories are a separate later experiment. Verify first-pair
equality, evaluate both second-step gradients at their shared parameter
state, and log masks, signs, A and the decomposition, including cancellations.
Measure student-output change; any labeled risk is evaluator-only and
secondary, not an unlabeled selection signal.

Pass/fail must separate correctness from evidence: equality or decomposition
violations beyond a declared dtype tolerance fail the harness; a validated
zero or negligible effect is a negative mechanism result, not a bug.
Fix a batch/sample budget and nontrivial-effect criterion before viewing
checkpoint outputs. Do not reuse an incompatible tanh checkpoint as the
official ReLU model or retune until an effect appears. Checkpoint provenance
and available CPU autograd are not established this cycle. The next protocol
resolves those prerequisites without running a performance study.

Novelty remains unverified. The chain rule, ReLU region property and Adam
equality are standard; the four-graph diagram alone is not a first paper.
Useful current-risk effects, generalization beyond two steps, continual
relevance and a distinct contribution remain required. Preserve all earlier
failed gates and do not rebrand this audit as their successful replacement.
