# Paper-claim decision: finite-time prediction-input feedback

29 October 2026, cycle 13. Completed the decision protocol saved in cycle 12. The preceding 28 October run was interrupted after source discovery and extracting the supplied EMA PDF; no proposal changes or completed decision were saved then. This run resumes that work. Status: bounded prior-art comparison and research decision, not a new performance result or novelty certification.

## Decision

**No-go on the nonsmooth closure branch as the main first-paper claim with present evidence.** The sign-box counterexample is a useful checked application, but the general modeling warning, rank condition, gradient closure and algorithm-aware nonsmooth calculus have substantial prior foundations. We have not shown that the exact constructed example was previously published; no-go means the present package lacks a sufficiently distinct implementation-relevant consequence. Do not erase its proof or portray it as disproved.

**Conditional alternative:** can initial derivatives and computable remainder bounds predict the finite-time output effect of feeding changing student predictions into a reference? Start with the existing smooth squared-loss stopped-versus-cached comparison and its known early forecast. The target is a certified range of validity of that forecast, not another explanation that EMA can fail. A useful theorem would state a checkable radius and finite-step error bound. A useful experiment would show that bound is nonvacuous without selecting a favorable horizon after inspection. Ordinary Taylor bounds alone are not a novelty claim.

## Two-source comparison and exact scope

Bolte--Pauwels, arXiv:1909.10300v4: Section 5.2(d), Theorem 8 (p.22), Corollary 5 (p.24), Theorem 1/Corollary 1 (pp.8-10). With conservative primitive derivative fields, automatic differentiation yields a conservative field; it equals the classical gradient almost everywhere. Clarke and algorithmic fields need not coincide at every point. Their Theorem 9 (p.26) concerns a fixed definable finite-sum objective, independent uniform minibatches, diminishing steps and a boundedness event; it does not supply our constant-step EMA/Adam guarantee. We do not reuse that convergence theorem.

Application to our graph: output detachment is not justified by ordinary nonsmooth differentiation of the same forward loss. Its identity forward operation with zero backward derivative violates the conservative chain rule. In the existing cycle-12 example, away from u=0, -F_stop=(sign(u)/2,-sign(u)/2) differs from grad|u|=(sign(u),0); Theorem 1 prevents treating that stopped field as a conservative derivative of this discrepancy. This conclusion does not exclude a different potential or a separately justified analysis.

The user's supplied Patsenker--Li--Kluger paper, arXiv:2310.13854v1, Section 3/Eqs.(6)-(7), Algorithm 1 (p.3), already treats averaged-weight feedback through BELAY. With its momentum coefficients canceled, the printed algorithm has

    theta_next = a*(theta + eta*g(theta,t)) + (1-a)*phi
    phi_next   = b*phi + (1-b)*theta.

Its second line uses the old student state. This differs from our student-first EMA. Its direct parameter mixing also differs from our prediction-input path and state-dependent Jacobians. Therefore broad two-way EMA feedback or spring analogies are not defensible novelty claims; neither are they a proof that this particular input-path comparison has already been settled. The paper's Section 5 leaves convergence bounds for future work. No BELAY code or experiment was reproduced.

Only these two additional papers were inspected. The second was chosen to resolve a specific overlap in the alternative feedback target and because the user supplied it. The earlier PeTTA/RMT/fixed-teacher/IT3 audits are reused. No broad literature expansion occurred. Full source identifiers and equation-convention caveats are in SOURCE_EVIDENCE.md.

## A useful consequence of the comparison, not new theory

For a finite locally Lipschitz loss program L whose standard primitive derivative fields satisfy the conservative-calculus assumptions, choose a locally bounded measurable ordinary-backprop field g. Theorem 8 and Theorem 1 give g=grad L almost everywhere. The established essential-gradient characterization therefore gives K[g]=partial_C L. For a constant student-coordinate projection P and continuous reference drift b,

    K[b-P*g](z) = b(z)-P*partial_C L(z).

For a frozen reference, apply the theorem to the student-only program with those constants substituted; an almost-everywhere statement in a joint space alone would not justify every frozen slice. For a moving reference, use the full joint loss program and project its derivative. This deduction applies to ordinary finite ReLU/absolute-loss programs with conservative primitives, as well as smooth ones. It DOES NOT give a formula using one selected ReLU Jacobian times independently chosen residual signs at a kink: that selected Jacobian is not continuous there. It also does not identify the entire algorithmic conservative field with the Filippov map. Exceptional-point selections can matter to finite discrete iterates even when the regularized continuous map ignores them. It supplies no stopped-output, Adam, trajectory-convergence or predictive-accuracy guarantee.

This reinforces the decision: merely extending an a.e.-gradient closure statement to ordinary ReLU backprop would be application of existing theory, not a sufficient paper contribution.

## Reviewable thesis and conditional first-paper package

Motivation: output consistency, parameter averaging and accuracy can diverge. The reference's prediction input is a concrete path that can be intervened on while retaining the same weight-update rule. The completed squared-loss study shows a large effect, but caching is retained per-input information and still fails to preserve initial accuracy. It is a diagnostic, not a ready adapter.

Question: **Can a computable finite-step certificate predict the output effect of prediction-input feedback and state when that prediction ceases to be reliable?**

Claim to test, not claim established: for the specified smooth student-first update, an initial affine forecast plus a validated local remainder enclosure yields a nonvacuous bound on stopped/cached output separation over the preselected early horizon. The theorem assumptions must be measurable/validatable properties of the declared graph, not an unexplained contraction or a fitted trajectory constant.

Model and information: deterministic repeated batch, squared discrepancy with the existing normalization, tanh networks, gradient descent, tau=1 student-first EMA, eta=.02/.01, 82 active coordinates per copy. Existing pretrained seeds 0,2,3,4 are primary;1 sensitivity-only.64 unlabeled adaptation inputs;192 independent evaluator inputs; no evaluator labels in forecasting or certificates. Each arm allocates the same128-number cache, used in adaptation only by the cached control. State norm uses the existing unscaled active-coordinate basis; output norm is RMS across192x2 outputs. No reparameterization to make the bound pass.

Theorem plan: write each exact joint update as Phi_a and its affine approximation around z0 as u_next=c_a+A_a*u. Bound the difference from the nonlinear state using a Lipschitz constant of D Phi_a on a declared ball; separately bound output Taylor error. Certify domain containment inductively rather than assuming the actual trajectory stays local. Track both arms and their output difference. Standard Taylor/perturbation machinery is the proof tool; practical tightness and a distinct validated mechanism remain the contribution burden.

Immediate gate: at T=2, both step sizes, at least3/4 primary seeds must admit a valid output-difference error bound B no larger than20% of the predicted separation, with non-negligible separation. By the reverse triangle inequality this implies at most25% error relative to the true separation. Reuse the previously declared scale floor. This is a new nonvacuity test on already observed data, not independent empirical confirmation. Long-horizon checks cannot rescue failure at T=2.

Falsification: failure to obtain valid derivative enclosures, loss of the certified domain, or an overly loose bound in2+ primary seeds rejects this certificate route in its present form. A bound violated by saved states is a correctness failure. Do not tighten it by fitting observed errors, changing thresholds/horizon, selecting seeds or assuming sign/activation margins not certified. Preserve cycle10's failed absolute-loss numerical gate.

Milestones: (1) derive the scoped discrete certificate and its enclosure recipe; (2) assess nonvacuity on existing data without new trajectories; (3) only after a pass, preregister a prospective test on one audited regression graph, with full-input-gradient and cached/full-gradient controls and separately evaluated risk. Declare activation/loss/optimizer assumptions anew before that test. Official absolute-loss/ReLU/Adam transfer is outstanding. Fixed teacher, no adaptation and original full-input-gradient baselines must be included in any later accuracy study, with equal update/data budgets. No claim about current risk follows from output separation alone.

A master's thesis can honestly present the graph audit, constructed stability distinctions, checked synthetic intervention/forecast, failed transfer and faithful-model analysis even if this gate fails. A first paper still needs new, useful evidence beyond those ingredients. Do not automatically rebrand a failed certificate as a successful method.

## This cycle's work and preservation

No new training, adaptation, labels, parameter sweeps, package installations or paid compute. Source pages were extracted/rendered; the exact BELAY update and Theorem8 were visually checked. The source and proposal edits are bounded by hashes and backups. Adding citations/research notes here is not uploading a PDF to the ChatGPT project's shared Sources; that separate user request remains uncompleted. The synced sources directory was not changed.
