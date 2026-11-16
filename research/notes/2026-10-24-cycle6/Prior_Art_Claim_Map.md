# Cycle 6: what remains after the closest teacher-feedback comparisons?

24 October 2026, afternoon cycle. Bounded audit of three primary papers and two official repositories. No new training, trajectory experiment, or benchmark reproduction was performed. Exact code snapshots and their hashes are in `code_sources.json`.

## Decision

**No-go as a first-paper claim:** generic EMA error accumulation, a benefit from frozen teacher weights, and existence of positive local modes. Those observations do not establish a contribution beyond the closest work. The cycle-5 mismatch diagnostic does not rescue that claim: unequal J and L were not necessary for its local signature.

**Conditional go for one discriminating experiment:** does feeding the evolving student prediction into the reference materially cause the stationary risk deterioration in the existing trained model? Freeze that input at its source-checkpoint value while retaining exactly the same reference-weight schedule. This is a realizable control, unlike independently replacing L in a derivative matrix. The proposed distinction is between two forward graphs, not between the presence and absence of an autograd detach alone.

This is a research decision, not a novelty certification. None of these three sources supplies the precise dual-input result we would need, but absence from this bounded comparison is insufficient to establish originality. A paper would additionally need a mechanism that predicts the controlled neural behavior and survives comparison with other bootstrapped-target learning analyses. Do not start that broader search before the discriminating result exists.

## Source evidence and exact scope

### PeTTA

[Persistent Test-time Adaptation in Recurring Testing Scenarios, arXiv:2311.18193v4, 2 November 2024](https://arxiv.org/html/2311.18193v4). Read Section 3.2, Assumption 1, Eq. (4)-(5), Theorem 1, Corollary 1, and Appendix B.3. Its binary Gaussian-mixture analysis uses static true-class proportions, deliberately perturbed pseudo-labels, moment estimation, and a mean update. Eq. (5) conditions on the predicted negative class; its epsilon must not be silently replaced by the usual class-conditional false-negative probability. Theorem 1 bounds movement of an estimated class mean toward the other mean. Appendix B.3 obtains this from a scalar recurrence and a triangle inequality; Corollary 1 adds a sustained error-sequence condition. This is not a full student/reference neural-Jacobian stability theorem. The scalar update coefficient is the new-state weight, whereas our alpha denotes teacher memory. No PeTTA implementation was audited or run in this cycle.

### RMT

[Robust Mean Teacher for Continual and Gradual Test-Time Adaptation, arXiv:2211.13081v2, 22 March 2023](https://arxiv.org/html/2211.13081v2). Read Section 3.1, Eqs. (1)-(6). Its binary output-probability derivatives hold teacher probability q fixed: CE gives (p-q)/(p-p^2); SCE adds log((1-q)/q). Consequently, agreeing probabilities can still be sharpened by SCE. These are loss-gradient comparisons, not a coupled local-stability or predictive-risk guarantee. Section 3.2 uses stored source prototypes; Section 3.3 allows source replay, so the full method's information budget differs from our surrogate.

Official code snapshot: [revision 8b2ed63255d906b6b0afaefc6ae630b9ec4bdaa1](https://github.com/mariodoebler/test-time-adaptation/tree/8b2ed63255d906b6b0afaefc6ae630b9ec4bdaa1), committed 29 May 2025; not asserted to be the publication-time revision. `classification/methods/rmt.py`: 57-59 detach teacher parameters; 235-247 feed images directly to separate student/teacher calls; 265 builds self-training loss; 297-308 update student then EMA. `classification/utils/losses.py`: 13-19 implements weighted SCE. No student prediction is a teacher input on this inspected path. Static inspection only.

### Fixed-teacher comparison

[Rethinking the Teacher-Student Framework for Test-Time Adaptation, arXiv:2609.02507v1, 2 September 2026](https://arxiv.org/html/2609.02507v1). Read Sections 4.1 and 5.1-5.2, Eqs. (1)-(5). Section 5.2 explicitly explains positive feedback through erroneous teacher supervision; its EMA unrolling is an identity, not a derived numerical stability boundary. The comparisons freeze trainable teacher weights while retaining the stated batch-normalization policy. This already overlaps our generic feedback interpretation. The paper's ground-truth-gradient decomposition is analytical; it does not grant training access to labels.

Official [revision bd6a4fd9cf45c50b83c10d582172c228c771333f](https://github.com/dmn-sjk/intransigent_teacher/tree/bd6a4fd9cf45c50b83c10d582172c228c771333f), committed 13 September 2026: `methods/losstest.py` 95-135 implements a teacher on augmented images under no-grad and skips EMA when `M_TEACHER.FROZ` is true. `methods/cotta.py` 74-94 corroborates the image-input path and update switch. These inspected paths do not feed student predictions to the teacher. Some methods return teacher outputs while separately tracking student accuracy; the evaluation driver was not audited, so no code-level claim about reported prediction timing follows. Repository metadata mentions CoLLAs; this audit uses the specified arXiv version and does not independently verify proceedings status.

## Comparison with the actual project claims

| Project claim | Decision after this audit | Remaining obligation |
|---|---|---|
| EMA can harm predictions; freezing weights can help | No-go as a distinct paper claim | Keep as a control and reproducible observation |
| Generic consistency-gradient geometry matters | No-go at this level of generality | Specify the forward graph and show a discriminating outcome |
| PeTTA directly proves our learned-model modes | Unsupported transfer | Different variables, update, and outcome; derive our own statement |
| Cycle-5 positive modes require J unequal to L | Rejected by our matrix diagnostic | Preserve the failed claim; do not rename it as a discovery |
| A prediction-dependent reference has an extra feedback path | Exact graph fact in our specified surrogate | Test whether removing it changes the actual deterioration |
| The extra path causes most of the observed risk increase | Untested hypothesis | Run the frozen-input control with the preregistered risk gate |

## The remaining distinction, derived for our model

Let q_theta(x)=f_theta(x,0). Our reference target is T_phi(x,q_theta(x)); an image-only teacher instead has no explicit student-prediction input. Holding phi fixed gives, for fixed x and fixed model state,

    d/dtheta T_phi(x,q_theta(x)) = A J,

where A=partial_q T_phi and J=partial_theta q_theta. Thus fixed reference weights need not imply fixed target values in this dual-input graph. Stopping the reference output changes the backward rule but still recomputes those target values after each student update. This chain-rule fact is already implicit in the proposal; it is not presented as a new theorem.

Define c(x)=q_theta0(x), computed once and held fixed. The realizable control uses target T_phi(x,c(x)) and residual r_c=T_phi(x,c(x))-q_theta(x), with student velocity

    F_c = E[J^T r_c].

The input-gradient and stopped-output rules coincide for this control because c is constant. At matched initialization c=q_theta0, its forward target and student velocity equal those of the original stopped-output rule. This is the clean primary comparison; comparison with the original full-gradient rule changes the initial velocity as well.

Using the cycle-5 notation at initialization, the new product-rule blocks are

    S_c0 = -E[J^T J],  T_c0 = E[J^T L].

The stopped-output endogenous blocks were S_s0=-E[J^T(I-A)J], T_s0=E[J^T L]. Their residual-prefactor student remainders agree initially, because q, J, and r agree, and both teacher remainders are zero. Hence the exact initial full-block difference is

    S_s - S_c = E[J^T A J],  T_s - T_c = 0.

This is an algebraic prediction to verify numerically before trajectories. Its sign is not prescribed for general A, and a matrix difference alone does not prove the risk hypothesis. With fixed reference weights, the control has constant targets on the repeated batch; its gradient-flow squared discrepancy is non-increasing. That still does not guarantee label correctness or protect evaluation risk. With EMA the target changes through phi and no such monotonicity is asserted.

## Reviewable next experiment

The accompanying `NEXT_EXPERIMENT_PROTOCOL.md` fixes the budget, six configurations, primary comparison, risk criterion, numerical checks, and rejection rule before any control results. It uses the existing five checkpoints and source-distribution data, with labels evaluation-only, and accounts for the 128-number cache in every condition. It is a mechanism diagnostic on a repeated batch, not a proposed online algorithm or a benchmark baseline claimed to reproduce RMT/PeTTA.

If the primary risk gate fails, demote the changing prediction input as the main cause here; do not tune the memory, seed eligibility, reference speed, or horizon until it succeeds. If it passes, retain only a conditional candidate question: how does the extra input path interact with weight averaging to amplify errors around useful predictions? A first-paper package would still require a predictive mechanism, a justified correctness regime, implementation/loss robustness, and a broader novelty check. Retention, recovery and distribution drift remain later questions.
