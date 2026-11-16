# IT³ update audit and narrowly scoped theory opportunities

## Which paper version and learning setting should the proposal use?

### Takeaway
Use the published ICML 2025 citation, retaining v1 only when discussing the supplied reading. Online adaptation and a moving reference already belong to IT³; neither is a new thesis contribution.

### Cited Findings
- Publication: Durasov, Shocher, Oner, Chechik, Efros, and Fua, *IT³: Idempotent Test-Time Training*, ICML 2025, PMLR 267, pp. 14867–14883. — [Published record](https://proceedings.mlr.press/v267/durasov25a.html)
- arXiv records v1 on 5 October 2024 and v2 on 25 May 2025; the latter is marked accepted at ICML 2025. — [Version history](https://arxiv.org/abs/2410.04201)
- In v1, Sec. 3.2 Eq. (5) fixes the second network's parameters; Sec. 3.3 Eq. (6) replaces it with an EMA model and carries adaptive parameters between test episodes. It motivates this with mostly smooth shifts and explicitly accepts deteriorating original-distribution performance. — [v1, Sections 3.2–3.3](https://arxiv.org/html/2410.04201v1)
- v2 retains these mechanisms as Eqs. (6) and (7). Its Appendix C discusses gradient pathways but does not distinguish frozen parameters from a detached output rigorously; the code audit below resolves particular implementations. Appendices B–C offer projection/consistency reasoning, not a proved stability or retention theorem. — [v2, Sections 3.2–3.3 and Appendices B–C](https://arxiv.org/html/2410.04201v2)

### Inferences
- Recommended proposal wording: “We study a smooth surrogate of the paper-specified online IT³ dynamics, with gradient paths checked against the regression implementation.” Avoid describing a detached-output student/teacher system as the established IT³ implementation.
- The research setting should explicitly add retention/recovery as an evaluation objective, rather than imply it was the original online IT³ objective. Distinguish current-task performance, source retention, previous-target retention, and recovery after a return.
- The paper's idempotence intuition needs mathematical qualification. A single condition at the model's initial prediction does not establish idempotence on every auxiliary input. For linear maps, idempotence alone is a projection, but orthogonality additionally requires self-adjointness. These are elementary mathematical observations, not proposed contributions.

### Gaps
- This pass read arXiv v1 and v2 online; it did not extract or hash the user’s local PDF. The supplied file's identification as v1 comes from the existing proposal/context and should not be represented as independently byte-verified here.
- The published PMLR record was verified. Its downloadable PDF was inaccessible to the web fetcher, so this pass does not assert byte identity between the proceedings PDF and arXiv v2.
- No numeric EMA coefficient, exact update ordering, optimizer reset policy, or online EMA experiment implementation was verified from the paper description. These must be specified as choices in a new implementation.

## What backward paths are actually present in the official code?

### Takeaway
The regression implementations preserve input gradients through the reference while excluding its weights from the student optimizer. The public code is heterogeneous: the CIFAR notebook and the linked library implementation use shared trainable models for both passes, so a single universal implementation claim would be false.

### Cited Findings
- The official repository is `nikitadurasov/ittt`; the inspected immutable revision is `100524001d890efc08911f3cba8f45f51f4aa7d1`. Repository files were retrieved from the public GitHub API/raw endpoints on 21 October 2026. — [Pinned repository](https://github.com/nikitadurasov/ittt/tree/100524001d890efc08911f3cba8f45f51f4aa7d1)
- **Tabular:** zero-based notebook cell 13, source lines 34–38 passes the student's first prediction into `model_original.forward(x,y1)` without detach and uses mean absolute difference. Lines 41–58 reset student weights per batch and backpropagate; lines 61–64 construct Adam from student parameters only. The optimizer is constructed outside the batch loop, so resetting weights does not itself reset Adam state. — [Pinned tabular notebook](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb)
- **Airfoil:** zero-based cell 24, source lines 5–14 injects the undetached student output (scaled by 500) into the graph input and evaluates the original model. Lines 21–38 use mean absolute discrepancy, per-batch weight reset, and Adam on the student only. — [Pinned airfoil notebook](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/airfoils/train_evaluate.ipynb)
- **Age:** zero-based cell 34, lines 40–59 inserts undetached predictions in an input channel and evaluates `old_network`; the discrepancy is mean absolute difference. Lines 107–113 disable BatchNorm tracking, freeze its affine parameters, and construct Adam from the adaptive network's parameters. `.cpu()` is used before backward but is not detach. — [Pinned age notebook](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/age/eval_ittt.ipynb)
- **CIFAR discrepancy:** zero-based cell 23, lines 15–21 invokes `f_ttt.f2` in both prediction passes and minimizes Jensen–Shannon divergence; the passed reference `f` is used for resetting, not the second adaptation forward call. — [Pinned CIFAR notebook](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/cifar/cifar_ittt.ipynb)
- **MNIST:** zero-based cell 6, lines 8–14 uses a random one-hot initial signal, then evaluates the reference's `f2` on both student features and student predictions. The output-to-reference-input path is undetached, but this graph has an additional feature path beyond the proposal's simple `T(x,q)` surrogate. — [Pinned MNIST notebook](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/mnist/mnist_ittt.ipynb)
- **Linked library:** the official README recommends `torch-ttt`. In its rendered `IT3Engine.ttt_forward`, lines 206–227 of the documentation page reuse `self.model`, embed the undetached first prediction, and compute the discrepancy after the second call. No fixed copy or EMA appears in that function. Default discrepancy is MSE. This is a separate implementation snapshot, not pinned to the original experiment commit. — [Official recommendation](https://github.com/nikitadurasov/ittt); [Rendered library source](https://torch-ttt.github.io/_modules/torch_ttt/engine/it3_engine.html)

### Inferences
For a deliberately chosen smooth squared surrogate, define `q=q_theta(x)`, `r=T_phi(x,q)-q`, and `D=(1/2) E ||r||²`. Then direct chain rule, holding phi fixed, yields

`G_frozen = E J_q^T (J_T,q-I)^T r`.

Detaching the whole reference output yields a different field,

`G_detached = -E J_q^T r`.

These formulas in the existing proposal are correct for their declared surrogate. They are calculations, not newly discovered general results. For the mean absolute discrepancy used in regression notebooks, replace the residual factor by a suitable subgradient, with averaging/scaling constants. For Jensen–Shannon loss use its two argument derivatives and any feature path. Freezing parameter updates is distinct from suppressing gradient computation; an unstepped reference may still accumulate gradients without changing weights.

The regression notebook examples support retaining the reference input Jacobian in the toy model. They do not justify assuming that every IT³ implementation has exactly that graph. Actual experiment reproduction should pin one task/commit, audit normalization state, distinguish training and evaluation mode, and make optimizer-state reset policy explicit.

### Gaps
- A repository-wide scan of `.py` and notebook source for `EMA`, `ema_`, `moving.average`, `AveragedModel`, and `.mul_(` found no online reference-update implementation. The sole moving-average hit was a different precision-statistics comment in `exps/age/models.py`. This bounded textual check is evidence of missing public reproduction details, not proof no EMA implementation exists anywhere.
- These were static source audits, not executed experiments. Forward internals were not exhaustively traced across every architecture. The code evidence is strongest for explicit undetached calls and optimizer membership.
- Notebook locations above are zero-based JSON cell indices and one-based lines within cell source, not raw `.ipynb` JSON line numbers.

## What could support a thesis or first paper without overstating novelty?

### Takeaway
A credible first target is a precise local stability boundary for a specified coupled update, followed by a retention counterexample or measured recovery effect. Generic gradient alignment, online adaptation, EMA teachers, and preventing test-time forgetting already have prior art.

### Cited Findings
- Sun et al., ICML 2020, Theorem 1 assumes the main loss is differentiable, convex, beta-smooth, both task gradients have norm at most G, and their inner product exceeds epsilon. A step of size epsilon/(beta G²) then improves the main loss. The paper also includes online carry-over and gradual-shift experiments. Thus “adaptation works when gradients align” and “online TTT under smooth shifts” are not new. — [Original TTT paper, Theorem 1, p. 8; online method p. 2](https://proceedings.mlr.press/v119/sun20b/sun20b.pdf)
- EATA already addresses degradation of in-distribution performance during test-time adaptation using sample selection and weight-preservation regularization. A generic “TTA without forgetting” proposal is insufficiently differentiated. — [EATA, ICML 2022](https://proceedings.mlr.press/v162/niu22a.html)

### Inferences
The following are candidate research tasks, not verified literature gaps or novelty claims:

1. **Coupled stability and the actual backward rule.** For fixed distribution and an explicit residual model, study `theta_dot=-grad_theta D`, `phi_dot=kappa(theta-phi)`. At a zero-residual equilibrium with theta=phi, let `H=E(J_r,theta^T J_r,theta)` and `K=E(J_r,theta^T J_r,phi)`. The linearization is the block matrix `M=[[-H,-K],[kappa I,-kappa I]]`. Derive conditions for its spectrum and compare the detached rule's different linearization. Positive semidefinite H alone does not settle the coupled dynamics. If M is Hurwitz, the standard Lyapunov equation gives a local quadratic certificate; that existence statement is classical, so the potential contribution must be a mechanism-specific boundary, interpretable condition, or unexpected failure regime. Handle zero modes/non-isolated equilibria rather than silently calling them asymptotically stable points.

2. **Tracking versus retention under a switch.** Analyze one shared-parameter model under A→B→A, keeping labels evaluation-only. Report both adaptation discrepancy and earlier-task risk. The identity `dR_A/dt=-<grad R_A,G>` is elementary and aligns with existing TTT gradient-correlation theory. A useful result would identify exactly how reference lag changes this quantity in a nontrivial coupled model, show when stable tracking still harms old risk, and compare recovery against fresh source restart under equal update/data budgets. Recovery by retraining alone does not establish intact representations. Adding replay or source regularizers changes the learner's information budget and must be stated.

3. **Residual correctness and identifiability.** A correct reference fixed point and a contractive neighborhood provide a possible route from consistency residual to prediction error; absent such structure, zero residual can correspond to wrong predictions. Prove only conditional bounds and build a small counterexample separating residual decline, dynamical stability, and task correctness. To be publishable, this must exceed a one-line observation by connecting the missing assumption to measurable Jacobians/teacher lag and demonstrating an informative intervention. Unlabeled streams with identical input marginals but conflicting labels also rule out unconditional correctness/retention guarantees; any positive theorem needs assumptions that distinguish these cases.

Useful next deliverable: one explicit smooth model tied to the audited regression graph, a mathematically checked two-timescale stability result or counterexample, a discrete-step counterpart with specified EMA ordering, and one controlled switching experiment. A toy model alone is a first research milestone; it is not automatically a paper.

### Gaps
- This bounded pass is not an exhaustive 2026 novelty search. It cannot establish that no one has analyzed this particular block system or derived related EMA tracking bounds.
- CoTTA and additional continual-adaptation theory remain necessary before finalizing the related-work section; the attempted primary CoTTA page returned a fetch error in this pass. Do not infer absence of prior art from that failure.
- No theorem, numerical experiment, or neural-network result was produced here. The analysis above gives a checkable starting point and restrictions on claims.
