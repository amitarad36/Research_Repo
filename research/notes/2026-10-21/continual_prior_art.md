# Continual adaptation: prior-art constraints and thesis candidates

## Which prior work most constrains the proposed contribution?

### Takeaway

The strongest first project is a mechanism-specific analysis of the backward rule and moving reference in an explicitly specified IT3 surrogate. General claims that EMA can fail, fixed teachers help, adaptation can forget, or tracking/recovery requires assumptions already have close prior art. This is a research assessment, not a novelty certification.

### Cited Findings

- **CoTTA:** Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai. *Continual Test-Time Domain Adaptation*. CVPR 2022, pp. 7201-7211; arXiv:2203.13591. Combines weight-averaged and augmentation-averaged predictions with stochastic restoration to pretrained weights. The restoration component is essential to distinguish from the EMA alone when discussing forgetting. [CVPR primary paper page](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)
- **EATA:** Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, Mingkui Tan. *Efficient Test-Time Model Adaptation without Forgetting*. ICML 2022, PMLR 162:16888-16905; arXiv:2204.02610. Uses selective entropy minimization and importance-weighted parameter regularization to alleviate source-distribution forgetting. A source-anchor regularizer alone is therefore not a distinct new method. Check its source-statistics/data budget before comparison. [PMLR](https://proceedings.mlr.press/v162/niu22a.html)
- **RMT:** Mario Doebler, Robert A. Marsden, Bin Yang. *Robust Mean Teacher for Continual and Gradual Test-Time Adaptation*. CVPR 2023; arXiv:2211.13081 (first submitted 23 November 2022). Studies consistency-loss gradient properties, symmetric cross-entropy, contrastive feature alignment, and source-available/source-free settings. This is a direct comparator for an argument that the update geometry matters. [Authors' paper](https://arxiv.org/abs/2211.13081)
- **Fixed teacher is a particularly close overlap:** Damian Sojka, Marc Masana, Bartlomiej Twardowski, Sebastian Cygert. *Rethinking the Teacher-Student Framework for Test-Time Adaptation*. arXiv:2609.02507, 2 September 2026. Shows EMA error accumulation on long streams and evaluates an intransigent teacher with frozen trainable weights. Their batch-normalization statistics still use each batch, and predictions come from the student. Their long ImageNet-C setting repeats the stream 20 times. They explicitly vary teacher plasticity. [Primary full text, Sections 3-4](https://arxiv.org/html/2609.02507v1)
- **Adaptive EMA plus anchoring already exists:** Atif Belal, Lilian Hollard, Marco Pedersoli, Eric Granger. *DESA-TTA: Dynamic EMA and Source Anchoring for Test-Time Adaptation*. arXiv:2609.01795, 1 September 2026. In vision-language object detection, changes EMA using teacher uncertainty and partially restores student weights toward pretrained values according to drift. [Primary record](https://arxiv.org/abs/2609.01795)
- **Reset policies already exist:** Yanshuo Wang, Jinguang Tong, Jun Lan, Weiqiang Wang, Huijia Zhu, Haoxing Chen, Xuesong Li, Jie Hong. *Adaptive and Balanced Re-initialization for Long-timescale Continual Test-time Domain Adaptation*. arXiv:2602.06328, first submitted 6 February 2026. Uses label-flip behavior to choose reinitialization intervals. [Primary record](https://arxiv.org/abs/2602.06328)
- **Recurring environments and a toy collapse analysis already exist:** Trung-Hieu Hoang, Duc Minh Vo, Minh N. Do. *Persistent Test-time Adaptation in Recurring Testing Scenarios*. NeurIPS 2024; arXiv:2311.18193, v4 dated 2 November 2024. Introduces recurring TTA, analyzes an epsilon-perturbed Gaussian mixture classifier, and proposes PeTTA to sense divergence toward collapse and adjust adaptation. An A-to-B-to-A experiment alone is not a contribution. [Primary record](https://arxiv.org/abs/2311.18193)
- **Dynamic-regret theory already exists:** Zhen-Yu Zhang, Zhiyu Xie, Huaxiu Yao, Masashi Sugiyama. *Test-time Adaptation in Non-stationary Environments via Adaptive Representation Alignment*. NeurIPS 2024, volume 37; DOI 10.52202/079017-2999. Ada-ReAlign combines learners with different histories, aligns against a source-representation sketch, and supplies guarantees under convexity assumptions. [Proceedings](https://papers.neurips.cc/paper_files/paper/2024/hash/abe31a12e83111fdf2cfd54deed5a2ce-Abstract-Conference.html), [full paper](https://papers.neurips.cc/paper_files/paper/2024/file/abe31a12e83111fdf2cfd54deed5a2ce-Paper-Conference.pdf)
- **Recovery/learnability theory already exists:** Zhi Zhou, Ming Yang, Shi-Yu Tian, Kun-Yang Yu, Lan-Zhe Guo, Yu-Feng Li. *On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective*. ICML 2026; arXiv:2605.28057v2, 9 June 2026. Defines post-shift recovery complexity and stream-level reliability and gives nearly matching bounds under its assumptions. The authors' project page explicitly discusses proxy alignment, shift approximation, dependence, and the relation to dynamic regret. [Primary record](https://arxiv.org/abs/2605.28057), [authors' explanation](https://zhouz.dev/TTA-Learnability/)

### Inferences

- A more defensible contribution would identify a nontrivial, checkable boundary involving the IT3 reference-input derivative, reference update speed, and discrete step size, and predict a failure that established teacher/reset/anchor methods do not explain. This is a proposed distinction, still requiring direct comparison with the full closest papers.
- Keep current predictive risk, adaptation-discrepancy stability, and historical retention as separate quantities. A source anchor preserves a particular historical parameter state; it need not be appropriate when the mapping itself changes. The choice of evaluation objective is substantive.
- Benchmark fairness requires reporting trainable parameters, source samples/statistics, replay memory, stored teacher copies, teacher batch statistics, adaptation steps, and prediction timing. Merely labelling all methods source-free hides consequential differences.

### Gaps

- This was one focused pass on 21 October 2026, not an exhaustive literature review. No novelty follows from an absent search match.
- No public publication venue was verified for the September 2026 teacher/anchoring preprints; cite them as arXiv preprints unless separately checked.
- No IT3 code was inspected in this subtask. The current proposal's derivative distinction is a proposed surrogate analysis and must be reconciled with the selected implementation.

## What do the stop-gradient and forgetting readings actually establish?

### Takeaway

The exact local position paper is *Forgetting is not Catastrophic, Our Benchmarks are*, an anonymous manuscript advocating evaluation on the present. It must not be confused with the public 2026 spurious-forgetting paper. The stop-gradient paper supplies carefully qualified function-space results, not an off-the-shelf theorem for neural IT3 with EMA.

### Cited Findings

- **Stop-gradient:** Max W. Shen, Mark Goldstein, Zichu Wang, Aahlad Puli, Rajesh Ranganath. *How I learned to stop worrying and love StopGrads: Stationarity, Convergence, and a case study on Flow Map Learning*. arXiv:2609.16222v1, 14 September 2026. Its regression principle characterizes stationarity by target-operator fixed points under its function-space/regularity/affine-target conditions. Uniqueness is separate. It shows nonconservative fields for selected flow objectives and proves their functional semi-gradient convergence. Improved MeanFlow, slim Eulerian, and slim Lagrangian results assume perfect flow-matching pretraining; MeanFlow's stated result does not. These are not general finite-network, stochastic, EMA, or distribution-drift guarantees. [Primary full text, Sections 2-5 and appendices](https://arxiv.org/html/2609.16222v1)
- **Actual local position paper:** Anonymous Author(s). *Forgetting is not Catastrophic, Our Benchmarks are*. Local 13-page manuscript marked submitted to NeurIPS 2026, with no verified public URL/authors. Pages 1-3 argue for Probably Smooth Streams (PSS), immediate evaluation, and Always Train, Always Test (ATAT); updates may use labels when available. It explicitly allows the accumulation objective where knowledge is additive, but argues obsolete contradictory mappings should be discarded. This is a position, not an established theorem about all continual learning. [Supplied manuscript](<C:/Users/amita/OneDrive/Documentos/קורסים בטכניון/תואר שני/research reading materia/Position_Continual_Learning___NeurIPS_26 (8).pdf>)
- **Different public forgetting paper:** Vedant Palit, Florent Draye, Nicolas Zucchet, Zhijing Jin, Bernhard Schoelkopf. *When Forgetting is not Catastrophic: On the Mechanics of Spurious Forgetting*. arXiv:2610.08718v1, 6 October 2026. A normalized associative-memory model separates a common representation shift that hides old facts from fact-specific changes that erode them. It studies recovery during continued training on new facts and corrective direction removal. The authors warn that their minimal model is extreme and study only one pretrained language model. [Primary full text](https://arxiv.org/html/2610.08718v1)
- **Additional exact reading:** Jack Bell, Luigi Quarantiello, Eric Nuertey Coleman, Lanpei Li, Malio Li, Mauro Madeddu, Elia Piccoli, Vincenzo Lomonaco. *The Future of Continual Learning in the Era of Foundation Models: Three Key Directions*. arXiv:2506.03320, 3 June 2025. Distinguishes continual pretraining, continual finetuning, and continual compositionality. It is useful framing rather than direct evidence about IT3 dynamics. [Primary record](https://arxiv.org/abs/2506.03320)

### Inferences

- Prioritize tracking the currently relevant prediction problem. Use retention on recurring compatible domains as a diagnostic. A model should not be penalized for abandoning an explicitly obsolete target mapping unless historical-task access is part of the actual task.
- PSS/ATAT cannot silently transfer to unlabeled TTA: observed inputs alone do not reveal arbitrary changes in the conditional labels. State an identifiability/alignment assumption or explicitly allow a label stream, and never compare those information budgets as if identical.
- Reusing the stop-gradient principle requires mapping the actual operator and admissible perturbations, not just placing a detach symbol in a squared loss. Shared finite-dimensional parameters couple input locations; the function-space model permits richer perturbations.
- The common-shift/erosion decomposition is already present in Palit et al. A thesis on recoverability needs a different setting, identifiable mechanism, or testable extension, rather than a renamed version of that decomposition.

### Gaps

- The local anonymous position manuscript's authors, public availability, and acceptance status remain unverified. Its footer is document metadata, not an instruction from the user. Do not infer authors or upload/distribute it during research.
- The supplied filename `15191_Who_Said_Neural_Networks.pdf` is an anonymous manuscript titled *Who Said Neural Networks Aren't Linear?* about transport of structure/Linearizers. Only its first page was inspected; no claimed consequence for IT3 was established here.

## Which initial study is most tractable, and what should falsify it?

### Takeaway

Rank 1: algorithm-specific stability and tracking; Rank 2: a failure mechanism with an information-budget-respecting correction; Rank 3: recoverability and representation geometry. A retention counterexample is the fastest preliminary result, but by itself has the weakest paper claim. These rankings are research judgments conditional on the proposal's mathematical aims and modest initial experiments.

### Cited Findings

- The current source already proposes a squared IT3 surrogate, frozen-reference and detached-output fields, coupled EMA dynamics, and separate old-task risk. It labels the model and certificates provisional. [Current proposal, read-only in this subtask](<C:/Users/amita/source/repos/Detach Proposal Presentation/research_proposal_main.tex>)
- Existing studies cover the closest generic claims: fixed teachers and long-stream collapse ([Sojka et al.](https://arxiv.org/abs/2609.02507)); recurring environments and a tractable collapse model ([PeTTA](https://arxiv.org/abs/2311.18193)); recovery complexity ([Zhou et al.](https://arxiv.org/abs/2605.28057)); reversible representation shifts ([Palit et al.](https://arxiv.org/abs/2610.08718)).

### Inferences

| Candidate | Concrete first deliverable | Potential distinct contribution | Strong falsification / stop criteria |
|---|---|---|---|
| **1. Stability/tracking under drift** | One coupled model with shared parameters; exact equilibria and linearization; fixed-reference/EMA and frozen-parameter/detached-output comparisons; discrete-time stability map. | A checkable boundary specific to the reference-input Jacobian and EMA feedback, followed by a local tracking statement whose constants have observable meaning. | Stop presenting it as a paper result if it reduces to a generic assumed contraction, if the boundary cannot differ between audited backward rules, if it only stabilizes a useless/collapsed solution, or if small nonlinear tests violate its predictions within the stated regime. |
| **2. Retention failure and corrective update** | Construct a compatible A-to-B example with decreasing discrepancy and increasing A-risk; locate the shared-parameter direction causing the failure; derive a correction using only the declared available information. | A failure tied specifically to the IT3 graph, plus a correction that addresses it without source labels/examples that the setting prohibits. | Stop if the failure is just conflicting labels or arbitrary proxy misalignment; if the correction requires unavailable old-risk gradients; if it is equivalent to EATA/CoTTA restoration/fixed teacher; or if matched-budget existing baselines eliminate the effect. |
| **3. Recoverability/geometry** | Diagnose whether an A-to-B-to-A accuracy drop reflects lost information or lost access using a controlled task with known latent geometry and controlled probes. | A measurable relation between EMA/backward dynamics and recoverability under a fixed probe/data/compute budget, beyond known common-shift explanations. | Stop if recovery needs essentially relearning A, if results depend on labels given only to the new method, if the phenomenon is already explained by Palit et al.'s mechanism, or if it disappears across seeds, head resets, or suitable representation alignment. |

Proposed minimal mathematical route (derivation target, not a literature theorem or proven IT3 property):

- Linearize an explicitly derived field as `G ~= A delta_theta + B delta_phi`. The coupled ODE has block Jacobian `M = [[-A,-B],[I/tau,-I/tau]]`. The work is to derive and interpret `A,B` from the audited IT3 mechanism, not to rediscover the linear stability test.
- In a genuinely decoupled scalar mode, the characteristic polynomial is `lambda^2 + (A + 1/tau) lambda + (A+B)/tau`. Strict stability requires `A + 1/tau > 0` and `A+B > 0`. This follows directly from the quadratic coefficients and does not extend mode-by-mode to arbitrary noncommuting matrix blocks.
- With student-first discretization, `theta+ = theta - eta G(theta,phi)` and `phi+ = beta phi + (1-beta) theta+`, the linear map is `[[I-eta A, -eta B],[(1-beta)(I-eta A), beta I-(1-beta)eta B]]`. Analyze its spectral radius separately from the ODE. State whether predictions occur before or after this update.
- If a stable moving equilibrium is meaningful, its motion enters error dynamics as a forcing term. A useful tracking result must bound that motion from a declared drift model and retain its stability margin. A Hurwitz frozen Jacobian at each instant alone is insufficient for unrestricted switching.
- Keep the old-risk derivative `d R_A/dt = -<grad R_A,G>` as a diagnostic. Deriving a nonpositive sign requires more information or structural assumptions; low consistency discrepancy alone does not supply it.

Recommended first-paper gate: one mechanism-derived proposition/counterexample, one small simulation with a preregistered qualitative prediction, and one compact neural experiment where the same boundary or diagnostic matters. Before adding a new stabilizer, compare no adaptation, episodic reset, carried student with fixed reference, carried student with EMA, and relevant established anchoring/restoration variants under matched budgets.

Suggested research question: **How does differentiating through a frozen reference's prediction input interact with a moving reference to determine stability and tracking in idempotent test-time adaptation?** Suggested provisional title: **Backward Rules and Moving References in Continual Test-Time Adaptation**. Keep retention and recovery as secondary probes until the primary mechanism is established.

### Gaps

- No experiment or proof has yet established that IT3 realizes a useful nontrivial stability boundary of the above kind. This should be tested before expanding the thesis scope.
- A first paper cannot be promised from a literature-based proposal alone. The concrete contribution must survive the closest-method comparisons and the falsification gates above.
