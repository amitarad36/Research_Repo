# Research state: backward rules and moving references

Updated: 22 October 2026 (Asia/Jerusalem), cycle 4. Working document: ../research_proposal_main.tex, edited in place. Working title: **Backward Rules and Moving References in Continual Test-Time Adaptation**. Preserve Amit Arad, advisor Assaf Shocher, and acknowledgment to Shoham Grunblat. Existing automation runs in this chat daily at 09:00 and 17:00 Asia/Jerusalem; do not duplicate it.

## Current decision

Current predictive risk is primary. Retention and recovery are secondary when old tasks remain relevant and compatible. The private anonymous position manuscript *Forgetting is not Catastrophic, Our Benchmarks are* remains framing context, with unverified authors/public status; it is not Palit et al.'s public paper.

Rank 1: explain the measured student/reference feedback and its effect on useful predictions. A stationary synthetic test now shows finite-time amplification and risk deterioration under fast EMA, with fixed-reference controls largely preserving initial accuracy. However, the predeclared claim that output detachment consistently amplifies more FAILED. Do not sell the tangent implementation, generic EMA failure, or fixed-teacher improvement as novelty. Rank 2: a compatible-task failure and justified correction, with explicit retained-information budget. Rank 3: recovery under matched new-data/probe/update budgets, after a distinct mechanism is established.

The strong rotational construction from cycle 2 remains an existence result; it did not appear in the first learned-model screen. Projection-like geometry is also unsupported. Avoid tuning until a desired geometry/effect appears. A new method or distribution-shift sweep is premature.

## Exact next action

Question: does ordinary positive student/reference feedback explain the measured growth, or are residual-weighted terms or parameter-Jacobian mismatch essential?

Use saved cycle-3 checkpoints and full derivative blocks; no retraining. Define J=partial_theta f(x,0), L=partial_phi f(x,q) holding q fixed, B=I-partial_q f_phi(x,q). Compute the zero-residual blocks:

- Input-gradient: D_s F=-E[J^T B^T B J], D_t F=E[J^T B^T L].
- Stopped-output: D_s F=-E[J^T B J], D_t F=E[J^T L].

Subtract them from the independently measured exact derivatives to quantify residual terms. Separately replace L by J as an explicitly diagnostic counterfactual, not a new implemented algorithm. Restrict to the 82 non-fixed coordinates; this block is not proved controllable/reachable under the particular stream. Compare the presence and observability of positive modes in all five seeds, with 0,2,3,4 primary. Freeze quantitative tolerances and a small budget before calculations. If L=J already explains growth, reject a novel mismatch explanation. If residual terms dominate, retain the nonstationary-trajectory setting instead of assuming an exact equilibrium. Do not infer risk from sensitivity ranking alone.

Before a paper claim, compare the narrowed mechanism with exact assumptions/results in RMT, PeTTA, fixed-teacher work, and the earlier listed teacher/control analyses. Generic anchoring/restoration, Lyapunov existence, eigenvalue stability checks, and ordered tangent products are established tools. No novelty claim is currently verified.

## Completed evidence

**Cycle 4: notes/2026-10-22-cycle4/Finite_Response_Result.md.** Prespecified stationary empirical-distribution stress test: same 64 unlabeled inputs reused, independent 192-input evaluator, two backward rules, fixed reference/EMA tau=1/EMA tau=20, 1000 steps at eta=.02 (T=20), with one halved-step check. All five prior checkpoints reused without training. Two fixed perturbation directions, student-only initial perturbations +/-1e-4 and +/-1e-5, teacher initially unchanged; base copies start matched. The modal direction intentionally probes a local growing mode and is not random/typical noise.

Amplification gate PASSED in seeds 0,2,3 (stopped-output/fast-EMA modal gain >=2 with fixed-reference gain <=1.25); seed 4 gain 1.91 misses the threshold. Uniform rule-ordering gate FAILED: only seed 3 has stopped/input gain ratio >=1.25. Seed 2 input-gradient gain is 645 versus stopped-output gain 52, while stopped-output risk is worse. All eight fast-EMA primary runs meet the prespecified material-risk-deterioration criterion. Fixed-reference final NMSE .00685-.03697; fast EMA .516-249.731; slow EMA .00771-.04738, with stopped/slow seed 2 still materially worse. Slow EMA is not proved safe. Fast-EMA discrepancy increases in this test; do not claim it monotonically decreases while risk worsens.

The actual time-dependent discrete tangent recurrence predicts symmetric nonlinear prediction differences to max relative error 1.272e-5 at epsilon=1e-5. Halving the step changes final modal gains <=2.397% and NMSE <=.517%. Both numerical gates pass. These are finite-time checks, not equilibrium stability, ODE convergence, drift, or recovery results. Thirty configurations, 195.95 CPU wall seconds; all histories/checkpoints and protocol hashes retained. No paid/cluster/GPU compute or package installs.

**Cycle 3: notes/2026-10-22-cycle3/Neural_Geometry_Result.md.** Five 114-parameter tanh regressors pretrained using the chosen squared zero/label recipe. Four passed NMSE<=.05; no strong-rotation probe in any seed. Relative idempotence-defect medians .46-.94 among passing seeds do not justify a nontrivial projection assumption. Initial sampled reference-input Jacobian norms <.902 coexist with positive full coupled spectral abscissae. Checkpoints have nonzero velocities/residuals. Thirty-two first-layer auxiliary-input student weights are structurally fixed; matched EMA keeps them fixed. Full-parameter strict Hurwitz claims are inappropriate. Initial source code/data/checkpoints and exact full-field derivatives remain in that directory.

**Cycle 2: notes/2026-10-21-cycle2/Vector_Stability_Result.md.** Constructed correct, nonconstant fixed point in one smooth dual-input model. Common rotational feedback: input-gradient stability for all finite tau, stopped-output stability iff tau>10 for specified covariance. Marginal boundary is not a proved nonlinear Hopf. Noncommuting finite cases have exact rational continuous/discrete certificates; no full noncommuting interval theorem. Correct baseline and feedback were planted, not learned. Local perturbation recovery is not distribution-shift adaptation.

**Earlier work:** Scalar_Certificate_Result.md (7 October learned-predictor certificate) is separate from EMA adaptation. notes/2026-10-21/theory_targets.md and verify_affine_ema.py establish an affine diagnostic: current risk/discrepancy decline while compatible old-task risk rises; finite EMA timescale changes transients, not limit. The two rules only change speed in that example. Supporting notes: it3_audit.md and continual_prior_art.md in the same directory.

## Source audit and boundaries

IT3 arXiv v1/v2 and published ICML 2025 metadata checked; official code commit 100524001d890efc08911f3cba8f45f51f4aa7d1. Regression notebook references retain input gradients and use absolute discrepancies. CIFAR/linked library graphs differ. No universal IT3 backward graph or reproducible public online EMA ordering was established. Squared losses and student-first GD/EMA are our explicit choices. Local-PDF byte identity is unverified. No official benchmark has been reproduced.

Closest work and exact source links are in the prior notes and proposal: IT3, original TTT, CoTTA, EATA, RMT, PeTTA, Sojka fixed-teacher preprint, DESA-TTA, Ada-ReAlign, Zhou recovery complexity, OGD, Palit spurious forgetting, StopGrads. The earlier pass was bounded and does not certify novelty. Frozen parameters do not detach input gradients. Low residual does not imply correctness; input-map contraction does not itself certify coupled parameter behavior. Pointwise spectra do not justify arbitrary switching.

## Continuation and preservation

Read this state, latest log, saved source, and relevant notes each cycle. Preserve concurrent edits with content checks. Choose one substantive action and record failures as well as support. No new papers/experiments merely to expand scope. Keep labels evaluation-only after pretraining. Keep the same TeX/editor and use the built-in compiler after source edits; never build a separate PDF or open a replacement tab. Do not contact people, publish, submit jobs, or incur paid compute without a new user request. Keep this chat project's synced sources/ read-only.

Cycle-4 evidence is under notes/2026-10-22-cycle4/. Writable workspace originals: C:\Users\amita\.codex\.chatgpt-projects\g-p-6a9fa12b13bc8191acd58cbcf6c03aef\research_notes\Continual adaptation finite response. Previous useful notes and backups are preserved. No cleanup is needed and no files were removed this cycle. Pre-edit backup: archive/2026-10-22_before_finite_response.tex; state/log backups accompany it. Validation: built-in compile_latex_document returned kind=success for the existing saved source. Source SHA256: 393fb79c0dcd11f4df212299c5fa5d928e068941045cfd32ea83d43e9d449a23. Protocol/script hashes and all 30 archives verified; author/advisor/acknowledgment preserved. Existing editor kept open; no separate PDF build or new tab.
