# Frozen prediction-input control: a large effect, not a safe adapter

25 October 2026, cycle 7. **New checked synthetic experiment**, following the unchanged protocol saved in cycle 6. Primary hypothesis supported in all four primary seeds at both step sizes. Novelty remains provisional; no new literature claim or official benchmark result.

## Question and intervention

Does the student prediction supplied to the reference materially drive the stationary risk deterioration beyond averaging the reference weights? The original stopped-output velocity and the control are

`F_o(theta,phi) = mean J_theta(x)^T [f_phi(x,q_theta(x)) - q_theta(x)]`,

`F_c(theta,phi) = mean J_theta(x)^T [f_phi(x,c(x)) - q_theta(x)]`,

where `q_theta(x)=f_theta(x,0)` and `c(x)=q_theta0(x)` is cached once. Caching changes the forward target input. Detaching a freshly recomputed input/output would not implement this control. The full input-gradient original is a secondary comparison because its initial velocity differs; for the cached control both backward rules coincide.

At initialization, the targets, residuals, student Jacobians and stopped/control velocities coincide. Let `S=d_theta F`, `T=d_phi F`, and `A=d_z f_phi(x,z)` at `z=q_theta0(x)`. Differentiating both full fields, including their residual-weighted product-rule terms, gives

`S_o-S_c = mean J^T A J`, and `T_o-T_c = 0`.

The prefactor-derivative terms cancel because their initial residuals agree. The remaining target-input derivative is `AJ`. This identity is not a positivity claim, a risk bound, or an equilibrium certificate. The checkpoints have nonzero velocities.

## Unchanged protocol and information budget

- Reused the five existing 114-parameter, 4-to-16-tanh-to-2 pretrained regressors, without retraining. Seeds 0,2,3,4 are primary; seed 1 had failed the earlier fit gate and remains sensitivity-only.
- Same 64 unlabeled adaptation inputs, reused throughout; the other 192 inputs and labels are evaluator-only. NMSE divides squared Euclidean output error by evaluation-label variance 0.06919046989163671. No labels enter adaptation, cache, stopping, or parameter selection.
- Six conditions per seed: original stopped-output, original full input-gradient, cached input, each with fixed reference or EMA tau=1. Student-first GD/EMA, eta=.02 for 1000 steps, repeated at eta=.01 for 2000 steps, both T=20. All 60 trajectories completed: exactly 90,000 student updates. No tangent/perturbed runs or extra seeds.
- EMA: `theta_next=theta+eta F(theta,phi)`; `phi_next=alpha phi+(1-alpha)theta_next`, `alpha=tau/(tau+eta)`. For fixed reference alpha=1. Matched initial copies; the same 32 structurally inactive coordinates are held exactly at their initial values, matching cycle 4.
- Every condition allocates a 64x2 float64 cache (128 scalars, 1024 bytes); only the control reads it during adaptation. Two parameter copies; no replay, extra source model, or optimizer state beyond GD. Saved research/evaluation archives are not learner memory. This retained per-input information does not define a fresh-input online method.
- Histories, both parameter copies and reference targets saved every .4 time units: 51 records per trajectory. Runtime: 38.1875 CPU seconds, 42.2452 wall seconds, below the 5-minute target. No installations, paid compute, cluster jobs or GPU work.

## Primary result

The protocol required original stopped/EMA error `Re >= 2 R0` and `Re-R0 >= .01`, and cached/EMA error `max(Rc-R0,0) <= .25 (Re-R0)`, in at least three of four primary seeds at **both** step sizes, with all numerical gates passing. The denominator is the original excess error, not its total terminal error.

| Seed | Initial R0 | Original EMA, .02 | Cached EMA, .02 | Original EMA, .01 | Cached EMA, .01 | Excess removed, .02 | Both-step gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | .01234034 | 3.809762 | .196154 | 3.793542 | .196023 | 95.1595% | pass |
| 2 | .03673134 | 249.731103 | .116331 | 250.012529 | .116243 | 99.9681% | pass |
| 3 | .01722448 | 9.749481 | .074193 | 9.699401 | .074181 | 99.4146% | pass |
| 4 | .00752071 | .671780 | .083442 | .670332 | .083388 | 88.5705% | pass |
| 1, sensitivity only | .06805732 | 237.805355 | 1.548004 | 237.888662 | 1.546013 | 99.3775% | pass, excluded from count |

Primary gate: **4/4 pass**. Across primary seeds and both steps, removed excess ranges from 88.5537% to 99.9682%. This is a paired intervention on the forward prediction path with identical initialization, optimizer and reference schedule. The later parameter trajectories differ as a consequence of that intervention; it does not hold realized teacher weights equal between arms or identify an additive percentage of causation under arbitrary interventions.

Crucially, all cached/EMA primary final errors remain above their initial error (approximately 3.17 to 15.90 times the starting NMSE). Removing this path is insufficient for preservation of current accuracy. There is no useful distribution-shift adaptation, retention, or recovery result here.

## Secondary observations (descriptive)

At eta=.02, reference-target RMS change means `sqrt(mean ||target_t-target_0||^2)` over the 64 adaptation inputs.

| Primary seed | Original stopped/EMA target change | Cached/EMA target change | Original full-gradient/EMA final NMSE | Cached/fixed final NMSE |
|---|---:|---:|---:|---:|
| 0 | .529954 | .109678 | 1.703069 | .012076 |
| 2 | 4.113175 | .075924 | 54.111342 | .035250 |
| 3 | .906940 | .063241 | 3.090701 | .015607 |
| 4 | .229795 | .074027 | .516327 | .006852 |

Fixed weights plus cached input produce exactly constant reference targets in all saved records. Original targets can move despite fixed weights; maximum saved RMS movement across all five seeds and both original rules at the main step is .0183283. This verifies the forward distinction, not correctness of those targets.

Cached/EMA discrepancy is lower at the terminal point than initially in every seed, while predictive error rises. This is an endpoint comparison, **not a monotonicity claim**. Original fast-EMA discrepancy rose in the prior runs, which are reproduced here; do not rewrite that earlier result as decreasing discrepancy. Both rules coincide only for the cached-input control.

## Verification and provenance

- Model, shared data and all five checkpoint hashes match cycle 4. Every original arm's main/fine endpoint NMSE reproduces its saved baseline exactly (maximum difference 0).
- Initial stopped/control targets and velocities match exactly. Full control loss-gradient central-difference error <=2.047e-11. Independent full student-block difference versus explicit sample-loop `J^T A J` error <=6.510e-11; reference-block difference exactly zero. All are inside the prespecified tolerances.
- Maximum step-size NMSE difference, denominator `max(abs(fine NMSE), .01)`, is .5163211% across all conditions/seeds, below 5%. No failed checks or retuning.
- Independent `verify_saved.py` recomputed all five metrics in all 3,060 saved records from the stored parameters/targets: exact agreement; 32 inactive coordinates remain exact. It performs no additional adaptation updates. Verified all 30 trajectory archives, five derivative archives, their hashes, and cache identity.
- Protocol SHA256: `86eca22e2ec0a9c4a1e8ef85b46ff7edb136a77593dce321733e6c60966a0a86` (byte-identical to cycle-6 NEXT_EXPERIMENT_PROTOCOL.md).
- Results SHA256: `bec2eb7dab4713c4ba7910d5ec5bf40642e4346af7e816af01b2c584faf496ac`.
- Source/data/script/checkpoint/archive hashes, Python 3.12.14, NumPy 2.3.5 and platform/thread details are in `frozen_input_results.json`; independent verifier hash and maxima are in `verification_summary.json`. Run `frozen_input_control.py` in a fresh directory beside a copy of PROTOCOL.md; it refuses to overwrite existing results. Model/data are loaded from sibling cycle 3 when installed in the research notes, otherwise from the original absolute location. The immutable cycle-4 report is also required. No downloaded code executed.

## Research decision and conditional thesis package

**Candidate thesis question:** When does feeding a student's changing prediction into a moving reference amplify predictive drift, and when can its effect be predicted separately from the reference-weight update and backward detachment?

This experiment makes that narrower question worth pursuing. It does not justify proposing cached-input adaptation as a new method. The closest-work map in `../2026-10-24-cycle6/Prior_Art_Claim_Map.md` still rules out generic EMA error feedback or fixed-teacher benefit as sufficient contributions. The audited image-teacher graphs lack this prediction input, but graph difference alone does not certify novelty.

A conditional first-paper package would need: (1) an interpretable, checked condition predicting the effect of the input path, not merely retrospective spectra; (2) the paired intervention and a reproducible failure case where removing it still harms accuracy; (3) robustness to an audited regression loss/implementation, with clear EMA/optimizer choices; and (4) a fresh closest-work comparison of the actual claim. Current evidence supplies (2) and a motivating derivative identity, not the whole package.

**Next single action:** test whether an untuned affine approximation from the initial full derivatives predicts early original-versus-cached output separation. Its analysis protocol is saved in `NEXT_ANALYSIS_PROTOCOL.md`; no new prediction calculation has been run. This is a prospective analysis of existing trajectories after terminal results are known, not a preregistered independent experiment. Failure would demote a cheap initial-geometry explanation, not erase the intervention effect. Long-horizon inference from a passing local gate is prohibited. Do not add a stabilizer, widen seeds, alter the loss or begin a drift benchmark in the same analysis.
