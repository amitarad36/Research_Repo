# Observable amplification is present; uniform backward-rule ordering is not

22 October 2026, cycle 4. Completed prespecified synthetic experiment. This is finite-time evidence in one stationary empirical-distribution setting, not a theorem about IT3, asymptotic instability, a general recommendation about EMA, or a novelty claim.

## Decision

The amplification gate passes (3/4 primary seeds), the proposed backward-rule ordering fails (1/4), and numerical/discretization gates pass. Keep the moving-reference effect as a phenomenon to explain. Demote the claim that the stopped-output rule uniformly amplifies more than the input-gradient rule. A successful tangent implementation is a verification tool, not a research contribution in itself; generic EMA failure and fixed-teacher improvement already overlap prior art.

All four fit-passing networks lose substantial current predictive accuracy under fast EMA with either backward rule, on stationary inputs. Fixed-reference comparisons remain close to their initial risk. Slow EMA is less damaging over this finite horizon but is not established as safe: it still degrades seed 2 under the stopped rule, and seed 1 (the sensitivity case) more strongly. No perpetual-retention, convergence, or tracking conclusion follows.

## Fixed protocol and information budget

PROTOCOL.md was saved before trajectories, with its hash recorded in finite_response_results.json. No pretraining, model selection, extra seeds, favorable horizon selection, or parameter tuning occurred in this cycle. All cycle-3 checkpoints are reused; seeds 0,2,3,4 are primary because they passed that prior fit gate, and seed 1 is fully reported as sensitivity evidence.

The first 64 saved held-out inputs form an unlabeled adaptation batch, reused at every update. The remaining 192 inputs and their labels are used only for evaluation. This is repeated adaptation to a stationary empirical distribution, not a fresh-sample stream. The model, training data, and field definitions come from notes/2026-10-22-cycle3/. The nominal IT3 source remains arXiv:2410.04201v2 Section 3.1 Eq. (2), Eqs. (6)-(7), https://arxiv.org/html/2410.04201v2 ; the squared surrogate, architecture, optimizer and EMA ordering are project choices, not an official implementation reproduction. No new literature claims or novelty search were made in this cycle.

Six configurations per seed: both backward rules, each with fixed reference, EMA tau=1, or EMA tau=20. Base copies start matched. Student-first gradient descent uses eta=.02, alpha=tau/(tau+eta), 1000 updates (T=20). The no-adaptation control has constant initial risk and unit perturbation gain. No training labels, stored source data, restoration, augmentation, or adaptive optimizer state enter adaptation. Each dynamic configuration retains student and reference parameter copies.

Two unit-norm perturbation directions are fixed before trajectories: the student component of the dominant stopped-output/tau=1 local mode, and one seeded Gaussian direction. Both use only the 82 coordinates not structurally frozen by the zero auxiliary input. The modal probe is intentionally adversarial to the local dynamics and is identical across all configurations of a seed; it is not typical noise. All perturbed references start at the original checkpoint while students start at p +/- epsilon*d, introducing a small initial mismatch. Epsilon=1e-4 and 1e-5. Every configuration is repeated at eta=.01, 2000 steps, for base/tangent discretization comparison. No broader sweep was run.

## Main results, including the failed prediction

Gain means final held-out prediction RMS sensitivity divided by its initial value for the same specified perturbation direction. It is neither parameter norm growth nor the worst-case singular value of the complete propagator. NMSE is squared Euclidean prediction error divided by total label variance on the 192-input evaluator, so its denominator differs from cycle 3.

Primary modal probe at T=20 with tau=1:

| Seed | No-adaptation NMSE | Input-gradient gain | Stopped-output gain | Input-gradient NMSE | Stopped-output NMSE |
|---|---:|---:|---:|---:|---:|
| 0 | .012340 | 4.83196 | 5.31561 | 1.70307 | 3.80976 |
| 2 | .036731 | 645.16516 | 51.87528 | 54.11134 | 249.73110 |
| 3 | .017224 | 23.35227 | 63.81393 | 3.09070 | 9.74948 |
| 4 | .007521 | 1.85864 | 1.91160 | .51633 | .67178 |

The predefined amplification criterion required stopped-output/fast-EMA modal gain >=2 and fixed-reference stopped-output gain <=1.25 in at least three primary seeds. It passes in seeds 0,2,3; seed 4 is below the >=2 threshold. The fixed-reference modal gains for the four primary seeds range .000960-.010433 across rules. Fast-EMA random-direction gains are also >1 in all primary runs (5.43-319.62), but a single random direction per seed is not a distributional robustness test.

The predefined rule-ordering criterion required gain(stop)/gain(input)>=1.25 in at least three primary seeds. Only seed 3 passes. Seed 2 is a particularly clear warning: input-gradient sensitivity is much larger while stopped-output risk is worse. A sensitivity ranking cannot be substituted for a risk ranking. The test also does not rule out differences explained largely by speed or phase along a trajectory.

For the primary seeds, fixed-reference final NMSE is .00685-.03697. Slow-EMA final NMSE is .00771-.04738, and modal gains remain below .507. Stopped-output/slow-EMA seed 2 meets the preregistered material-deterioration criterion (risk ratio >=1.25 and absolute NMSE increase >=.005). All eight fast-EMA primary runs meet that criterion. No asymptotic protection claim for slow EMA is justified. The fast-EMA consistency discrepancies increase over the interval in all primary runs; this experiment does not demonstrate monotonically decreasing consistency with worsening accuracy. That separate phenomenon was only established in the earlier scalar construction.

Seed 1 missed the earlier fit gate and is excluded from primary gate counts, but not omitted from results. Its final fast-EMA NMSE is 182.16 (input gradient) and 237.81 (stopped output); stopped-output/slow-EMA NMSE is .77183 versus .06806 initially. Full numeric results for all 30 configurations are in SUMMARY.json and finite_response_results.json.

## What was predicted and what was checked

For student velocity F=-G, the actual discrete tangent recurrence is

    delta_s_next = delta_s + eta*(D_s F delta_s + D_t F delta_t),
    delta_t_next = alpha*delta_t + (1-alpha)*delta_s_next.

For a fixed reference alpha=1 and initial delta_t=0, the reference variation remains zero. Iterating this recurrence evaluates the ordered tangent product applied to each of the two directions. The observable derivative is D_s q(s_n) delta_s_n. This is the elementary chain rule, not a new theorem. It is evaluated along each actual moving base trajectory; the t=0 eigenvalues are not extrapolated as constant growth rates.

Complex-step directional differentiation was used for the smooth, holomorphic tanh implementation. It was checked against independent cycle-3 full central-difference matrices and additional central differences at initial, midpoint, and final states. The stopped field remains a recomputed vector field: its local derivative must include changing reference targets even though those targets are detached inside each individual student update. The tangent calculation differentiates that update map, not an incorrectly fixed target across steps.

Maximum relative observable tangent/nonlinear discrepancy across both directions, all recorded times, and all 30 primary-step configurations:

- epsilon=1e-5: 1.272e-5 (0.001272%), below the predefined 1% bound.
- epsilon=1e-4: 0.001269 (0.1269%). The largest errors shrink by approximately the expected factor of 100 under the tenfold symmetric-perturbation reduction; small errors are dominated by floating-point cancellation.
- Halving the step at equal T changes final primary modal gains by at most 2.397% and NMSE by at most 0.517%; both are below the fixed 5% threshold. This is a finite-step sensitivity check, not proof of an ODE limit.
- Largest initial full-block/JVP absolute difference: 3.37e-11. Largest independent along-trajectory field-direction difference: 6.03e-11.
- The 32 frozen coordinates remain constant; tangent variations there remain zero. The other 82 coordinates form a non-fixed parameter block, not a proved controllable/reachable set for the particular stream.

The test supports a numerical finite-time perturbation approximation. It does not provide an unlabeled predictor of task error, a proof of useful adaptation, an asymptotic stability certificate, or a new method.

## Compute, files, and reproducibility

Main run took 195.95 wall seconds on this CPU; no packages were installed and no GPU, paid compute or cluster job was used. There are 30 configurations. Each contains one base plus eight perturbed trajectories at 1000 steps, and one base trajectory at 2000 steps, totaling 330,000 student trajectory updates; directional tangent/verification evaluations are additional. Two tangent directions are propagated at both step sizes. No retraining occurred.

Files: PROTOCOL.md; finite_response.py; finite_response_results.json; summarize_response.py; SUMMARY.json; 30 seedN_RULE_SETTING.npz archives containing recorded base/tangent states, fine-step states, and fixed directions. The raw JSON contains the sampled histories and numerical checks. The scripts import the hash-matched cycle-3 model and checkpoints from the sibling notes directory when copied into the proposal research folder. Script/data/checkpoint hashes and environment versions are recorded. Prior checkpoints are never modified.

## Next single research question

Does ordinary positive student/reference feedback already explain the observed growth, or are residual-weighted terms or mismatched parameter Jacobians essential? Before designing a stabilizer, decompose the existing initial field derivatives into their zero-residual blocks and remainder. Use J=partial_theta f(x,0), L=partial_phi f(x,q), B=I-partial_q f_phi(x,q):

    D_s F_input (zero residual) = -E[J^T B^T B J],
    D_t F_input (zero residual) =  E[J^T B^T L],
    D_s F_stop  (zero residual) = -E[J^T B J],
    D_t F_stop  (zero residual) =  E[J^T L].

Compare these blocks with the independently measured full derivatives on all saved seeds, restricted to non-fixed coordinates. Separately replace L by J as a diagnostic counterfactual, explicitly not an implemented algorithm. Check whether the observed positive modes survive each ablation and remain observable in predictions. Fix a small quantitative tolerance/protocol before computing the comparisons. If ordinary shared-feature positive feedback already explains the effect, do not claim a new mismatch mechanism. If residual terms dominate, the next theory must address a moving trajectory rather than manufacture an exact equilibrium. Do not add another parameter sweep or distribution shift yet.

Any paper framing must then be compared directly with the closest teacher-collapse/stability analyses (especially RMT, PeTTA, and the previously listed fixed-teacher work). Generic EMA failure and fixed-reference benefit are insufficient novelty; the current evidence does not yet support a first-paper claim.
