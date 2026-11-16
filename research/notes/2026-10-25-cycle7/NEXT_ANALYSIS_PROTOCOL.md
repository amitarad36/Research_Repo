# Next analysis: can initial derivatives predict early output separation?

Saved 25 October 2026 after cycle-7 terminal results are known, before computing this approximation or inspecting intermediate prediction-separation curves. This is an analysis plan on existing trajectories, not an independent confirmatory experiment. No thresholds or horizon will be tuned to the result.

## Question and specified approximation

The stopped-output and cached-input fields have identical initial velocity F0, but their initial student derivative blocks differ by mean J^T A J. Does a frozen affine approximation explain the early difference between their actual outputs under the declared EMA update?

Use the full residual-inclusive derivative blocks S_o,T_o,S_c,T_c saved in cycle 7. Restrict to the same 82 non-fixed coordinates per copy. From u_0=v_0=0, separately propagate each affine model:

`u_next = u + eta (F0 + S u + T v)`

`v_next = alpha v + (1-alpha)u_next`, with `alpha=1/(1+eta)`.

Predict output difference on the same 192-input evaluator by `Delta_q_hat = J_eval,0 (u_original-u_cached)`. Compare to `Delta_q = q(theta_original)-q(theta_cached)` from the already saved unperturbed nonlinear states. No evaluator labels are used to construct or fit the prediction. F0 and all matrices are computed once at initialization; no trajectory fitting, rescaling, moving Jacobians, or selecting coordinates by observed accuracy.

## Checks and budget

- Reuse all five seeds, primary 0,2,3,4 and seed 1 sensitivity-only. Only stopped/EMA and cached/EMA at tau=1; eta=.02 and .01. Fixed-reference and full-gradient results remain earlier descriptive evidence.
- Verify all hashes and initial identities. Derive the student-first affine map explicitly; verify the first affine step is identical in both arms and its second-step separation is eta^2 (S_o-S_c) F0, up to roundoff, because their reference derivative blocks agree. This is a check of the affine map, not an exact finite-step identity for the nonlinear models.
- Confirm initial evaluator Jacobian against finite differences with max absolute error <=1e-7. Verify two independently implemented affine propagations agree <=1e-9 relative error. Preserve failures rather than retune.
- At most 30 CPU seconds target, 2 minutes hard cap; no training, new nonlinear adaptation trajectory, extra seed, perturbation run, or literature expansion. Save the matrices/provenance and all predictions at the 51 available record times through T=20. All existing nonlinear state histories remain immutable.

## Primary comparison, fixed before calculation

Primary time **T=2** (100 main steps, 200 fine steps; an existing saved record). Let `rms(v)=sqrt(mean ||v(x)||^2)` on the evaluator and `scale=max(rms(Delta_q), 1e-8 max(rms(q0),1))`. Relative prediction error is `rms(Delta_q_hat-Delta_q)/scale`. A primary seed passes if its nonlinear separation exceeds the scale floor and relative prediction error <=.25 at both step sizes. Require at least three of four primary seeds, plus all numerical checks, for a supported early-output predictor. Report the actual magnitudes and signs/projections as descriptive quantities. A zero/tiny true separation does not count as a pass.

Full T=0..20 curves and the first time the approximation exceeds 25% error are descriptive validity diagnostics only; do not select a successful time after seeing them. No risk-ranking, drift, retention, recovery, asymptotic-stability or novelty inference follows from this output-separation gate. Frozen initial derivatives may fail as nonlinear trajectories leave their neighborhood.

If it fails, reject this simple initial-geometry predictor at the declared time; retain the demonstrated intervention effect. Decide next whether a justified local regime or an implementation/loss robustness check is more valuable. If it passes, derive the assumptions needed for an error bound and test loss/implementation relevance before claiming a mechanism or a paper. The affine approximation and chain rule themselves are established tools, not novelty.
