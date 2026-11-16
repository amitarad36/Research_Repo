# The constructed rotation did not appear in a small learned model

22 October 2026, cycle 3. Status: completed synthetic experiment and checked derivative calculations; negative mechanism screen. No benchmark reproduction, asymptotic neural stability theorem, distribution-shift result, or novelty claim.

## Decision

The fixed five-seed test passed its prediction-fit gate (4/5 seeds) and failed its strong-rotation gate (0/5, with zero qualifying points at both prediction and label probes in every seed). Deprioritize the cycle-2 strong rotational construction as an empirical explanation for this model. Preserve it as an existence result. Do not replace it with an assumed projection model: the measured normalized idempotence defects are substantial. The next question should concern the actual coupled student/reference parameter derivatives and observable finite-time behavior.

This is one small architecture and one smooth synthetic labeling function on the source distribution. It does not establish that IT3 in general lacks rotation, or that ordinary training cannot learn it elsewhere. One seed missed the predefined fit threshold and is reported rather than retrained or discarded silently. There was no architecture, seed, loss, or hyperparameter search.

## Protocol and primary-source scope

PROTOCOL.md was saved before training and its SHA256 is in neural_gate_results.json. Five 4->16(tanh)->2 MLPs, 114 parameters each, were trained for 6000 Adam updates on 256 inputs with both (x,0)->y and (x,y)->y. There are 256 independent held-out inputs. The labels are fixed, smooth, nonconstant, and nonzero. The training and adaptation discrepancy are squared surrogates. See the protocol and code for exact functions, seeds, schedule, normalization, and thresholds.

Source checked again this cycle: Durasov et al., IT3, arXiv:2410.04201v2, 25 May 2025, Section 3.1 Eq. (2) for the zero/label training recipe; Eqs. (6)-(7) for fixed/EMA reference discrepancies: https://arxiv.org/html/2410.04201v2 . The original displayed training/discrepancy loss is an unsquared norm. Architecture, squared losses, data, optimizer, EMA ordering, and geometry gates here are project choices. This experiment did not execute the official repository or establish missing online implementation details.

No test labels enter either adaptation field. Label-input derivative probes use held-out labels only as diagnostics. The neural experiment did not run adaptation trajectories, shift inputs, or evaluate continual retention/recovery.

## Results

NMSE is mean squared Euclidean output error divided by total held-out label variance (not the absolute MSE). Fit requires both zero-input and label-input NMSE <= 0.05. Geometry numbers below are medians over 256 held-out inputs, evaluated at the network prediction z=q.

| Seed | Zero-input NMSE | Label-input NMSE | Fit gate | median ||A||2 | median relative idempotence defect | Strong rotation fraction |
|---|---:|---:|---|---:|---:|---:|
| 0 | .012026 | .004052 | pass | .197149 | .825480 | 0 |
| 1 | .066201 | .014347 | fail | .747314 | .299946 | 0 |
| 2 | .034339 | .006985 | pass | .662153 | .462177 | 0 |
| 3 | .017082 | .006121 | pass | .388671 | .679543 | 0 |
| 4 | .007193 | .001459 | pass | .099477 | .943169 | 0 |

Here A=partial_z f(x,z). Relative defect is ||A^2-A||F/(||A||F+1e-12). Seed 0 has complex eigenvalues at 13/256 prediction probes, but their rotation is weak: none meets the preregistered spectral inequality. Other seeds have real eigenvalues at all prediction probes. All measured prediction-probe operator norms are below 0.902; this is sampled input contraction, not a neighborhood/global contraction proof or a guarantee for coupled parameter updates. Small absolute defects in seeds with small A must not be described as evidence for nontrivial projection structure.

The rotation screen uses eigenvalues a +/- i*w with 0<a<1 and w^2>a*(1-a). For a normal A=aI+wJ, the fast-reference output-level input-gradient field has real rate a*(1-a)-w^2, whereas the stopped field has real rate a. This is the cycle-2 rotation mechanism. For general A, this spectral screen alone would not prove any stability result. The additional actual-parameter audit is therefore necessary even if the screen passes.

## Actual parameter dynamics: diagnostics, not an equilibrium certificate

Both backward fields were evaluated with independent student/reference copies, on the first 64 held-out inputs. The 114-by-114 derivatives with respect to each copy were computed directly by central differences of the entire field; the coupled ODE has dimension 228. These include nonzero-residual terms rather than replacing them by a zero-residual Gauss-Newton formula.

At matched trained weights, residual RMS is .0179-.0294 and student velocity norms are .00854-.01452. The checkpoints are not equilibria. Every measured coupled Jacobian has positive spectral abscissa at tau=1 and tau=20. For the four fit-passing seeds, at tau=1 the input-gradient abscissae range .0925-.3426 and stopped-output abscissae .0996-.6304. At tau=20 the corresponding ranges are .00773-.05729 and .00799-.06701. The specified discrete derivatives also have radius >1. These are instantaneous derivatives at moving states: they do not establish sustained trajectory growth, asymptotic instability, worse predictive risk, or a benefit from slowing EMA. Full spectra and residuals are retained in the JSON and checkpoints.

There is an exact architecture-specific degeneracy. The student's first pass always has z=0, so the 32 first-layer weights multiplying z receive zero gradient under both backward rules. With a matched reference initialization they also stay constant in the EMA. All 32 corresponding rows of both student-field derivative blocks are exactly zero. Perturbing these constants is outside the reachable adaptation subspace for a fixed pretrained initialization. Restricting both copies to the remaining 82 coordinates removes this structural issue; the maximal real eigenvalues change by at most 4.11e-15. A full-parameter strict Hurwitz claim would be inappropriate even at a hypothetical equilibrium. Additional small singular values and the nonzero residual remain separate issues. The identifiable output directions, rather than arbitrary parameter distance, must be monitored in the next test.

Why measuring A alone was insufficient: write J=partial_theta f_theta(x,0) and L=partial_phi f_phi(x,q), holding q fixed in L. At zero residual, B=I-A yields

    D_theta F_input = -E[J^T B^T B J],  D_phi F_input = E[J^T B^T L],
    D_theta F_stop  = -E[J^T B J],      D_phi F_stop  = E[J^T L].

The constructed additive family assumed a special relationship between J and L. A general trained network need not have it. These chain-rule identities are elementary reproduced calculations, not a novelty claim; at the nonstationary checkpoints the exact derivatives also contain residual-weighted terms. The observed positive abscissae cannot yet be attributed specifically to J/L mismatch, residual terms, or another mechanism.

## Verification and compute

- Initial and trained-checkpoint tests cover supervised gradients, reference-input Jacobians, parameter Jacobians, the true input-gradient discrepancy, and the stopped-target surrogate with its reference target held fixed. Maximum absolute directional/Jacobian error: 2.44e-10.
- All full-field derivatives were repeated with central-difference steps 1e-5 and 5e-6; largest entrywise difference: 1.23e-10. Independent mixed student/reference directional checks passed (<2.74e-11).
- The post-result invariant-subspace audit is reproducible with inspect_saved.py. It does not train or select new networks.
- Main run: 30,000 minibatch updates total; NumPy float64 CPU, 22.07 seconds measured wall time for training and diagnostics on this host. Environment versions are in the JSON. No package installation, GPU, paid compute, or cluster job.
- Saved shared data, all five final checkpoints, A matrices at both probe choices, full derivative blocks, spectra, protocol, and scripts. No deletion of unique research material.

## Next single question and stop condition

Do the observed positive local modes produce observable finite-time prediction amplification under EMA, and can a tangent model along the actual trajectory predict it? Reuse all five saved checkpoints, with the fit-passing subset declared as primary and seed 1 retained as a sensitivity case. Avoid more pretraining/search. First keep the input distribution stationary so that instability of adaptation is not confused with tracking drift.

Before running, freeze a small protocol: matched fixed-reference and EMA (tau=1,20), both backward rules, and no-adaptation control; explicit student step, batch/update budget, and held-out evaluator separate from adaptation inputs. Compare paired small perturbations in the reachable 82-parameter subspace with propagation by the actual time-dependent update Jacobian. Measure prediction separation, current risk, discrepancy, and reference lag; labels stay evaluation-only. Check perturbation size and discretization once. Spectra at t=0 are not the predictor of finite-horizon growth; the ordered tangent product is. A local nonlinear agreement failure at sufficiently small perturbation falsifies the implementation/approximation. If no meaningful observable amplification or rule distinction survives, demote the stability mechanism instead of tuning until it appears. Only after a mechanism is supported should covariate drift and continual baselines be added.

This next test may support a narrower thesis question about when output-level consistency geometry fails to describe coupled parameter dynamics. It is not yet a defensible first-paper claim; closest teacher-dynamics and control prior art still needs exact comparison.
