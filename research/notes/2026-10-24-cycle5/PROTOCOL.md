# Cycle 5: local feedback decomposition

Written before the decomposition results, 23 October 2026. One bounded diagnostic using existing checkpoints; no training, trajectory sweep, distribution shift, or proposed new adaptation method.

## Question

Are residual-weighted terms or mismatched student/reference parameter Jacobians necessary for the positive, prediction-observable initial modes measured in cycle 3? This is a local derivative question. Removing a term from a matrix does not establish a realizable neural intervention or explain the entire cycle-4 nonlinear risk trajectory.

## Fixed budget

Use all five hash-matched cycle-3 checkpoints and saved full derivative blocks. Seeds 0,2,3,4 are primary; 1 is a sensitivity case. Same first 64 saved evaluation inputs for the field, remaining 192 for output observability; no labels are needed. Both backward rules, tau=1 and 20, and three block models: exact measured full derivative, zero-residual block with actual L, zero-residual block with L replaced by J. Total 60 coupled matrices, each 164x164 after restricting both copies to the 82 non-fixed coordinates. Also report their specified student-first discrete derivatives at eta=.02, alpha=tau/(tau+eta), without a new sweep.

At the matched trained checkpoint define J=partial_theta f_theta(x,0), L=partial_phi f_phi(x,q) with q held fixed, A=partial_q f_phi(x,q), and B=I-A. Use these blocks:

    input: S0=-E[J^T B^T B J], T0=E[J^T B^T L];
    stop:  S0=-E[J^T B J],     T0=E[J^T L].

Subtract S0,T0 from independently measured full S,T. Verify the remainder directly by differentiating C(theta,phi)*r0 with r0 frozen at its baseline value: C_input=J^T B^T, C_stop=J^T. This separates residual-weighted product-rule terms without assuming the checkpoint is an equilibrium. L=J is a diagnostic matrix replacement only, not a valid parameter change asserted for this network.

## Predeclared checks and gates

1. Verify analytic reference-parameter Jacobians with independent complex-step derivatives holding q fixed. Verify the complete block decomposition using complex-step residual-term derivatives versus saved full central differences. Maximum absolute errors must be <=1e-7. For stopped output the teacher residual-weighted derivative must be zero to 1e-12. Eigenpair backward residuals must be <=1e-10 relative to max(1, Frobenius norm of M).
2. A positive observable mode has Re(lambda)>1e-6 and ||J_eval*u||/(sqrt(N_eval)*||u||), divided by ||J_eval||2/sqrt(N_eval), >=1e-3, where u is its student component. Inspect all positive modes, not only a possibly output-invisible dominant eigenvector. Report the greatest real part among qualifying modes, the unrestricted spectral abscissa, leading-mode observability, and eigenpair residual. This tests local output sensitivity, not risk or reachability.
3. Residual terms are not necessary for this signature if the zero-residual/actual-L matrix keeps a positive observable mode in >=3/4 primary seeds for EACH of the four rule/tau settings. A stronger quantitative approximation is declared only if its leading observable rate differs from the full rate by <=10% in every primary setting. Otherwise report magnitudes and exceptions without relabeling the gate.
4. Derivative mismatch is not necessary for the simplified signature if the zero-residual/L=J matrix keeps a positive observable mode in >=3/4 primary seeds for EACH rule/tau setting. If this passes, reject a claim that mismatch is required for these modes. A separate <=10% all-primary rate comparison decides whether L=J quantitatively reproduces the full rates; mere sign survival does not mean it explains their values or nonlinear trajectories.
5. Record Frobenius remainder ratios, ||L-J||F/||J||F, and all five seeds. No tuning of tau, coordinate threshold, seed eligibility, or classification thresholds after results. If gates fail, retain that failure and choose the next research question accordingly. No theorem about future risk or useful adaptation follows from local eigenvalues.

Provenance: IT3-inspired squared surrogate and graph remain as audited in prior notes; this cycle derives and checks project calculations, not new literature facts. Any generic feedback result must be compared with closest teacher dynamics/control work before a novelty claim.
