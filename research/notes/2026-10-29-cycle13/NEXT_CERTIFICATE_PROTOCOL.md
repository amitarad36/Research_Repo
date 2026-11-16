# Next cycle: can the early forecast be certified nonvacuously?

Saved29 October2026 before computing any certificate. One question: can valid local remainder bounds certify the existing stopped/cached squared-loss output-separation forecast at the unchanged T=2 horizon? No new neural adaptation or training. This is retrospective analysis of known trajectories, not independent confirmation.

## Fixed model and data

Reuse cycle7 squared-loss trajectories and cycle8 affine matrices/checkpoints, verifying their manifests first. Primary seeds0,2,3,4;seed1 sensitivity-only. Original stopped-output/EMA versus cached-input/EMA, tau=1, eta=.02/.01, same student-first reference update, same64/192 inputs and128-number cache. Preserve original squared normalization. Use82 active coordinates per parameter copy with Euclidean norm, no fitted rescaling. Output map Q is the stacked192x2 evaluator predictions divided by sqrt(384), so its Euclidean norm is RMS. Do not load evaluator labels.

## Analytic obligation before numerical use

Let Phi_a be the exact discrete joint update for arm a; u^a_0=0 and u^a_{n+1}=c_a+A_a u^a_n, with c_a=Phi_a(z0)-z0 and A_a=D Phi_a(z0). For a Euclidean ball radius R around z0, obtain a valid Lipschitz bound L_a for D Phi_a and a valid Hessian bound H_Q for Q. Establish an induction for state-error enclosure E^a_0=0 and a candidate recurrence

    E^a_{n+1} = ||A_a|| E^a_n + (L_a/2) (||u^a_n||+E^a_n)^2,

valid only while ||u^a_n||+E^a_n<=R (also check the final state for output use). Then justify

    B^a_n = ||D Q(z0)|| E^a_n + (H_Q/2)(||u^a_n||+E^a_n)^2,
    B_n=B^original_n+B^cached_n.

Do not regard these proposed recurrences as a proved certificate until domain/constant/norm/order details are checked. A valid tighter paired bound may be derived analytically before looking at certificate outcomes, but cannot be fitted to saved nonlinear errors. Initial spectral/finite-difference samples alone are not supremum derivative bounds. Prefer explicit tanh derivative and weight-norm enclosures; document conservative floating-point treatment. If rigorous numeric enclosure cannot be implemented with existing dependencies, report that blocker to certification instead of relabeling sampled bounds as proofs.

## Prespecified choices and gate

Use the radius R=2*max_{a,n<=T/eta}||u^a_n|| for each seed/step pair, based only on the affine forecast, before inspecting actual errors. If zero, handle the exact stationary case separately and do not count it as nontrivial separation. No radius search, norm rescaling, horizon/step changes or Hessian fitting after outcomes.

At T=2 for both steps, require valid containment and B<=.20||Delta_Q_hat|| in at least3/4 primary seeds. Also require ||Delta_Q_hat||-B above the cycle8 floor 1e-8*max(RMS(q0),1). Then the triangle inequality yields error/||Delta_Q_true||<=.25. Validate against saved trajectories only after the certificate is fixed; any bound violation fails correctness. Missing rigorous constants/containment, overflow, or a loose bound are failures, not omitted seeds. Preserve all results. Descriptive later-time comparisons cannot rescue the primary gate.

Baselines for this analysis: the unchanged initial-affine forecast and its zero-separation baseline, using the same saved input-path intervention; original full-gradient and fixed-reference risk arms remain prior controls, not part of this new primary gate. This analysis does not forecast risk, certify useful adaptation or resolve the failed absolute-loss study.

## Budget and outcome

One analytic/reuse cycle, at most60 CPU seconds for optional computations, using installed libraries; no new trajectories, training, labels, seeds, smoothing, tuning, packages, GPU or paid work. If explicit derivative bounds prove intractable within the cap, save the scoped theorem and exact remaining obstacle, not a positive gate. A failed gate demotes this certificate approach in its present form; no automatic new sweep. A pass supports planning, separately, one prospective implementation-relevant study with declared graph, loss, optimizer, reference ordering, data/compute budgets, no-adaptation/fixed-reference/full-input controls and evaluator-only labels. Publication novelty still needs an application-specific claim beyond standard perturbation estimates.
