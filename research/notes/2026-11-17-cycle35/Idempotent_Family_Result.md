# Cycle 35 — exact idempotence and backward-rule stability

17 November 2026. Status: new checked calculation in a constructed smooth squared-loss model. This is supporting theory, not a trained-model result or a publication-novelty claim. The submission proposal and backbone TeX are left unchanged while the user reviews Section 3.

## Question and answer

Does enforcing exact idempotence of the reference eliminate the stability difference between propagating the reference-input derivative and stopping its output?

**No.** An explicit globally idempotent affine-in-input family has a nonconstant, task-correct baseline. With the reference fixed, that baseline is locally exponentially stable for the full input-gradient flow and unstable for the stopped-output flow. With an EMA reference the same family has a neutral equilibrium manifold, but its transverse stability still depends on the backward rule and reference speed. Its input Jacobian has eigenvalues only zero and one: the mechanism uses an oblique projection, not the rotating eigenvalues of the earlier construction.

This narrows a useful distinction: idempotence implies a projection Jacobian at a fixed point, but does not imply an orthogonal projection. Orthogonality gives equality of the two linearizations under the assumptions below.

## 1. General zero-residual calculation

Fix the input distribution. Assume the smooth squared discrepancy, ordinary Euclidean parameter updates, differentiability of the expectation, and a matched point theta=phi=w with zero residual almost surely. Put J=partial_theta q, A=partial_z T, C=partial_phi T with its input held fixed, and B=I-A, all at this point. The residual derivatives are -BJ and C. Consequently:

    H_input = E[J^T B^T B J],   K_input = E[J^T B^T C],
    H_stop  = E[J^T B J],       K_stop  = E[J^T C],

where H=partial_theta G and K=-partial_phi G. Derivatives of J or A multiplied by the residual vanish at this point. This is direct differentiation, not a new general calculus theorem.

Now assume the stronger local-family identity T_w(x,q_w(x))=q_w(x) for every w in a parameter neighborhood. Differentiating along the diagonal gives C=BJ, hence K=H for both rules. Exact idempotence of each f_w(x,.) is sufficient for this family identity because q_w(x)=f_w(x,0). Idempotence at a single trained parameter value alone is not enough to infer C=BJ.

For perturbations u,v, let delta=u-v. The joint linearization becomes

    delta_dot = -(H + I/tau) delta,   v_dot = delta/tau.

There are p neutral diagonal directions. For the student-first discrete update,

    delta_next = alpha (I - eta H) delta,
    v_next = v + (1-alpha)(I-eta H) delta.

Its spectrum contains p unit multipliers. These are statements about the linearization; no strict convergence to one parameter pair follows. If Jv is nonzero, the neutral direction changes predictions and cannot be dismissed as parameter redundancy.

If A is idempotent and symmetric almost surely, B^T B=B. Then H_input=H_stop, and under the family identity K_input=K_stop as well: the two joint linearizations coincide. The same equality of H holds for the fixed-reference zero-residual linearizations. Without symmetry, A^2=A alone does not establish these equalities.

## 2. Explicit smooth shared architecture

Take equally likely inputs x=-1,+1. One scalar parameter w controls the two-output predictor. Define

    a_w = 1-3w,             b_w(x) = x(1+w),
    A_w = [[1,a_w],[0,0]],  q_w(x) = (-a_w b_w(x), b_w(x))^T,
    f_w(x,z) = q_w(x) + A_w z.

For every w and x, A_w^2=A_w and A_w q_w=0. Thus f_w(x,0)=q_w(x) and f_w(x,f_w(x,z))=f_w(x,z) for every z. Every member of the family is exactly idempotent.

Use the task y(x)=q_0(x)=(-x,x)^T only for interpretation/evaluation. Predictions at the baseline differ between the two inputs, and f_0(x,y(x))=y(x). Correctness is deliberately supplied by the constructed baseline; it is not inferred from unlabeled inputs. Both model copies have this same architecture.

For student theta and reference phi,

    r = x(theta-phi)(a_phi,-1)^T,
    D = (1/2)(theta-phi)^2 (a_phi^2+1),
    G_input = (theta-phi)(a_phi^2+1),
    G_stop = -(theta-phi)[(2+6theta)a_phi-1].

The backward rules have identical forward predictions and discrepancy at every shared state. Stopping the reference output changes only the derivative used for the student update.

### Fixed reference phi=0

    theta_dot_input = -2theta,
    theta_dot_stop = theta + 6theta^2.

At theta=0 the respective eigenvalues are -2 and +1. The full-input flow converges globally in theta to zero; the stopped flow has a locally unstable zero equilibrium. This is a local stability comparison, not a claim of uniform risk decrease everywhere or successful distribution-shift adaptation. The other stopped equilibrium is not needed for the claim.

### Moving reference phi_dot=(theta-phi)/tau

At theta=phi=0, H_input=K_input=2 and H_stop=K_stop=-1. The spectra are

    input: {0, -2-1/tau},    stop: {0, 1-1/tau}.

The diagonal is an exact equilibrium manifold, with predictions varying along it. Transverse perturbations contract for input gradients at every tau>0. The stopped rule contracts transversely for tau<1 and is transversely unstable for tau>1. No nonlinear conclusion at tau=1 is made here.

The transverse discrete multipliers are alpha(1-2eta) and alpha(1+eta), respectively. With alpha=tau/(tau+eta), the stopped multiplier crosses one at tau=1 for every eta>0; the input rule also requires |alpha(1-2eta)|<1. The continuous-time classification alone does not certify arbitrary step sizes.

With eta=1/10, tau=1/2 gives transverse multipliers 2/3 and 11/12; tau=2 gives 16/21 and 22/21. One neutral unit multiplier remains in every case.

## 3. What this resolves and what it does not

The earlier backbone witness required rotating reference-input eigenvalues and noted that exact projection geometry would exclude that particular witness. This example shows that exact idempotence does not exclude every backward-rule stability separation. Here A_0=[[1,1],[0,0]] is oblique; J=x(2,1)^T, BJ=x(-1,1)^T, so J^T BJ=-1 while ||BJ||^2=2. This is the source of the sign change.

The input-Jacobian spectrum by itself cannot distinguish this oblique projector from an orthogonal one. A relevant geometry check must include parameter-output directions J and projection asymmetry, not only eigenvalues of A. The symmetry condition is sufficient for equality, not necessary: some nonsymmetric A can still give equal compressed fields.

This is not evidence that trained IT3 learns these projections. The architecture and labels are constructed; the loss is squared and smooth; there is no Adam, sampling noise or distribution drift. Starting with identical copies anywhere in the exact family yields no update at all. The diagnostics perturb the student relative to the reference and study recovery of their disagreement. EMA agreement can settle at a different, inaccurate prediction; it is not recovery of the correct baseline. No retention result is inferred.

## 4. Verification and next decision

verify_idempotent_family.py checks exact symbolic idempotence, the two forward definitions, residual, both gradient rules, fixed-reference eigenvalues, and continuous/discrete characteristic polynomials. It evaluates 16 exact rule/tau/eta cases, independently checks gradients by finite differences at 40 fixed-seed parameter pairs, and checks eight small nonlinear finite-step diagnostics (tau=.5,2; eta=.1,.05; horizon 2; initial theta=1e-6, phi=0). The diagnostics agree with the transverse classifications. Full machine-readable results are in verification.json. Maximum gradient discrepancy is 3.151e-11; measured script CPU time is under one second on this run. No training or prior held-out data was accessed.

Next bounded target: derive a perturbation bound for the zero-residual linearizations when projection asymmetry and the diagonal derivative defect C-BJ are small, explicitly controlling J and the transverse spectral margin. Test any proposed bound against this exact oblique witness and an orthogonal control before considering trained-network diagnostics. Residual-weighted second derivatives must be included away from zero residual. Do not run another retrospective accuracy sweep on the existing 40 examples.

Novelty remains provisional. The classical difference between residual-gradient and direct/semi-gradient learning must be compared carefully; the value here is the explicit shared, exactly idempotent, prediction-input architecture and the separation of transverse stability from accuracy. See SOURCE_EVIDENCE.md.
