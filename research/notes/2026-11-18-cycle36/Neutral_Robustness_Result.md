# Cycle 36 — robustness near an idempotent equilibrium family

18 November 2026. Status: checked calculation and a narrowly stated sufficient bound, plus a counterexample to a stronger robustness claim. No publication novelty, trained-model behavior, or predictive improvement is claimed. No TeX edit was needed.

## Question

Can the cycle-35 transverse stability margin be made robust using small projection asymmetry and a small diagonal derivative defect?

**Answer:** these quantities give useful bounds for frozen-reference or exactly diagonal-consistent dynamics. Small diagonal defect alone does not preserve stability of the full moving-reference state, because the unperturbed equilibrium family has neutral directions. A shared-architecture example proves this for arbitrarily small defects, in continuous time and for the stated finite-step EMA update.

## 1. Zero-residual identities and a conditional positive result

Use the cycle-35 definitions at a matched zero-residual equilibrium: J=partial_theta q, A=partial_z T, B=I-A, C=partial_phi T at fixed auxiliary input. Assume smooth squared loss and differentiation under expectation. Let Gamma=C-BJ measure the derivative of the diagonal residual. Then

    H_i=E[J^T B^T B J],    H_s=E[J^T B J],
    E_i=K_i-H_i=E[J^T B^T Gamma],
    E_s=K_s-H_s=E[J^T Gamma].

All norms below are spectral/Euclidean norms. The exact identity

    B^T B-B = (A^T-A)(A-I) + (A^2-A)

gives, writing d=||H_i-H_s||,

    d <= E[||J||^2 (||A^T-A|| ||B|| + ||A^2-A||)],
    ||E_i|| <= E[||J|| ||B|| ||Gamma||],
    ||E_s|| <= E[||J|| ||Gamma||].

Thus approximate idempotence and approximate symmetry bound a *compressed gradient-field difference*, provided J is controlled. Idempotence defect alone does not. These are elementary norm bounds, not a new perturbation theorem in general control theory.

Let H_i >= m I on the parameter coordinates being analyzed. Since H_i is symmetric, sym(H_s) >= (m-d)I. If m>d, the fixed-reference stopped linearization -H_s is strictly dissipative, with exponential contraction rate at least m-d in Euclidean norm. The full-input fixed-reference linearization has rate at least m. These are local nonlinear stability statements under the usual differentiability hypotheses, not global claims.

For the moving reference, put delta=u-v and E=K-H. The exact joint linearization is

    delta_dot = -(H+I/tau)delta + E v,
    v_dot = delta/tau.

If sym(H)+I/tau >= beta I with beta>0, variation of constants gives

    ||delta(t)|| <= exp(-beta t)||delta(0)||
                  + ||E|| integral_0^t exp(-beta(t-s)) ||v(s)|| ds.

When E=0 this proves disagreement contraction. For the stopped field, beta=m+1/tau-d is a sufficient positive margin. When E is nonzero the bound is conditional on the size of v; it cannot be closed by simply declaring the neutral coordinate bounded. This is a linearization bound, not a nonlinear finite-basin or risk bound.

The exact compressed d can be substantially sharper than the primitive-Jacobian upper bound. In cycle 35, H_i=2, H_s=-1 and d=3: beta=1/tau-1 matches the stopped transverse boundary. The more conservative primitive bound can abstain even when this exact calculation certifies contraction.

## 2. An arbitrarily small defect creates a full-state instability

Take equally likely scalar inputs x=-1,+1, scalar prediction, scalar parameter w, and a fixed scalar architecture coefficient epsilon. Define

    f_w(x,z)=x(1+w)+epsilon*w*z,
    q_w(x)=f_w(x,0)=x(1+w).

Use the same architecture for student and reference. At theta=phi=0 the two predictions equal x, so residual is zero. If evaluator labels are y=x, the constructed baseline is correct and nonconstant. Labels are not used by the update.

At w=0 the reference is exactly idempotent: f_0(x,z)=x for every z. Its input derivative A=0 is an orthogonal projector. At the matched baseline,

    J=x, B=1, C=x(1+epsilon), Gamma=epsilon*x,
    H_input=H_stop=1, K_input=K_stop=1+epsilon.

Hence the projection-asymmetry defect and idempotence defect are both exactly zero there, and the diagonal derivative defect can be arbitrarily small. For epsilon != 0, the whole parameter family is not exactly idempotent; this is precisely the hypothesis being relaxed from cycle 35.

Writing r=x*rbar, the exact residual and fields are

    rbar=(1+epsilon)phi-theta+epsilon*phi*theta,
    G_input=rbar*(epsilon*phi-1),
    G_stop=-rbar.

With the reference frozen at zero, both rules give theta_dot=-theta. With an EMA reference, both have the same zero-residual joint Jacobian

    M = [[-1, 1+epsilon],
         [1/tau, -1/tau]].

Its characteristic polynomial is

    lambda^2 + (1+1/tau)lambda - epsilon/tau.

For every epsilon>0 and tau>0 the determinant is negative, so one real eigenvalue is strictly positive: the useful matched equilibrium is a local saddle for both backward rules. The transverse coefficient at epsilon=0 is strictly contracting, -(1+1/tau), yet there is no positive defect-size threshold that preserves full-state stability in all directions. For epsilon<0 both coefficients are positive and the matrix is Hurwitz. The *sign and direction* of the neutral-mode restoring effect matter, not merely the defect magnitude.

At epsilon=0 the zero eigenvalue is genuine: changing both parameters together changes the prediction by x*w. This is not a parameter redundancy that can be removed while preserving all observable predictions.

The positive eigenvalue for epsilon>0 is

    lambda_plus = (-(1+1/tau) + sqrt((1+1/tau)^2+4epsilon/tau))/2
                = epsilon/(tau+1) + O(epsilon^2).

A slow instability can therefore be missed by a short finite-horizon disagreement check. The asymptotic expansion is at fixed tau, epsilon -> 0.

## 3. The finite-step update has the same obstruction

For the proposal's student-first rule, eta>0 and 0<alpha<1,

    N = [[1-eta, eta(1+epsilon)],
         [(1-alpha)(1-eta), alpha+(1-alpha)eta(1+epsilon)]].

The characteristic polynomial at one satisfies

    det(I-N) = -eta(1-alpha)epsilon.

If epsilon>0, this is negative. Since the monic real quadratic tends to positive infinity on the positive real axis, it has a real root greater than one. Thus every positive step size and every strictly moving EMA retain an unstable linear mode in this example. This is a direct discrete calculation, not an inference from the ODE.

For epsilon<0, strict Schur stability requires the remaining step-size condition

    eta < 2(1+alpha) / [2alpha-(1-alpha)epsilon].

For epsilon=0, one multiplier is exactly one and the other is alpha(1-eta). Alpha=1 is the fixed-reference endpoint and is excluded from the strict moving-reference claim.

## 4. Verification and scope

verify_neutral_robustness.py verifies both fields, both joint Jacobians, the characteristic polynomial, the discrete determinant identity, and the projection-defect identity symbolically. It checks 36 exact rational epsilon/tau/eta fixtures: epsilon=+/-1/10,+/-1/100,+/-1/1000000; tau=1/4,1,4; eta=1/100,1/10; alpha=tau/(tau+eta). Positive-defect continuous and discrete instabilities pass; negative-defect Schur conditions pass in exact arithmetic on the selected grid. Independent central differences check 40 field probes. Fifty matrix fixtures check the norm inequalities and block identities.

Results: maximum finite-difference discrepancy 1.337e-11; maximum matrix-identity discrepancy 8.130e-15; script CPU .515625s. No training, optimizer state, real-data labels or published IT3 implementation was executed. The first exploratory symbolic check failed because the characteristic-polynomial comparison used inconsistent SymPy assumptions for the polynomial variable; declaring a single assumption-free lambda fixed that representation issue. The persisted verifier contains the corrected check.

This result rejects only the attempted *full-state* robustness implication from a positive disagreement margin plus a small unconstrained diagonal defect. It does not invalidate the cycle-35 exact-family result, the current proposal's conditional Hurwitz/Lyapunov statement, or stability guaranteed by a separate restoring constraint. Residual-weighted second derivatives must also be accounted for at nonzero residual.

## 5. Source evidence and next decision

- Established background: S. Boyd, EE363, Linear quadratic Lyapunov theory, two-page support note, accessed 18 November 2026. https://ee363.stanford.edu/archive/notes/lq-lyap-notes.pdf . Page 1, Lyapunov theorems and converse theorems, gives the strict positive-definite certificate conditions. Page 2's integral solution assumes an asymptotically stable system. These facts are standard and are not proposed contributions here. A neutral mode cannot satisfy a strict full-state certificate.
- Model provenance: current proposal Section 3 (definitions and student-first EMA) and Section 4.2 (H,K and block matrix), plus cycle-35 Idempotent_Family_Result.md section 1. All were read and preserved. The construction and calculations in sections 1–3 above are new local checks, not claims from those external notes.
- IT3 graph and loss provenance stays in cycle-35 SOURCE_EVIDENCE.md and the earlier pinned implementation audits; no official-code reinterpretation is made in this cycle.

Next bounded question: identify a model-specific restoring condition on the common student/reference directions, or explicitly restrict the result to disagreement contraction. Before any new trained-model run, require a full-block local certificate on prediction-relevant coordinates and include the positive/negative epsilon fixtures as mandatory rejection tests. Source anchoring is existing prior art, not a novelty claim. Do not pursue full-state robustness from defect magnitude alone or another retrospective accuracy sweep.
