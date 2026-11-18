# Cycle 37: restoring geometry and an exact finite-step boundary

18 November 2026, evening. Status: checked local calculation using classical stability tools. No trained-model, risk-improvement, or novelty claim. The user's committed **Submission ver no1** (0289fd6404c40d7f0f57f71097d92b1441575e9a) remains unchanged; this is a research-note supplement to the backbone.

## Question and decision

What additional structure repairs cycle 36's failure of a magnitude-only full-state certificate?

The missing condition concerns the **direction** of the diagonal restoring effect. Let H be the student-field derivative, K the negative reference-field derivative, and L=H-K. For symmetric H positive semidefinite and symmetric L, strict positivity of L exactly characterizes continuous-time stability. For positive L, the specified student-first EMA map also has an exact maximum step size, even when H and L do not commute. This provides a bounded positive control to accompany the previous neutral-mode counterexample. It is not yet a novel theorem about trained IT3 networks.

## 1. Model and continuous-time result

Use the proposal's fixed-distribution linearization at a matched equilibrium:

    u_dot = -H u + K v,
    v_dot = (u-v)/tau,       tau>0,
    L = H-K.

H and K are real p-by-p matrices. In this result assume H=H^T >=0 and L=L^T. No commutation is assumed. Since u=v+tau*v_dot, eliminating u gives exactly

    tau*v_ddot + (I+tau*H)*v_dot + L*v = 0.

**Claim:** the joint linearization is Hurwitz if and only if L is positive definite. When L>0 this holds for every tau>0, not uniformly in unbounded tau.

Proof of sufficiency: use

    E(v,v_dot) = (tau/2)||v_dot||^2 + (1/2)v^T L v,
    E_dot = -v_dot^T(I+tau*H)v_dot.

The energy is positive definite and nonincreasing. A trajectory remaining in E_dot=0 has v_dot=0 and then L*v=0, so v=0. The only invariant trajectory in the zero-dissipation set is the origin. Equivalently, the semidefinite quadratic dissipation is observable for this linear system. The classical Lyapunov/observability criterion gives asymptotic, hence exponential, stability. A small smooth nonlinear remainder therefore preserves local exponential stability for the original equilibrium. This does not imply global convergence, monotone prediction error, or task correctness.

Proof of necessity: the real symmetric quadratic pencil is

    Q(lambda)=tau*lambda^2*I+lambda*(I+tau*H)+L.

If L is singular, Q(0) is singular and the joint system has a zero eigenvalue. If L has a negative eigenvalue, its least eigenvalue is negative at lambda=0, whereas Q(lambda) is positive definite for sufficiently large positive lambda. Continuity gives a positive real zero of det Q, hence an unstable mode. These alternatives cover symmetric L that is not positive definite.

Cycle 36 has H=1 and L=-epsilon. Its positive, zero and negative epsilon cases therefore give instability, neutrality and strict stability respectively. A small norm of L alone contains none of this sign information. Exact diagonal consistency gives L=0 and fails this full-state test, as it must when changing both models together changes their predictions.

## 2. Exact student-first discrete boundary

Fix 0<alpha<1 and eta>0 independently. For H=H^T>=0 and L=L^T>0, use

    u_next=(I-eta*H)u+eta*K*v,
    v_next=alpha*v+(1-alpha)*u_next.

**Claim:** its full state matrix is strictly Schur stable if and only if

    eta < 2(1+alpha) / lambda_max(2*alpha*H+(1-alpha)*L).        (1)

At equality it has a multiplier -1. Above this boundary there is a real multiplier below -1. This is an exact statement about this update ordering; it is not an Euler discretization claim and does not cover Adam.

Eliminating u gives the recurrence

    v_next=[(1+alpha)I-alpha*eta*H-(1-alpha)*eta*L]v
           -alpha*(I-eta*H)v_previous.

Its matrix polynomial is

    P(z)=z^2 I-z[(1+alpha)I-alpha*eta*H-(1-alpha)*eta*L]
         +alpha*(I-eta*H).

Its determinant is the characteristic polynomial of the full 2p-dimensional map, including multiplicities. For any polynomial root z, choose a complex nonzero null vector w. Dividing w^*P(z)w=0 by ||w||^2 gives the real-coefficient scalar polynomial z^2-t*z+d, with

    h=w^*Hw/||w||^2 >=0, l=w^*Lw/||w||^2 >0,
    t=1+alpha-alpha*eta*h-(1-alpha)*eta*l,
    d=alpha*(1-eta*h).

The real quadratic Schur conditions are

    1-t+d=(1-alpha)*eta*l >0,
    1+t+d=2(1+alpha)-eta*(2*alpha*h+(1-alpha)*l) >0,
    1-d=1-alpha+alpha*eta*h >0.

Condition (1) makes all three strict for every possible w. This proves sufficiency without simultaneously diagonalizing H and L.

For necessity, P(-1)=2(1+alpha)I-eta*(2*alpha*H+(1-alpha)*L). At equality it is singular. Above the boundary it has a negative eigenvalue. P(z) is symmetric for real z and becomes positive definite as z tends to negative infinity, so continuity forces a singular P(z) at some z<-1. This proves necessity. The quadratic Schur test can also be obtained by mapping the disk to the left half-plane: (1-s)^2 p((1+s)/(1-s)) has coefficients 1+t+d, 2(1-d), 1-t+d, all positive exactly under the three inequalities.

If alpha is chosen as tau/(tau+eta), substitute that value into (1); alpha must not be treated as fixed while varying eta along that schedule. Endpoints alpha=0,1 are outside the theorem. No stability guarantee is asserted for nonsymmetric L or a stopped-rule H with an indefinite symmetric part.

## 3. A noncommuting shared-architecture realization

Take equally likely inputs x=+1,-1, two-dimensional predictions and parameter w, and

    b=(1,0)^T, R=diag(1,2), L=[[2,1],[1,2]],
    U=-R^{-T}L,
    f_w(x,z)=x(b+Rw)+(Uw)(b^T z),
    q_w(x)=x(b+Rw).

At w=0 the reference is the constant-in-z map x*b, which is exactly idempotent. With evaluator target y=x*b, the baseline is nonconstant and correct by construction. No labels enter adaptation. At the matched equilibrium theta=phi=0,

    J=xR, A=0, C=x(R+U),
    H_input=H_stop=R^TR=diag(1,4),
    K_input=K_stop=R^T(R+U)=H-L.

Thus H and L fail to commute, yet both backward rules satisfy the theorem. This is a positive full-state control, not a backward-rule separation. The architecture away from w=0 is not a globally idempotent family. Since R is invertible, every nonzero student perturbation changes predictions; no prediction-changing direction was removed.

For alpha=1/2 the boundary is eta*=6/(7+sqrt(10))=0.5904188215. Direct spectral radii at .99, 1 and 1.01 times this boundary are respectively 0.98191284, 1.00000000 and 1.01809842. The exact symbolic determinant at multiplier -1 vanishes at eta*. These are linearization diagnostics, not nonlinear learning-curve or accuracy experiments.

## 4. Connection to model geometry and limitations

At a zero-residual equilibrium of the smooth squared loss, let B=I-A and C=partial_phi T at fixed auxiliary input. The earlier derivations give

    H_input=E[J^T B^T B J],  L_input=E[J^T B^T(BJ-C)],
    H_stop =E[J^T B J],      L_stop =E[J^T(BJ-C)].

The full-input H is automatically symmetric positive semidefinite. The theorem still requires symmetry and strict positivity of L; these are substantive restrictions on the diagonal feedback derivative, not consequences of a small residual or idempotence. The stopped rule additionally needs the stated condition on H. At nonzero residual, second-derivative terms enter the Jacobians and these reduced formulas cannot simply be reused. Equilibrium and identifiable-coordinate assumptions must be checked before interpreting spectra as local asymptotic stability.

This result covers stationary local dynamics only. It supplies no uniform time-varying certificate, tracking guarantee, unlabeled accuracy certificate, retention/recovery advantage, or trained-model evidence. It does not repair the failed standalone paper directions in the research state.

## 5. Verification and source evidence

Run `python verify_restoring.py` with NumPy 2.1.3 and SymPy 1.13.3. Verification includes exact coordinate-change, energy, matrix-polynomial and -1-boundary identities; exact differentiation of both rules in the shared architecture; eight independent finite-difference columns; 240 continuous matrix cases; and 1,200 discrete cases below, at and above the predicted boundary. The deterministic seed is 3718; dimensions are 1 through 4, 20 matrix pairs each, including zero H. There are 57 noncommuting pairs. Cycle-36 signs are explicit rejection controls. All checks passed. Maximum boundary error 1.78e-14; central differences at this polynomial equilibrium had zero reported discrepancy. CPU time .625 seconds excluding imports. Results and script SHA-256 are in verification.json. Tests corroborate the proof; they do not establish it or empirical usefulness. No data, training, held-out labels, or nonlinear optimization were used.

Primary evidence accessed 18 November 2026:

- S. Boyd, *Linear quadratic Lyapunov theory*, EE363, two-page course note, https://ee363.stanford.edu/archive/notes/lq-lyap-notes.pdf . Page 1, Lyapunov theorems, supplies the positive-energy, semidefinite-dissipation plus observability criterion. Classical supporting machinery, not proposed novelty.
- J. Patsenker, H. Li and Y. Kluger, *Exponential weight averaging as damped harmonic motion*, arXiv:2310.13854v1, 20 October 2023, https://arxiv.org/pdf/2310.13854v1 . Inspected PDF pp.2-3, Sections 2-3, Eqs.(1)-(7) and the BELAY introduction. These already connect EMA to mechanical dynamics and introduce feedback from averaged weights. This cycle's second-order rewrite is not a new physical interpretation. The derivative-defined H,K and exact student-first matrix boundary above were derived locally; novelty against the wider quadratic-matrix-polynomial literature has not been audited. No claim of absence from this or other literature is made. Their alpha convention must not be silently identified with ours.
- Local model provenance: saved proposal Sections 3, 4.2 and 6.2; backbone's constructed-equilibrium subsection; cycle-35 zero-residual identities; cycle-36 Neutral_Robustness_Result.md. The current submission sources and PDFs were preserved.

## Next bounded decision

Before any new training or accuracy sweep, audit whether the archived trained model can support this certificate at all: check the declared equilibrium/residual assumption and rank, then the actual H,L symmetry and restoring signs with a fixed numerical tolerance. A nonstationary checkpoint or nonsymmetric L is an abstention, not evidence of instability and not permission to substitute sym(L) without a new proof. If the certificate has no measurable foothold, retain it as an analytic control and keep the thesis centered on graph/optimizer attribution. For extending the theorem instead, a bounded skew-L perturbation must be compared with a strict full-state stability margin; cycle 36 prohibits falling back to a transverse margin alone.
