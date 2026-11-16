# A useful fixed point with backward rule dependent stability

21 October 2026, second research cycle. Status: a constructed smooth model, an exact rotational-mode calculation, and exact-rational finite-case stability certificates. These are project results, not a publication-novelty claim or an empirical finding about IT3.

## Result and scope

The same smooth dual-input architecture and forward consistency loss can have a correct, nonconstant prediction equilibrium that is locally exponentially stable when the reference-input derivative is retained but unstable when the reference output is detached. This difference is not a positive rescaling of time. In the example, slower EMA restores local stability of the detached rule. The effect holds for an explicitly noncommuting pair of feedback matrices and for the declared student-first discrete update.

Correctness is built into a fixed pretrained baseline h(x). No new labels enter the adaptation updates. This establishes mathematical realizability; it does not show that supervised neural pretraining learns the required geometry, that current distributions drift, or that EMA avoids forgetting. An exactly initialized model stays at the equilibrium: our recovery tests perturb its parameters.

## 1. A smooth shared architecture, not an arbitrary linear field

Use two inputs, each with probability one half:
x1=(1,0)^T, x2=(3/5,4/5)^T.
Choose a fixed pretrained baseline h(x1)=(1,0)^T and h(x2)=(0,1)^T. The corresponding evaluation labels equal h. The outputs are nonzero, distinct, and span R^2. The architecture can be extended smoothly in x around these two locations. Only the fixed two-point support is used here.

For h(x) nonzero, define

f_W(x,z) = h(x) + W x + A_x (z-h(x)) [h(x)^T z / ||h(x)||^2].

W is an adapted 2-by-2 matrix and V is its EMA reference. The baseline h and A_x stay fixed. Thus

q_W(x)=f_W(x,0)=h(x)+Wx,
f_0(x,h(x))=h(x),
partial_z f_0(x,h(x))=A_x.

Consequently W=V=0 is a task-correct, noncollapsed consistency fixed point. No label is queried during adaptation. The strong assumption is that h already represents the correct function on both input locations.

Put u=Wx, v=Vx, B_x=I-A_x, and ell_x=h(x)/||h(x)||^2. The actual residual is

r = f_V(x,q_W(x))-q_W(x)
  = v - B_x u + A_x u (ell_x^T u).

For the squared discrepancy D=(1/2) E||r||^2, holding V fixed during the W derivative gives

dot W_input = -E[(partial_u r)^T r x^T],
partial_u r = -B_x + A_x[(ell_x^T u)I + u ell_x^T],

whereas explicitly detaching the whole reference output gives dot W_stop=E[r x^T]. Both use tau dot V=W-V.

These are the same derivative conventions as the proposal. This construction, including its quadratic z term, is our model and is not an architecture claimed to appear in the IT3 paper.

## 2. Exact local linearization and current-risk interpretation

For column vectorizations w=vec(W), v=vec(V), the Jacobian at zero is

M = [ -H, K ; I/tau, -I/tau ],

with

H_input=E[xx^T tensor B_x^T B_x], K_input=E[xx^T tensor B_x^T],
H_stop=E[xx^T tensor B_x], K_stop=E[xx^T tensor I].

The input covariance is C=[[17/25,6/25],[6/25,8/25]], with eigenvalues 4/5 and 1/5. Current evaluation risk is R(W)=(1/2)E||Wx||^2. Since C is positive definite, R is a positive-definite quadratic form in W. Hurwitz M now implies local exponential recovery to the correct h, not zero prediction. It does not imply monotone risk at every instant.

For the smooth nonlinear ODE, the usual linearization result transfers strict local stability or instability of the hyperbolic equilibrium. Positive-real-part eigenvectors have a nonzero student component, so the instability affects prediction error, rather than only an unobserved teacher variable. The same principle applies to the smooth discrete map when its Jacobian is strictly inside or has eigenvalues outside the unit circle.

## 3. An interpretable exact boundary with rotational feedback

Let J=[[0,-1],[1,0]], and first set both B_x=B0=(1/2)I+J. Diagonalizing C splits the linearized parameter dynamics into output-rotation modes with student speed mu in {1/5,4/5}. Write c=1/tau.

For input-gradient updates, a complex mode has polynomial

lambda^2 + ((5/4)mu+c)lambda + mu c(3/4 +/- i).

Its real quartic is (lambda^2+a lambda+d)^2+e^2,
where a=(5/4)mu+c, d=(3/4)mu c, |e|=mu c.
The Routh condition reduces to a^2 d > e^2. This holds for all mu,c>0 because

((5/4)mu+c)^2 >= 5 mu c,
(3/4)*5 mu c > mu c.

All other quartic positivity conditions hold. Therefore the input-gradient equilibrium is locally exponentially stable for every finite tau>0.

For the output-detached rule, a complex mode has polynomial

lambda^2 + ((1/2+i)mu+c)lambda + c mu(-1/2+i).

Writing an imaginary-axis candidate lambda=i nu gives
nu= -c mu/((1/2)mu+c), and the remaining equation reduces to c=mu/2.
There is no zero root. At small c>0, the fast pair approaches -mu(1/2 +/- i), and the slow pair has negative real part because Re(1/(1/2+i)-1)=-3/5.
This is the only positive-c crossing; in the large-c limit the finite pair approaches mu(1/2 +/- i) and is unstable. Thus each mode is stable exactly when tau>2/mu. At equality it has imaginary roots, so strict exponential stability is lost. This is a linear stability crossing, not a proved nonlinear Hopf bifurcation.

For both covariance modes together:
- input gradient: locally exponentially stable for every finite tau>0;
- output detach: locally exponentially stable iff tau>10;
- at tau=10 the slow covariance mode is marginal in the linearization;
- below 10 at least one mode is unstable.

This already gives a directional distinction although both feedback matrices coincide. Noncommutation across inputs is therefore sufficient complexity, not a necessary condition. Rotational/nonsymmetric output feedback is enough to defeat the scalar time-rescaling reduction.

## 4. An exact-rational noncommuting witness

Now use B1=B0 and B2=B0+(1/10)diag(1,-1).
Their commutator is [[0,1/5],[1/5,0]], so they do not commute. With fixed reference parameters, both rules are locally stable in this example: H_input is positive definite, and the symmetric part of H_stop is positive definite. The instability below is a property of the coupled EMA system.

At tau=1 and tau=20, all entries of the 8-by-8 local matrix are rational. The verification script computes its characteristic polynomial by Faddeev-LeVerrier with Fraction arithmetic, verifies Cayley-Hamilton exactly, and applies a nondegenerate exact Routh array. Full coefficients and rational first columns are saved in vector_stability_verification.json.

| Rule | tau | Routh first-column signs | Right-half-plane roots | max Re(lambda), approximate |
|---|---:|---|---:|---:|
| input gradient | 1 | +++++++++ | 0 | -0.09933055 |
| output detach | 1 | +++++-+-+ | 4 | +0.16592210 |
| input gradient | 20 | +++++++++ | 0 | -0.02133731 |
| output detach | 20 | +++++++++ | 0 | -0.02156458 |

The sign certificates, not rounded eigenvalues, establish the local classifications. No full interval boundary is claimed for this noncommuting perturbation. Classical Routh analysis is not itself a novel contribution.

## 5. Discrete updates and finite nonlinear checks

For student-first EMA, eta=1/10 and alpha=tau/(tau+eta), use

W_next=W-eta G(W,V),
V_next=alpha V+(1-alpha)W_next.

Its local matrix is

[ I-eta H, eta K ;
  (1-alpha)(I-eta H), alpha I+(1-alpha)eta K ].

The coefficient alpha is a chosen rational discretization, not a verified IT3 default. For each rational case the script computes the exact characteristic polynomial p(z), transforms it to (1-s)^8 p((1+s)/(1-s)), and applies the exact Routh test. The transformation sends the open unit disk to the open left half-plane. The endpoint z=-1 and degenerate pivots are explicitly excluded by checks.

The noncommuting cases retain the same stable/unstable distinction:
input/tau1 spectral radius 0.99023905;
stop/tau1 1.01767378 (four roots outside the unit disk);
input/tau20 0.99786304;
stop/tau20 0.99782237.
This verifies these specific step sizes, not arbitrary discrete learning rates.

Additional checks:
- 8 exact ODE certificates and 8 exact transformed discrete certificates (delta=0 or 1/10, tau=1 or 20, both rules).
- 42 numerical mode tests agree with the analytic rotational boundary away from equality.
- Central finite differences of the nonlinear field agree with M to 1.25e-12.
- Five full-architecture finite-difference checks of the actual squared loss agree with the input-gradient field to 3.2e-12.
- Matched small student/reference perturbations were integrated in the actual nonlinear architecture with RK4; halving the step from 0.02 to 0.01 changes final risk by at most 2.8e-9 relatively in the three selected runs.
- Input/tau1 recovers, stop/tau1 grows, and stop/tau20 recovers after a small transient. These runs corroborate the local result and do not establish a global basin.

The initial eigenvalue scan was exploratory. The nonlinear recovery/growth predictions were written into the verification script before its nonlinear runs. No claim of preregistered discovery is made.

## 6. Limits, rejected extrapolations, and next gate

1. The correct baseline h is constructed, not learned in this cycle. This solves existence only, not empirical reachability or task identifiability.
2. The feedback Jacobians A=I-B are chosen. They are not locally idempotent projections: if a differentiable reference were truly idempotent on a neighborhood of a fixed point, its derivative there would obey A^2=A. Our architecture matches the two supervised consistency points but does not impose global idempotence. Approximately projection-like learned Jacobians could invalidate this particular rotational mechanism.
3. Zero perturbation stays fixed. This is local recovery/instability under parameter perturbations, not a guarantee of improvement under an unseen covariate or label shift.
4. No forgetting, drift/tracking, stochastic, optimizer-state, or network-wide guarantee follows. A slower teacher can have transient risk growth even when asymptotically stable.
5. The squared loss differs from the audited regression notebooks' absolute losses. Nonsmooth losses require a separate analysis.
6. Generic delayed feedback and rotational stability are classical control phenomena. The result is a plausible mechanism candidate, not established new prior art.

The next highest-value action is a small pretraining/measurement experiment, not more arbitrary toy-model expansion: train a two-output dual-input regressor with the original zero/label input training recipe, inspect the reference-input Jacobians at its predictions (including ||A^2-A||, skew part, spectrum and residual), and test whether useful near-correct states exhibit the geometry needed by this witness. Match the exact squared surrogate first and explicitly distinguish it from original absolute-loss experiments. Stop or switch to transverse/projection-like analysis if the required rotational geometry is absent. Any claimed boundary must use the full parameter-to-output and reference derivatives, not an input Jacobian alone.

## Source record

- IT3, arXiv:2410.04201v2, accessed 21 October 2026: https://arxiv.org/html/2410.04201v2 . Sections 3.2-3.3 motivate frozen/EMA reference differentiation; the construction and result above are not claimed by this paper.
- D. Rowell, MIT 2.004, Lecture 25 (2008), Section 2.1: https://ocw.mit.edu/courses/2-004-dynamics-and-control-ii-spring-2008/e098a44ee402a52174eb0f205c5d542f_lecture_25.pdf . Used only for the classical exact Routh-array sign criterion.
- How to Scale Your EMA, arXiv:2307.13813v3: https://arxiv.org/abs/2307.13813 . The abstract was checked; its proofs were not reviewed. It is a reminder that EMA scaling is established prior work, not evidence for or against this particular stability witness.
- Prior cycle's implementation audit and closest-work notes remain the source for those claims. The limited search in this cycle did not establish novelty.

Reproducible files: verify_vector_stability.py, vector_stability_verification.json, gradient_verification.json. All code runs locally with numpy and the Python standard library.

