# A tractable theoretical target: EMA consistency can improve while old-task risk increases

Date: 21 October 2026. Focused research pass. The proposal and existing scalar PEIRA certificate were read without modification. The calculation below is a newly derived diagnostic for this project, **not a claim of publication novelty or a theorem about the actual IT3 implementation**. References were checked on the date above.

## 1. Can one explicit shared-parameter model separate stability, consistency, and retention?

### Takeaway

Yes. In the affine model below, both the correctly differentiated frozen-reference loss and an explicitly detached-output comparison give stable EMA dynamics. Their adaptation discrepancy and current-task error decrease monotonically, while old-task error increases monotonically. For every finite positive EMA timescale, the final forgetting is the same: changing the timescale changes transients, not the endpoint.

### Cited Findings

- IT3 v1 defines a dual-input predictor, uses a frozen reference in episodic adaptation, carries adaptive weights forward online, and replaces the reference by EMA online. Its online setting expressly accepts loss of original-distribution performance in favor of present performance. This motivates, but does not establish, the model analyzed here. [IT3 v1, Sections 3.1–3.3](https://arxiv.org/html/2410.04201v1#S3)

### Inferences

Everything in this subsection is an explicit proposed model or a calculation proved below; it is not attributed to a paper.

#### Model and information budget

Let the input dimension be two, with unit feature vectors x_A and x_B and overlap c=x_A^T x_B, where 0<|c|<1. Both domains use the **same** adaptive parameter theta in R^2. Fix a known input-dependent feedback gain a(x) with

\[
a(x_A)=0,\qquad a(x_B)=1-\beta,\qquad \beta>1.
\]

The architecture is affine in the auxiliary prediction z:

\[
f_\theta(x,z)=\theta^T x+a(x)z,
\quad q_\theta(x)=f_\theta(x,0)=\theta^T x,
\quad T_\phi(x,z)=\phi^T x+a(x)z.
\]

The gain is fixed; only theta is adapted, and phi follows EMA. This is a restricted same-architecture student/reference model. It is not two independent parameters assigned to two inputs. A smooth a(x) can be chosen to attain these two values; only the two-point support is used in the proof.

Initialize theta(0)=phi(0)=theta_0. Define the source label y_A=theta_0^T x_A and the new-domain label y_B=0. Let u_0=theta_0^T x_B be nonzero. Labels are used **only for evaluating risks**, never in the adaptation equation. A is exactly idempotent at initialization, since f_theta(x_A,z) ignores z. At B the initial consistency residual is nonzero.

During the B phase let rho_B be a point mass at x_B. Put u=theta^T x_B, v=phi^T x_B, and r=v-beta u. The squared surrogate is D=r^2/2. The derivative paths matter:

\[
\partial_z T_\phi(x_B,z)=1-\beta,
\qquad G_{\rm frozen}=-\beta x_Br,
\qquad G_{\rm detached}=-x_Br.
\]

Here `frozen` means phi is held fixed **during the theta derivative**, while the input derivative through T remains active. It does not mean phi stays fixed throughout the online run. `detached` means the entire reference output is stopped, so that input derivative is removed. These are different computational graphs sharing a forward value.

Both can be written

\[
\dot\theta=k x_B(v-\beta u),\qquad
\tau\dot\phi=\theta-\phi,
\quad k=\begin{cases}\beta&\text{frozen-reference backprop},\\1&\text{detached-output comparison},\end{cases}
\quad\tau>0.
\tag{1}
\]

Thus the adapted two-dimensional parameter and the two-dimensional teacher form a four-dimensional ODE. Reduction to two variables is exact:

\[
\dot u=k(v-\beta u),\qquad
\dot v=(u-v)/\tau,\qquad (u(0),v(0))=(u_0,u_0).
\tag{2}
\]

The component of theta orthogonal to x_B is constant; the corresponding component of phi agrees with it throughout under matched initialization. In particular,

\[
\theta(t)=\theta_0+[u(t)-u_0]x_B,
\quad
\phi(t)=\theta_0+[v(t)-u_0]x_B.
\tag{3}
\]

#### Proposition (proved diagnostic baseline)

For beta>1, k>0, tau>0, and matched initialization with u_0 nonzero:

1. The trajectory is bounded and converges to theta_infinity=phi_infinity=theta_0-u_0 x_B.
2. D(t) and R_B(t)=u(t)^2/2 strictly decrease to zero at every finite positive time.
3. The source risk is
   \[
   R_A(t)=\tfrac12 c^2[u_0-u(t)]^2,
   \qquad R_A(\infty)=\tfrac12 c^2u_0^2.
   \tag{4}
   \]
   It increases strictly for positive time. Its endpoint is independent of tau and k.
4. Consequently convergence, decreasing consistency, and improving current accuracy do not imply retention, even together, for this model.

**Proof.** The reduced Jacobian is

\[
M=\begin{pmatrix}-\beta k&k\\1/\tau&-1/\tau\end{pmatrix}.
\]

Its characteristic polynomial is z^2+(beta k+1/tau)z+k(beta-1)/tau. Its two eigenvalues are -s and -f, where

\[
s,f=\frac{\beta k+1/\tau\mp\sqrt{(\beta k-1/\tau)^2+4k/\tau}}2,
\qquad 0<s<f.
\tag{5}
\]

The full four-dimensional Jacobian additionally has eigenvalues 0 and -1/tau, corresponding to the conserved student component and teacher tracking perpendicular to x_B. There is therefore a line of equilibria, not an isolated globally asymptotically stable point. Formula (3) identifies the particular limiting point, without incorrectly inferring point convergence from a negative semidefinite certificate.

Let h=k(beta-1). The polynomial p(z)=z^2-(beta k+1/tau)z+k(beta-1)/tau obeys p(h)=-k^2(beta-1)<0. Thus s<h<f. The exact solution is

\[
u(t)=u_0\left[w e^{-st}+(1-w)e^{-ft}\right],
\quad w=\frac{f-h}{f-s}\in(0,1).
\tag{6}
\]

It follows that u(t) keeps the sign of u_0, its absolute value strictly decreases, and u(t) tends to zero. Since r=dot u/k,

\[
r(t)=-\frac{u_0}{k}\left[ws e^{-st}+(1-w)f e^{-ft}\right].
\tag{7}
\]

The magnitude of r also strictly decreases. Equations (6)–(7) prove the discrepancy and current-risk assertions. Equation (3) gives q_theta(x_A)-y_A=c[u(t)-u_0], which proves (4). Finally v=r+beta u tends to zero, so (3) proves convergence of both parameters. All assertions follow from explicit finite sums of decaying exponentials. QED.

#### Concrete numbers

Choose x_A=(1,0), x_B=(0.6,0.8), theta_0=(1,0), beta=2. Then c=u_0=0.6. For either backward rule and every finite tau>0,

\[
\theta_\infty=(0.64,-0.48),\quad
q_\infty(x_A)=0.64,\quad q_\infty(x_B)=0,
\quad R_A(\infty)=0.0648.
\]

Both tasks are simultaneously realizable, because x_A and x_B are independent; this forgetting is not forced by inconsistent labels or inadequate parameter dimension. It is caused by the update direction selected by the current-input loss.

#### Why a large teacher timescale is not permanent memory

For a genuinely fixed reference phi=theta_0, (2) becomes dot u=k(u_0-beta u), and u_infinity=u_0/beta. Its final source risk is c^2u_0^2(1-1/beta)^2/2, while its current risk remains u_0^2/(2 beta^2). For the example, the risks are 0.0162 and 0.045. EMA eventually moves further along the same current-accuracy/retention tradeoff. The limits tau→infinity and t→infinity do not commute. This comparison does not establish a finite-time monotonic ordering in tau.

There is another singular limit: beta→1 from above makes the initial discrepancy arbitrarily small, but does not change eventual forgetting for finite tau. At beta=1 exactly, matched initialization is stationary. Any bound on eventual retention based only on the initial discrepancy must account for this loss of contraction rate.

### Gaps

- This is deliberately a counterexample model, not a faithful empirical surrogate. The input-dependent negative feedback on B and the choice y_B=0 are explicit assumptions that make it analyzable. They must be tested against measured IT3 feedback Jacobians before drawing architectural conclusions.
- Full neural pretraining, stochasticity, discrete EMA, optimizer state, normalization layers, and nonquadratic losses are absent. Source idempotence alone does not constrain off-source a(x).
- Frozen and detached fields differ only by a positive scalar here. At equal progress on B their retention is identical. This model cannot establish a direction-dependent advantage of either backward rule.
- No novelty claim is justified by the proposition. The two-mode algebra is an illustrative baseline and could fit an appendix or motivating figure.

## 2. What does this imply for recovery and a retention mechanism?

### Takeaway

Returning to a source domain need not recover old accuracy when the unsupervised consistency criterion cannot distinguish the old solution from the forgotten one. A constraint preserving old output directions solves this particular counterexample, but that requires retained information and is closely related to established projection methods.

### Cited Findings

- OGD projects new-task updates onto directions that preserve previous predictions; its mechanism already establishes the general idea of protecting old output directions. A projected version of this toy model is not a new algorithmic principle. [Farajtabar et al., AISTATS 2020](https://proceedings.mlr.press/v108/farajtabar20a.html)
- Linear continual-learning work already derives exact forgetting expressions and relates sequential fitting to alternating projections and Kaczmarz methods. [Evron et al., COLT 2022](https://proceedings.mlr.press/v178/evron22a.html)

### Inferences

These are further calculations within the same model, not general claims about real continual learners.

**No recovery after a fully converged B phase.** At the limiting state theta=phi, returning to A gives r_A=(phi-theta)^T x_A=0 because a(x_A)=0. Thus both backward rules are stationary, while R_A is positive. The consistency signal cannot identify which of its many source-domain fixed points has the correct old label. This shows a concrete failure of an inference from “A→B→A” alone to recoverability; the adaptation signal must retain or reveal task-relevant information.

**Finite B phase.** If the switch back occurs at finite time T, let p=theta^T x_A and q=phi^T x_A. On A, dot p=q-p and tau dot q=p-q, so p+tau q is conserved and p-q decays. The final source prediction after return is

\[
p_\infty^{\rm return}=y_A+c\left[\frac{u(T)+\tau v(T)}{1+\tau}-u_0\right].
\tag{8}
\]

Teacher lag can therefore produce partial recovery. For u_0>0, both u(T),v(T) are positive and below u_0 at positive T, with v(T)>u(T). It does not restore the original prediction completely at T>0; as T tends to infinity it gives no recovery. The sign-reversed case is symmetric. Formula (8) uses the same tau during both phases.

**An available capacity check.** Because |c|<1, the direction

\[
d=\frac{(I-x_Ax_A^T)x_B}{1-c^2}
\quad\text{satisfies}\quad x_A^Td=0,\quad x_B^Td=1.
\]

Replacing the student direction x_B in (1) by d would preserve the old prediction exactly while leaving the reduced u,v ODE unchanged; phi still follows ordinary EMA. This is possible because the tasks are jointly realizable. However d requires knowledge of the old feature direction. Treat it as an OGD-style oracle baseline or explicitly budget a stored old gradient/subspace. Do not describe it as a free, memoryless fix. Near |c|=1, its norm 1/sqrt(1-c^2) blows up, exposing the conditioning cost.

### Gaps

- The recovery result concerns output error, not whether a learned representation contains recoverable information. This linear model has no separately learned representation.
- In a neural experiment, retaining an old Jacobian sketch costs memory and depends on where it is evaluated; its error and drift need their own analysis.
- The counterexample does not prove all possible memoryless learners fail: it proves failure for the specified learner and consistency signal.

## 3. What should the first thesis theorem actually target, and where is prior work close?

### Takeaway

Use this proposition to eliminate the unsupported premise that slow EMA itself guarantees retention. The next tractable research question is a finite-dimensional, vector-output IT3-inspired model where frozen-reference backprop and output detachment change update directions. Fixed-domain stability and current-task behavior should be primary; tracking can follow once a moving target is defined. Retention is a conditional secondary diagnostic, compared at matched current-task progress. A publication case requires a result specific to the feedback Jacobian and EMA interaction, plus verification in an audited IT3 implementation; a generic linear forgetting formula will not suffice.

### Cited Findings

- Teacher-student continual-learning theory already studies how feature and readout similarity affect initial and long-time forgetting. [Lee, Goldt and Saxe, ICML 2021](https://proceedings.mlr.press/v139/lee21e.html)
- A general linear-regression SGD analysis covers both underparameterized and overparameterized regimes and examines task sequence, covariance spectra, and step size. [Ding et al., ICML 2024](https://proceedings.mlr.press/v235/ding24c.html)
- Recent theory studies continual linear models with jointly realizable tasks, proves equivalence of fitting a task to an SGD step on a modified objective, and obtains forgetting rates under randomized task orderings. [Evron et al., ALT 2026](https://proceedings.mlr.press/v313/evron26a.html)
- A 2026 preprint claims an interference-energy characterization of forgetting in the frozen-feature regime and a method based on protecting conflicting directions. Its abstract was inspected, not its proofs; treat it as a priority related-work check, not an established theorem imported here. [Störk, arXiv:2607.09202](https://arxiv.org/abs/2607.09202)

### Inferences

#### A concrete next theorem question, not a promised result

Use vector predictions with f_W(x,z)=Wx+A_x z, fixed matrices A_x, adapted W, and EMA V. Let B_x=I-A_x and r=Vx-B_xWx. The exact population update fields are

\[
\dot W_{\rm frozen}=\mathbb E[B_x^T r x^T],\qquad
\dot W_{\rm detached}=\mathbb E[r x^T],\qquad
\tau\dot V=W-V.
\tag{9}
\]

This follows by differentiating the squared discrepancy while holding V fixed. Start with output dimension two, two nonorthogonal input features, and two feedback matrices that cannot be simultaneously diagonalized. This prevents the study from splitting into independent copies of the scalar calculation.

For column-vectorization w=vec(W), v=vec(V), the fixed-distribution system has block matrix

\[
\frac{d}{dt}\begin{pmatrix}w\\v\end{pmatrix}
=\begin{pmatrix}-H&K\\I/\tau&-I/\tau\end{pmatrix}
\begin{pmatrix}w\\v\end{pmatrix},
\tag{10}
\]

where frozen-reference backprop uses

\[
H=\mathbb E[(xx^T)\otimes(B_x^TB_x)],\qquad
K=\mathbb E[(xx^T)\otimes B_x^T],
\]

and output detachment uses

\[
H=\mathbb E[(xx^T)\otimes B_x],\qquad
K=\mathbb E[(xx^T)\otimes I].
\]

An achievable **first extension** is: characterize, in a specified two-matrix family, when changing tau or the backward rule changes stability or transient old-output amplification; derive a sharp bound or matching counterexample at equal current-task risk. Compute conserved directions and equilibrium manifolds explicitly. Do not simply label the whole block matrix Hurwitz and call that the theorem. A useful theorem must translate its condition into feedback geometry and provide an interpretable retention consequence.

The proof plan is finite and falsifiable: (i) choose the two feedback matrices from an audit of actual reference-input Jacobians; (ii) derive the exact reduced block system; (iii) analyze its nontrivial coupled modes without assuming commutativity; (iv) evaluate the old-output operator on its state transition and either bound transient growth or exhibit an explicit extremal initialization; (v) verify whether the claimed effect survives discrete updates. The key null outcome is valuable: if both fields differ only by time parametrization in the chosen family, broaden the family before an empirical comparison.

#### Thesis versus first-paper scope

- **Useful master's starting point:** the fully audited model above, a vector-output extension with a provable stability/retention distinction, a reconstruction of IT3 backward paths, and a controlled switch experiment. Keep the older learned-predictor PEIRA certificate as background expertise, not as evidence that the EMA theorem follows.
- **Possible paper claim, conditional on new evidence:** a feedback-Jacobian condition predicting a specific EMA-induced retention or transient-amplification failure, accompanied by a modification addressing that diagnosed failure under a stated memory budget, with matched adaptation-performance comparisons. Novelty depends on a detailed comparison with the papers above and related continual-test-time-adaptation work.
- **Do not sell as novelty:** a scalar eigenvalue calculation; the identity dR_A/dt=-<grad R_A,G>; consistency versus correctness; projection away from old gradients; or a standard composite Lyapunov/tracking construction by itself.

#### Reproducible verification

`verify_affine_ema.py` in this folder evaluates the closed-form ODE identities and monotonicity at deterministic times for beta in {1.01,1.2,2,5}, both k choices, and tau in {0.01,0.1,1,10,100}. It writes `affine_ema_verification.json`. These checks corroborate the handwritten formulas; they do not replace the proof and do not test a neural implementation.

The run on 21 October 2026 passed all 40 parameter combinations at 502 times each. Maximum absolute closed-form ODE residual was approximately 1.06e-13.

### Gaps

- Equation (10) is a derived starting system, **not** a proved stability or retention theorem for noncommuting matrices. No fabricated positive result should be added to the proposal.
- Actual code paths, adapted parameter subsets, and EMA scheduling in the selected IT3 code version were not audited in this focused pass.
- Abstract-level prior-art screening does not establish novelty. In particular the newer interference-energy preprint must be read and assessed before writing a novelty paragraph.
- Discrete-time monotonicity cannot be inferred from the continuous-time proof. A step-size condition or direct discrete analysis is still required.
