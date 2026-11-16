# Initial result: self-distillation with a learned scalar predictor

Date: 7 October 2026. Status: analytical working result with symbolic and numerical checks. No claim of publication novelty. The matrix convergence extension is not proved here.

## Main conclusions

For two scalar encoder coefficients and one learned predictor, with shared positive regularization:

1. Every trajectory is bounded and exists for all positive times, at every positive predictor speed.
2. Equilibria are independent of predictor speed. Away from the fold, their local stability classification is also independent of positive predictor speed: collapse and the upper nonzero branch are exponentially stable; the lower branch is a saddle.
3. An explicit composite Lyapunov certificate proves convergence to a single equilibrium for every initial condition when predictor speed exceeds a computable sufficient bound. The bound is conservative, not a necessary stability threshold.

These are results for the regularized self-distillation flow discussed in PEIRA, not the flow of PEIRA's new trace-based objective. Convergence does not guarantee noncollapse or semantic usefulness.

## 1. Precise model and normalization

Choose a canonical feature pair phi(X), psi(Y) with E[phi^2] = E[psi^2] = 1 and E[phi psi] = c, where 0 <= c <= 1. The functions may be nonlinear in the raw inputs. Set U(X) = u phi(X), V(Y) = v psi(Y). The coefficient state is (u,v,p) in R^3.

The symmetric predictor regression is

$$
R(p;u,v)=\frac12\mathbb E[(pU-V)^2+(pV-U)^2]+\frac\lambda2p^2.
$$

Encoders differentiate only through their respective prediction inputs, with the other view detached, and each receives regularization lambda/2 times its squared coefficient. Encoder learning speed is one; predictor learning speed is gamma > 0. Thus

$$
\begin{aligned}
\dot u&=-(p^2+\lambda)u+pcv,\\
\dot v&=-(p^2+\lambda)v+pcu,\\
\dot p&=\gamma[2cuv-p(u^2+v^2+\lambda)].
\end{aligned} \tag{1}
$$

Assume lambda > 0 throughout. Separate regularizers, stochastic updates, EMA teachers and discrete-time training are outside this result. Even though each encoder coefficient is scalar, the joint flow is three-dimensional.

Write x=(u,v)^T, n=||x||^2, m=n+lambda, S=[[0,c],[c,0]], sigma=x^T S x=2cuv, and h(x)=sigma/m. Notice |sigma| <= cn and |h| <= c.

## 2. Boundedness for every predictor speed

**Proposition 1.** For every finite initial state and gamma > 0, the solution of (1) exists globally and remains bounded.

**Proof.** The predictor obeys pdot = gamma m(h-p). Since |h| <= c, the interval [-Pbar,Pbar] is invariant, where Pbar=max(|p(0)|,c). Define D=n-p^2/gamma. Direct differentiation gives

$$
\dot D=-2\lambda n+2\lambda p^2
       =-2\lambda D+2\lambda(1-1/\gamma)p^2. \tag{2}
$$

Let D0=n(0)-p(0)^2/gamma. The integrating-factor formula implies

$$
D(t)\le\max\{D_0,(1-1/\gamma)_+\bar P^2\}.
$$

Consequently

$$
n(t)\le \frac{\bar P^2}{\gamma}
 +\max\{D_0,(1-1/\gamma)_+\bar P^2\}. \tag{3}
$$

Here z_+=max(z,0). The polynomial field is locally Lipschitz; a bounded solution cannot have finite-time escape. This proves the proposition. For gamma >= 1, the convenient, gamma-independent bound n(t) <= R := n(0)+Pbar^2 follows. At gamma=1, the balance identity is especially simple: D(t)=D0 exp(-2 lambda t).

## 3. All equilibria and exact local stability

The origin is always an equilibrium, with eigenvalues -lambda,-lambda,-gamma lambda; it is locally exponentially stable for every gamma > 0.

At a nonzero equilibrium the encoder equations force v=epsilon u and p=epsilon q, where epsilon is +1 or -1 and q > 0. They give

$$
q^2-cq+\lambda=0,\qquad n=q^2.
$$

If c^2 < 4 lambda, the origin is the only equilibrium. If c^2 > 4 lambda, define

$$
q_\pm=\frac{c\pm\sqrt{c^2-4\lambda}}2.
$$

Every nonzero equilibrium is

$$
(u,v,p)=\left(\tau\frac q{\sqrt2},\epsilon\tau\frac q{\sqrt2},\epsilon q\right),
\quad q\in\{q_-,q_+\},\quad\epsilon,\tau\in\{-1,+1\}. \tag{4}
$$

There are eight nonzero equilibria in this case. At c^2=4 lambda the two branches coincide, giving four nonzero equilibria, each nonhyperbolic. We do not assign nonlinear stability to this fold from linearization alone.

For the representative epsilon=tau=1, use modal coordinates r=(u+v)/sqrt(2), s=(u-v)/sqrt(2). At (r,s,p)=(q,0,q), the Jacobian is

$$
J=\begin{pmatrix}
0&0&q(c-2q)\\
0&-2cq&0\\
2\gamma q(c-q)&0&-\gamma cq
\end{pmatrix}. \tag{5}
$$

The transverse eigenvalue is -2cq. The remaining two eigenvalues have characteristic polynomial

$$
z^2+\gamma cq\,z-2\gamma q^2(c-2q)(c-q). \tag{6}
$$

At q=q_+, its linear coefficient and constant term are positive, so both roots have negative real parts. At q=q_-, the constant term is negative, so there is one positive and one negative root. Sign symmetries transfer this classification to all equilibria in (4).

**Proposition 2.** For every gamma > 0 and c^2>4 lambda, the upper branch is locally exponentially stable and the lower branch is a saddle. No equilibrium changes this local stability classification as gamma varies over positive values. Transient rates and the basins may still change.

## 4. Exact reduced certificate and metric identity

Set p=h(x), and denote the resulting encoder velocity by f0(x). Specialize the reduced certificate to

$$
V_0(x)=\frac13(n+\lambda)^3-\frac12\sigma^2,
\qquad g=\nabla V_0=2(m^2x-\sigma Sx). \tag{7}
$$

A direct algebraic identity is

$$
f_0=-K(x)g,\qquad K(x)=\frac{mI-xx^T}{2m^2}. \tag{8}
$$

The radial eigenvalue of K is lambda/(2m^2); the perpendicular eigenvalue is 1/(2m). At x=0 they coincide. Hence K is positive definite everywhere. In particular,

$$
\dot V_0=-g^TKg,\quad
-\dot V_0\ge 2\lambda\|f_0\|^2,\quad
\|g\|\le\frac{2m^2}{\lambda}\|f_0\|. \tag{9}
$$

For the first inequality, the largest eigenvalue of K is at most 1/(2lambda), so ||Kg||^2 <= (1/(2lambda)) g^T K g. The second uses the smallest eigenvalue of K. These are exact estimates for our physical-time flow; there is no implicit time rescaling.

## 5. The moving-optimum error and composite certificate

Define e=p-h(x), and write f(x,p) for the full encoder velocity. Then

$$
f=f_0+\Delta,\quad
\Delta=-(2he+e^2)x+eSx,
$$

$$
\dot e=-\gamma m e-\nabla h(x)^T f,
\qquad \nabla h=\frac{2(Sx-hx)}m. \tag{10}
$$

Thus contraction for frozen encoders does not by itself prove tracking: the moving-optimum term must be controlled.

Fix a>0 and form

$$
W(x,p)=V_0(x)+\frac a2(p-h(x))^2. \tag{11}
$$

For gamma >= 1, Proposition 1 gives n<=R=n(0)+Pbar^2 and |e|<=E=Pbar+c. Valid explicit constants are

$$
b=\sqrt R(3c+E),\qquad
H=\frac{2c}{\sqrt\lambda},\qquad
L=\frac{2(R+\lambda)^2}{\lambda}. \tag{12}
$$

They ensure ||Delta||<=b|e|, ||nabla h||<=H and ||g||<=L||f0||. The global bound for H follows from ||nabla h|| <=4c sqrt(n)/(n+lambda) and sqrt(n)/(n+lambda)<=1/(2sqrt(lambda)).

Writing z=||f0|| and differentiating (11),

$$
\dot W\le-2\lambda z^2+(Lb+aH)z|e|-a(\gamma\lambda-Hb)e^2.
$$

Young's inequality yields

$$
\dot W\le-\lambda z^2-\delta e^2,
\qquad
\delta=a(\gamma\lambda-Hb)-\frac{(Lb+aH)^2}{4\lambda}. \tag{13}
$$

**Theorem 3 (constructive sufficient-speed convergence).** Compute R,E,b,H,L from the initial condition as above. If

$$
\gamma>\max\left\{1,\ \frac{Hb}{\lambda}
                   +\frac{(Lb+aH)^2}{4a\lambda^2}\right\}, \tag{14}
$$

then every trajectory from that initial condition converges to a single equilibrium of (1).

**Proof.** Proposition 1 supplies a compact invariant enclosing region, independently of W, so the derivative estimates are not circular. Also V0 >= n^3/3-c^2 n^2/2 >= -c^6/6. Thus W is bounded below. Equation (13) makes its derivative zero only when f0=0 and e=0, which are exactly full equilibria. LaSalle's invariance principle puts the omega-limit set in that equilibrium set. Equations (4) and the fold case show that this set is finite for every lambda>0. A bounded continuous trajectory has a nonempty connected omega-limit set (the intersection of nested connected compact tail closures); a connected subset of a finite set is one point. Therefore the trajectory converges to a single equilibrium. This last step is specific to the scalar model and does not directly transfer to rotational matrix equilibrium families.

The estimate is sufficient and initial-condition dependent. No global convergence claim for arbitrary gamma is proved here. The first-exit argument used in our earlier conversational derivation is also valid, but the independent boundedness identity above makes it unnecessary.

## 6. Checks and their limits

`verify_scalar.py` checks four exact SymPy identities: reduced metric, balance, encoder perturbation, and modal Jacobian. It also checks the certificate inequalities at 10,000 deterministic random states, branch eigenvalues at gamma=0.01,1,100, four stiff-integrator trajectories, and the matrix balance/tracking identities at 1,000 nonsymmetric matrix states. Results are in `scalar_verification.json`. Numerical checks complement the written proof; sampling does not prove a universal inequality or a matrix convergence theorem.

Example: c=0.8, lambda=0.1, a=1 and (u0,v0,p0)=(0.3,0.2,0.5). Formula (14) gives the speed bound 84840.57682006386. The checked speeds 0.1,1,10 and 169681.1536401277 all approach the upper equilibrium near (0.45604779,0.45604779,0.64494897). At speed 0.1 the terminal velocity norm after time 500 is about 8.81e-8. This shows the sufficient bound can be very loose; four trajectories do not establish all-speed global convergence or exclude other basins.

Optional tuning: with R,E fixed, minimize the speed estimate over a. If Lb>0 and H>0, a=Lb/H minimizes (Lb+aH)^2/a. This changes the bound to gamma>max{1,Hb/lambda+LbH/lambda^2}. It still uses coarse uniform estimates. Degenerate zero constants should be treated directly rather than divided by.

## 7. The matrix research target

Choose paired canonical feature vectors with identity within-view second moments and cross moment C=diag(c1,...,cm). Let U=A phi, V=B psi, A,B in R^(d x m), and P in R^(d x d). Use shared lambda>0. The full coefficient model is

$$
\begin{aligned}
\dot A&=-(P^TP+\lambda I)A+P^TBC,\\
\dot B&=-(P^TP+\lambda I)B+P^TAC,\\
\dot P&=\gamma[\Sigma-P(N+\lambda I)],\\
N&=AA^T+BB^T,\qquad\Sigma=BCA^T+ACB^T.
\end{aligned} \tag{15}
$$

For a genuinely invariant canonical truncation, these are canonical functions of the cross-view conditional operators, not just arbitrary correlated features with matching moments. A finite basis projection also defines (15), but should be described as a projected model.

Two identities already extend without commutativity assumptions:

$$
h=\Sigma(N+\lambda I)^{-1},\quad E=P-h,\quad
\dot E=-\gamma E(N+\lambda I)-Dh[\dot A,\dot B],
$$

$$
Dh[Z]=D\Sigma[Z](N+\lambda I)^{-1}
-h\,DN[Z](N+\lambda I)^{-1}, \tag{16}
$$

and, writing n=||A||_F^2+||B||_F^2,

$$
\frac d{dt}\left(n-\frac{\|P\|_F^2}{\gamma}\right)
=-2\lambda n+2\lambda\|P\|_F^2. \tag{17}
$$

Equation (17) follows by taking traces of (15); it does not imply boundedness until predictor size is controlled. The scalar bound |h|<=c does not automatically become a matrix operator-norm bound: Sigma(N+lambda I)^(-1) need not be symmetric or normal.

The proposed matrix certificate is

$$
W=\frac13\operatorname{Tr}(N+\lambda I)^3
-\frac12\operatorname{Tr}(\Sigma^2)+\frac a2\|P-h\|_F^2. \tag{18}
$$

**Next precise question:** on the full d=m=2 system with c1>c2>0 and no imposed diagonalization or predictor symmetry, can one prove an explicit, initial-condition-dependent sufficient-speed convergence theorem using (18), or exhibit why that particular certificate fails? Required work: verify reduced derivative/relative bounds with our normalization, control the matrix tracking coupling, establish an invariant bounded region, then handle rotational equilibrium families when proving point convergence. The scalar metric K is not asserted to generalize unchanged.

Finite-speed matrix local stability is another target: analyze perturbations between canonical modes, not only independent diagonal scalar blocks. The scalar case shows that a speed-induced local instability cannot be assumed in advance.

## 8. Position relative to prior work

These references were checked on 7 October 2026:

- [Tian, Chen and Ganguli (2021)](https://proceedings.mlr.press/v139/tian21a.html): finite learned predictors and relative predictor rates already occur in linear noncontrastive SSL analysis.
- [Ponce, Terver, Hebert and Arbel, Dual Perspectives](https://arxiv.org/html/2507.01028v3): analyzes coupled linear encoder/predictor learning and stability. This is a close baseline; exact theorem-by-theorem comparison remains necessary before claiming novelty.
- [Arbel, Terver and Ponce, PEIRA](https://arxiv.org/html/2605.17671v1): the reduced certificate originates here; section 7 explicitly leaves suboptimal-predictor analysis using singular perturbations for future work.
- [Chen, Sun, Xiao and Yin, STABLE (2022)](https://proceedings.mlr.press/v151/chen22e.html): tracking-error Lyapunov constructions in bilevel optimization are established methodology, not a new principle by themselves.

Our scalar results provide an audited baseline and a concrete construction. They are not currently positioned as independently publishable novelty. The potential contribution is a sharp model-specific matrix theorem and a reusable set of checkable conditions, with an explicit distinction between convergence, stability and representation quality.
