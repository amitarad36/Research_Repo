# A scalar nonsmooth baseline: obstruction, unique inclusion and discrete bound

28 October 2026, cycle 11. Completed the unchanged cycle-10 analytic protocol. These are checked application calculations using established discontinuous-system and monotone-operator ideas. They are not a novel theorem claim, a trained-network result, or a repair of cycle 10's numerically unresolved risk experiment.

## 1. One forward graph and its backward rules

Take f_theta(z)=h+theta+a*z, h=1,a=1/4, theta_0=phi_0=0; b=1-a=3/4 and c=a*h=1/4. The evaluator target h is planted, so q_theta=h+theta starts correct and nonzero. It is not an idempotence-trained model. No labels enter any update. Original residual r=phi-b*theta+c; cached input q_0=h gives r_c=phi-theta+c. For mean absolute loss in this scalar model (d=1), negative student-gradient fields are

    F_stop = sign(r),  F_input = b*sign(r),  F_cached = sign(r_c),
    sign(0)=0.

The original full-input gradient differentiates the reference input; the original stopped rule detaches the whole reference output. Caching removes dependence of that input on the current student. Stopped/cached fields coincide wherever their signs agree; initially both are +1. Here A=a is constant, so the full-input field is also constant within a sign region. This special affine fact does not invalidate the nonzero full-input teacher blocks checked in the nonlinear model.

For any of the three fields, the declared student-first rule is

    theta_next=theta+eta*F,
    phi_next=alpha*phi+(1-alpha)*theta_next, alpha=tau/(tau+eta).

With e=theta-phi, it gives exactly

    e_next=alpha*(e+eta*F),
    phi_next-phi=(eta/tau)*e_next.

For cached input, F=sign(c-e), making the lag recurrence autonomous. For the original rules r=c+a*theta-e, so the same reduction does not close in e alone. Their EMA dynamics are not obtained merely by rescaling time while holding tau fixed.

## 2. Fixed reference: finite arrival or a two-cycle

With phi fixed at zero, each update has the form theta_next=theta+delta*sign(theta_star-theta). Cached: theta_star=c,delta=eta. Original stopped: theta_star=c/b,delta=eta. Original full-input: theta_star=c/b,delta=b*eta. Here b>0.

For any initial theta, if |theta_star-theta_0|/delta is an integer, the iterate reaches the root in that many steps and stays there under sign(0)=0. Otherwise, successive steps toward the root eventually bracket it between two adjacent grid points theta_low and theta_low+delta; every later update alternates between those points. Both their distances to the root are below delta. At the exact root the iterate stays fixed. This follows directly from subtracting delta from the distance before the first crossing and then reversing its sign after crossing.

For the protocol's sole checked eta=1/10, cached input from zero reaches the cycle 1/5,3/10 around c=1/4. This is not an EMA cycle claim. With fixed reference, the pointwise cached ODE has the unique absolutely continuous solution theta(t)=min(t,c) from zero: it can remain at the root because the field there is zero. The moving-reference case below differs.

## 3. Moving reference: the pointwise ODE has no continuation

The cached formal continuous model is

    theta'=sign(c-e), phi'=e/tau,
    e'=sign(c-e)-e/tau, e(0)=0.

The following arguments hold for any tau>0 and 0<c<tau, covering c=1/4,tau=1. Until the first boundary,

    e(t)=tau*(1-exp(-t/tau)), theta(t)=t,
    t_star=-tau*log(1-c/tau).

Thus the boundary e=c is reached in finite time (t_star=0.28768207245... for the fixed model). There is **no absolutely continuous Caratheodory continuation satisfying this pointwise ODE almost everywhere** beyond t_star.

Proof: suppose such a continuation starts from e=c. Set w=e-c and V=w^2/2. The chain rule for absolutely continuous functions gives, almost everywhere,

    V'=-|w|-w^2/tau-c*w/tau
       <= -(1-c/tau)*|w|-w^2/tau <= 0.

Since V initially vanishes and is nonnegative, it must remain zero. Then e=c throughout the continuation, so e'=0 almost everywhere; the specified pointwise field at e=c is -c/tau, a contradiction. This is stronger than absence of a continuously differentiable solution. The obstruction is not a nonuniqueness claim and does not affect existence of the discrete algorithm.

## 4. Explicit inclusion and its unique continuation

Define Sign(v)={sign(v)} for v!=0 and Sign(0)=[-1,1]. The Filippov extension of this scalar field is

    e' in Sign(c-e)-e/tau = -partial Psi(e),
    Psi(e)=|e-c|+e^2/(2*tau).

For two inclusion solutions e_1,e_2 with selections s_j in Sign(c-e_j), the sign graph is decreasing, so

    (1/2)*d/dt |e_1-e_2|^2
      =(e_1-e_2)*(s_1-s_2)-|e_1-e_2|^2/tau
      <= -|e_1-e_2|^2/tau.

This proves uniqueness and contraction of the lag. Existence for the specified initial state is verified directly by

    e(t)=min(tau*(1-exp(-t/tau)),c),
    theta(t)=t for t<=t_star,
    theta(t)=t_star+(c/tau)*(t-t_star) for t>=t_star,
    phi(t)=theta(t)-e(t).

After contact the inclusion selects s=c/tau in [-1,1], giving theta'=phi'=c/tau while e'=0. This selected velocity is **not** the pointwise choice sign(0)=0. It is an explicitly changed solution concept. The scalar field is measurable and locally bounded; the potential's subdifferential is maximal monotone with full domain. Its resolvent can also be checked directly: (I+partial Psi)(e)=(1+1/tau)*e+Sign(e-c), a strictly increasing graph whose range is R. This supplies the classical maximal-monotone assumptions without claiming them for a general stopped neural field.

The standard definitions and existence framework are documented in Cortes, Eqs. (16),(19)-(21), and Proposition 3; the contraction setting is consistent with Benilan--Brezis, Definition 1.1, Lemma 1.1 and Proposition 1.1. Exact versions/pages are in SOURCE_EVIDENCE.md. The explicit formula and elementary proof above verify this application independently.

In this constructed example, zero discrepancy after contact does not stop common student/reference motion. Lag contraction therefore does not imply bounded absolute parameters or preservation of the initially correct prediction. This is a property of the planted example, not an explanation of the trained network's error.

## 5. A uniform lag bound and finite-horizon parameter bounds

Let eta>0, alpha=tau/(tau+eta), C=1+c/tau, and let (theta_n,phi_n,e_n) be the exact cached discrete update from zero. Let (theta(t),phi(t),e(t)) be the explicit inclusion solution above. For every grid index n>=0, t_n=n*eta,

    |e_n-e(t_n)| <= C*eta,
    |phi_n-phi(t_n)| <= eta*(C*t_n/tau+c/tau),
    |theta_n-theta(t_n)| <= eta*(C*(1+t_n/tau)+c/tau).

These bounds hold for all eta>0; no small-step restriction is needed for their validity, although they are informative only at sufficiently small eta on a fixed horizon. They are grid-state bounds, not derivative convergence, convergence of switching times, or a uniform-in-time bound on theta. In the fixed model C=5/4. Replacing t_n by a fixed H bounds all n with t_n<=H. No claim is made for the original full-input/stopped EMA systems or for the neural checkpoints.

Proof, before first discrete contact: write u_n=tau*(1-alpha^n) and v_n=tau*(1-exp(-n*eta/tau)). Let N be the first index u_N>=c; it exists because u_n increases to tau>c. For n<=N, e_n=u_n. With d=eta/tau and p=exp(-d), alpha>=p and

    0<=v_n-u_n=tau*(alpha^n-p^n)
      <=tau*(alpha-p)/(1-alpha)
      =tau*[1-(1+d)*exp(-d)]/d <= eta/2.

The last inequality follows from 1-(1+d)*exp(-d)=integral_0^d s*exp(-s) ds <= d^2/2. For n<N, clipping v_n at c cannot increase its distance above u_n, hence |e_n-e(t_n)|<=eta/2, including the delay between continuous and discrete contact.

Proof, after contact: set L=alpha*(c-eta), U=alpha*(c+eta). We have L<c<U<tau and e_N in [c,U]. The two affine branches of the discrete map and its exact-zero value alpha*c map [L,U] into itself: below c the map lies between alpha*(L+eta)>=L and U; above c it lies between L and alpha*(U-eta)<=U; the zero case is between L and U. Since v_N>=u_N>=c, e(t_n)=c for every n>=N. Consequently

    |e_n-c| <= max(c-L,U-c)=alpha*eta*(1+c/tau) <= C*eta.

Combining the two regions proves the lag bound because C>1/2. Finally, phi_n=(eta/tau)*sum_{j=1}^n e_j and phi(t_n)=(1/tau)*integral_0^{t_n} e(s) ds. The inclusion lag increases from 0 to c, so its right Riemann sum overestimates that integral by at most eta*c. Adding the discrete lag error gives the stated phi bound; theta=e+phi gives the last bound. This includes the boundary without differentiating a discontinuous sign map.

## 6. Verification, limitations and next decision

The byte-identical protocol hash is 0bc6760d30e7787459e794847cc8ace0e6b5fbeb129ae061838169b40f29ec9a. `verify_nonsmooth.py` uses only standard-library exact fractions and 60-digit decimals. It checks the six declared conditions for 12 steps each at eta=1/10 (72 scalar updates, 78 state/field records), exact local loss-gradient identities away from zero, the cached recurrence/sum identity, fixed-reference arrival/cycling cases and invariant-interval branch endpoints. It also substitutes the closed-form inclusion solution and checks the three derived bounds on those illustrative grid points. Runtime .015625 CPU seconds, below 30 seconds. No neural data, labels, new checkpoints, smoothing, eta sweep or neural adaptation was used. The analytic proofs, not these finite checks, establish the all-step statements.

For the EMA illustration, first discrete contact is n=4, whereas continuous contact is t_star approximately .28768. The invariant interval is [3/22,7/22]. Exact rational calculations reproduce all discrete identities; substitution into the inclusion at 60-digit precision has zero recorded residual. Recorded maximal error/bound ratios are .8553 for lag, .2301 for reference and .4212 for student. These are implementation checks, not fitted constants or convergence-rate estimation.

Only standard scalar relay/subgradient facts and an elementary application bound have been established. This closes a foundational gap and warrants correcting the proposal, but **does not justify a first paper by itself**. Cycle 10 remains numerically unresolved. The next bounded question is whether replacing every zero residual sign by an independent interval is actually the correct Filippov closure of a multivariate stopped field. Jointly reachable sign patterns can matter; scalar well-posedness cannot simply be transferred. A faithful closure or a counterexample is needed before a vector Lyapunov claim, and any such calculus result still needs predictive/implementation content to support a paper.
