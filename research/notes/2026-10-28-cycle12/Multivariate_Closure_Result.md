# Faithful closure of absolute-loss adaptation fields

28 October 2026, cycle 12. Completed the unchanged cycle-11 transfer audit. Status: standard calculus applied and proved for the declared graph; exact checked counterexample. Not a novelty certification, an official IT3 failure, or a resolution of cycle 10's neural numerical gate.

## Decision and practical consequence

An independent interval for each zero residual sign is always an outer model in the smooth finite-batch setting below, but is not always the actual Filippov closure of a stopped field. A two-output, globally idempotent frozen reference gives a strict counterexample: the correct stopped dynamics have a unique stationary solution from the matched initialization, whereas the independent-sign model admits a path with increasing current-task error. The apparent failure belongs to the enlarged mathematical model, not to the algorithm.

Two sufficient routes to equality are established. Full row rank of the active residual Jacobian gives equality for any continuous coefficient matrix. More specifically, true partial-gradient fields of the smooth absolute loss have equality without that rank condition. This includes the original full-input-gradient and cached-input rules, with either a fixed reference or continuous EMA state drift. It does not include an arbitrary stop-gradient field. These are applications of established Filippov and generalized-gradient calculus; the general rank rule is explicitly present in Paden--Sastry (1986), not a proposed contribution.

## 1. State, field and exact joint-sign formula

Let z belong to an open subset of R^p. Frozen reference parameters are constants; only when the reference moves are its parameters coordinates of z. Let r:U->R^m be C^1, let M:U->R^{p x m} and b:U->R^p be continuous, and define the locally bounded field

    F(z)=b(z)+M(z) sign(r(z)), sign(0)=0.

For N inputs and d outputs, m=N*d. For the stopped student, the student block of M is J_q^T/m, with zero reference rows. The EMA drift is b=(0,(theta-phi)/tau). Full-input and cached rules have their own matrices derived below; no mean factor is omitted.

Fix z0 and let Z be the residual coordinates that vanish there. The other signs are fixed in a sufficiently small neighborhood. Define S(z0) to be the finite set of full sign arrays s in {-1,0,1}^m such that, in every ball B(z0,delta) inside U, the set {z:sign(r(z))=s} has positive p-dimensional Lebesgue measure. Then the exact closure is

    K[F](z0)=b(z0)+M(z0) conv S(z0).                     (1)

Proof: a nonessential sign pattern has zero measure in some sufficiently small ball. There are finitely many patterns, so one common ball and one null set remove all nonessential patterns. On the remainder, continuity of M and b puts F uniformly arbitrarily close to the finite values b0+M0*s for essential s, proving the upper inclusion. For any essential s, every ball still contains points of that sign after removal of any null set. A sequence of such points approaching z0 has field limit b0+M0*s; it therefore belongs to the convex closure used in every Filippov neighborhood. Convexifying proves the reverse inclusion. This is a finite-pattern application of the standard essential-limit definition, not a new regularization.

Zero signs must be included when a residual vanishes on a set of positive measure; otherwise a purported universal formula using only +/- signs can itself be wrong. If the relevant zero sets are locally null, S has only +/- entries on Z. Formula (1) is an identity for a set-valued map. It does not by itself imply uniqueness, Euler convergence, crossing-time accuracy or correctness of predictions.

Define B(r0) by fixing nonzero-coordinate signs and allowing each coordinate in Z to vary independently in [-1,1]. Since S is a subset of this box,

    K[F](z0) subset b0+M0*B(r0).                         (2)

Equality need not require all sign patterns: M0 can map different patterns to the same velocity. The relevant criterion is equality after applying M0 to the convex hulls, not mere counting of sign arrays.

## 2. A standard full-rank sufficient condition

If D_z r_Z(z0) has full row rank |Z|, then the submersion theorem provides local coordinates with r_Z as the first |Z| coordinates. Each open sign orthant has a nonempty open preimage arbitrarily near z0; it has positive measure. Individual zero surfaces are locally null. Thus every +/- pattern on Z is essential, conv S=B(r0), and equality holds in (2). The case Z empty is ordinary continuity.

This is the full-rank composition rule in Paden--Sastry Theorem 1(4), followed by its continuous-matrix and continuous-drift rules. It is sufficient, not necessary. Rank must be taken in the actual state space: adding frozen parameters as fictitious perturbation directions can falsely make it pass.

## 3. True partial-gradient fields: no rank requirement for smooth residuals

Let D(z)=m^-1 sum_j |r_j(z)|. For C^1 residuals,

    D(z)=max_{s in {-1,1}^m} m^-1*s^T r(z).

It is a finite maximum of C^1 functions. At z0, its Clarke generalized gradient is exactly

    partial D(z0)=m^-1 D_z r(z0)^T B(r0).                (3)

Indeed the active smooth maxima are precisely the sign choices fixed off Z and arbitrary +/- on Z; their gradient convex hull is (3). This uses the classical finite-max rule, not a claim that the independent signs are jointly reachable in state space.

Let P be the constant linear projection that keeps student gradient coordinates and zeroes reference gradient coordinates (P=I for fixed reference). The pointwise full-gradient field agrees almost everywhere with -P grad D. At a differentiable point of this finite maximum, all active gradients agree, so the zero-sign convention yields the same derivative. By the standard identity K[grad D]=partial D, the continuous-matrix rule and addition of continuous b,

    K[b-P grad D](z0)=b0-P partial D(z0)
       = b0-m^-1 P D_z r(z0)^T B(r0).                  (4)

Thus the independent-sign image is exact for this partial-gradient structure even when D_z r_Z is rank deficient. For original full-input adaptation, D_theta r=(A-I)J_q. For cached input, D_theta r_c=-J_q. Each rule is exactly -partial_theta D for its declared loss, with the reference held constant during that student derivative; if the reference moves, its drift is added as b. In contrast, the original stopped field uses J_q^T sign(r)/m instead of -D_theta r^T sign(r)/m, so (4) cannot automatically be applied to it.

Scope: finite batch, C^1 forward/residual maps and continuous reference drift. This does not prove the analogous statement for a ReLU graph at activation boundaries, arbitrary automatic-differentiation selections, Adam states, or gradients through a differently defined loss. No general stability or unique-solution conclusion follows from (4), especially with moving-reference drift.

## 4. One globally idempotent reference with a spurious box trajectory

One input, two outputs, two student parameters theta=(u,v), and the same architecture for the reference:

    f_w(z)=h+w+A*z,
    h=(1,-1)^T, A=[[0,0],[1,1]], phi=0 fixed,
    q_theta=f_theta(0)=h+theta.

Here A^2=A and A*h=0, so f_0(f_0(z))=f_0(z) for every z. It is an affine oblique projection, not an assumed contraction. Initialize theta=phi=0; q0=h is nonzero. Evaluation target h is planted and never used in an update. The original reference input is the current q_theta. Then

    r=f_0(q_theta)-q_theta=(-u,u)^T,
    D=(|r_1|+|r_2|)/2=|u|,
    F_stop=(1/2)*(-sign(u),sign(u))^T.

At any point (0,v), only the sign patterns (-1,1) and (1,-1) occur on sets of positive area. The true closure is the line segment

    K[F_stop](0,v)={(-lambda/2,lambda/2):lambda in [-1,1]}.

The independent-sign construction instead gives the square [-1/2,1/2]^2. This is a strict enlargement with a real solution-level consequence, not merely an unused extra velocity.

Every true inclusion solution satisfies u' in -(1/2)Sign(u) and (u+v)'=0. The energy (u^2/2)'=-(1/2)|u| is nonpositive. From u(0)=0, it forces u(t)=0; the invariant then forces v(t)=0. Thus the true inclusion solution from the origin is uniquely stationary. The specified pointwise discrete algorithm also stays there for every eta because sign(0)=0; this discrete fact is exact algebra, not a convergence experiment.

The enlarged square model admits theta(t)=(0,t/2), since its derivative (0,1/2) belongs to that square along the entire path. It has r=0 throughout but, for evaluator risk R=||q_theta-h||^2/2, R(t)=t^2/8. It violates the true invariant u+v=0. Therefore an apparent accuracy failure or nonuniqueness inferred from this box path would not describe the stopped algorithm. A Lyapunov inequality established for **every** velocity of the outer box would remain sufficient for the true closure, but can be overly conservative; a bad trajectory or a favorable selection in the box cannot be assumed realizable.

## 5. Backward rule and state-space checks on that same model

Original full input-gradient: D=|u| gives F_input=(-sign(u),0). Its true closure at u=0 is [-1,1] x {0}. The independently chosen residual signs map through -(A-I)^T/2 to exactly this segment. The extra sign patterns do not add velocities here. This directly checks (4) in a rank-deficient example.

Cached prediction input q0=h: the target is f0(h)=h, so r_c=(-u,-v), F_cached=-(sign(u),sign(v))/2. Its residual Jacobian is -I and its sign box is exact by both arguments above.

If the same reference is instead an EMA state phi=(p,q), then r=(p-u,q+u). Its Jacobian with respect to (u,v,p,q) has rows (-1,0,1,0) and (1,0,0,1), hence rank 2. All sign combinations are locally reachable through reference-state perturbations, so the stopped box is exact in this full state space. The fixed-reference counterexample is **not** an EMA counterexample. Starting the two copies matched does not make the diagonal an invariant state restriction. Nor does this rank check prove EMA stability, uniqueness or accuracy.

The published regression implementation previously audited retains the full input gradient; the stopped rule is our controlled comparison. This result therefore does not expose a bug in that implementation. Its ReLU/Adam details also lie beyond the smooth-map statement (4).

## 6. Verification, prior art and paper decision

`verify_closure.py` uses exact fractions, without integration or neural data. It verifies A^2=A, A*h=0, the residuals/three fields at 25 rational substitutions, the full loss-gradient by exact differences away from the kink, chamber images, the separating invariant, and four substitutions in the spurious analytic path. It checks rank 1 for the frozen residual map and rank 2 with EMA state, and realizes all EMA/cached sign choices explicitly. Runtime was below the CPU clock's .015625-second resolution (reported 0.0); no adaptation, training, parameter sweep or labels were used. The proofs establish all-state claims; samples check algebra only.

The protocol is byte-identical to cycle 11's NEXT_ANALYSIS_PROTOCOL.md, SHA256 991ac18a54beb2d9c2ed97d347e6db55bd813d892ecf56aab2c8261c6fd58f01. Exact source hashes and locations are in SOURCE_EVIDENCE.md. The original Paden--Sastry calculus already contains the product outer inclusion, rank condition, matrix rule and gradient-closure identity; Cortes supplies the finite-max rule. These central ingredients are established literature. The constructed idempotent-reference application is checked here, but its novelty has not been audited against nonsmooth learning/automatic-differentiation work.

**Paper decision:** no-go on presenting an independent-sign warning or the rank rule as a new general theorem. Conditional thesis value: the backward rule determines which standard nonsmooth model is faithful, and wrong closures can invent current-task failures. A first paper still requires a distinct prediction or finite-step consequence in an implementation-relevant setting. The remaining gap is not filled by adding another toy example. Next perform a bounded claim/novelty and implementation-scope decision against the closest nonsmooth-learning analysis, then select one concrete paper target or demote this branch to background. Cycle 10 remains numerically unresolved; none of these proofs validate or retune its neural endpoints.
