# A valid recurrence, but a failed concrete certificate construction

29 October 2026, cycle 14. Status: checked derivation and exact-arithmetic
negative feasibility test for one explicit analytic enclosure recipe.
No new nonlinear trajectories, training, labels, parameter tuning or literature
claims. The cycle-13 fixed T=2, both-step, three-of-four-seed gate is not met.
This does not refute the previously observed accurate affine forecast.

## Scoped theorem and proof

Let Phi be C1 on a convex ball B(z0,R), with D Phi Lipschitz constant L.
Let Q have Jacobian norm J at z0 and Lipschitz derivative constant H on
the same ball. All norms are Euclidean, with Q scaled to RMS per output
entry. Write c=Phi(z0)-z0, A=D Phi(z0), u0=0, u_next=c+A*u.
If E0=0 and

    E_next = ||A|| E + (L/2)(||u||+E)^2,

then ||z_n-z0-u_n||<=E_n by induction as long as ||u_n||+E_n<=R
at every step used in the Taylor expansion. Indeed the integral remainder
is bounded by L||z_n-z0||^2/2, and subtracting the affine recurrence
leaves A times the preceding error plus that remainder. Check the final
point also before using the output expansion. This gives

    ||Q(z_n)-Q(z0)-D Q(z0)u_n||
       <= J E_n + (H/2)(||u_n||+E_n)^2.

Sum the two arm bounds for their output difference. If B<=.20||Delta_hat||
and ||Delta_hat||>B, reverse triangle inequality gives relative true-output
error <= B/(||Delta_hat||-B)<=.25. No labels or risk guarantee appear.
This is classical Taylor/perturbation reasoning reproduced here, not a
novel theorem or a verified network-domain containment claim.

### Necessary repair for the saved numerical forecast

Cycle-7 initial_checks computes student/teacher blocks with central
differences at h=1e-5; cycle-8 affine_map uses those blocks. They are
numerically checked approximations, not exact D Phi(z0). Given an arbitrary
saved numerical recurrence u_next=c_hat+A_hat*u+rho_n, valid bounds
dc>=||c-c_hat|| and dA>=||A-A_hat|| require

    E_next = ||A_hat|| E + dc + dA*(||u||+E)
             + (L/2)(||u||+E)^2 + ||rho_n||.

The output formula similarly needs a dJ*||u|| term for an approximate
output Jacobian, plus any output-evaluation/arithmetic defect. The proof
comes from adding/subtracting A_hat and c_hat. Sampling agreement alone
does not give dc,dA,dJ upper bounds. A positive computer certificate would
need those enclosures and rounded arithmetic. None is claimed here.
The negative tests below drop all such nonnegative terms; adding them
cannot rescue this particular construction.

## Evidence and normalization audit

Model: neural_geometry_gate.py, unpack/forward/parameter_jacobian, saved
cycle-3 hash a28b93e5ca4d1acac923279c2cd87c5d3eb097695f0917c87be46d65b6501f88.
q(x)=tanh(x W_x+b1) V+b2, 16 hidden units, two outputs. Active coordinates
are 0:32 and 64:114, 82 per copy. The 32 prediction-input weights stay fixed.
The teacher's input derivative is retained when differentiating the update
map as a state map, even though the stopped loss's backward rule omits it.
There is no assumption that a stopped field is a gradient of its forward loss.

Verified cycle-7 and cycle-8 script/protocol/data/checkpoint hashes, all 35
cycle-7 archives and all five cycle-8 forecast archives against their recorded
manifests. No evaluator labels were extracted. See exact hashes in the JSON.
Cycle-8 rms is sqrt(mean(sum_outputs q^2)); cycle-13 Q uses
sqrt(mean_all_entries q^2)), smaller by sqrt(2). Convert both signal and
old floor by that factor. Relative gates are invariant. The separate
nontrivial-floor gate is not reached because the error-budget gate fails.

## Two tests, with their recipes fixed before their respective outcomes

1. OUTPUT_OBSTRUCTION_PROTOCOL.md tests an intrinsic necessary lower bound.
For unit parameter directions V_jo and b1_j, the mixed second derivative
is sech^2(x W_xj+b1_j) in output o. Exact rational preactivation checks
find at least one unit with |preactivation|<=1 on all 192 evaluator inputs
in every seed. The cosh series obeys
cosh(1)<=3/2+(1/24)/(1-1/30)<8/5, so sech^2>=25/64. Thus any valid
H obeys H^2>=(25/64)^2/2. The prescribed output budget has the necessary
floor H*(||u_stop||^2+||u_cached||^2)/2. Its ratio to separation is
1.14%-4.52% on primary seeds, below 20%. This test is INCONCLUSIVE: it
does not show that every scalar-Hessian certificate is too loose.

2. ENCLOSURE_RECIPE.md then fixes one standard, conservative analytic
upper-bound construction using tanh derivative bounds, input and weight
norms. It derives valid q1,q2,q3 and F2, where
F=mean Jq^T r and ||D^2F||<=q3*r0+2*q2*r1+q1*r2.
For the student-first reference update, beta=eta/(1+eta),
||D^2 Phi||<=eta*sqrt(1+beta^2)*F2<=eta*(1+beta)*F2.
The output bound is H=q2_eval/sqrt(2). The recipe's detailed composition
rules, including the frozen prediction-input weights, are preserved.

The radius is twice the maximum displacement of a replay of the saved
affine recurrence through T=2, all steps, both arms. Its nominal values
range .01817-.03691 across all seeds/steps. Derivative constants use a
rational OUTER radius on the 1/1024 grid; the allowed containment radius
is not enlarged. Square roots for constants are rounded outward by exact
integer arithmetic. Bound-budget comparisons are exact fractions of stored
binary64 data, not arbitrary floating-point epsilon tests. Forecast replay
matches the saved endpoints within 1e-12 (exact differences in JSON).

| Primary seed | Output term / separation, eta=.02 | eta=.01 | State-term lower budget, .02 / .01 |
|---|---:|---:|---:|
| 0 | .80640 | .80632 | 13.72782 / 6.86205 |
| 2 | .27638 | .27565 | 5.15778 / 2.57522 |
| 3 | 1.20241 | 1.20300 | 22.95885 / 11.48096 |
| 4 | .52901 | .52848 | 10.57610 / 5.28139 |

Required ratio <=.20. All eight primary conditions fail already with
E=0 in the output term. Independently, E_N>=L||u_(N-1)||^2/2 and
||DQ(z0)||>=1/sqrt(2), from the output-bias direction, show the chosen
state remainder budget also fails. These numbers are lower bounds on
THIS CONSTRUCTED UPPER BOUND, not lower bounds on the true prediction
error. Sensitivity seed 1 also fails both conditions, but is not counted.
No certified state-domain containment or positive network enclosure was
produced. The necessary-budget failure makes full propagation and a
comparison to saved nonlinear errors unnecessary; neither was performed.

Combined numerical CPU time: 1.59375 seconds, below the fixed 60-second cap.
Scripts and complete exact signed margins are saved. No outcome was omitted.

## Decision and limits

Reject the tested norm-envelope construction and demote the certificate
route in its present form. Do not change radius/norm/horizon/seeds/threshold,
fit constants, or run a tightening sweep to claim this gate passed. The
inconclusive intrinsic test prevents a stronger impossibility statement:
optimal constants, paired cancellations or directional enclosures remain
unresolved. They are not automatically adopted as a replacement project.
The observed <=1.51% early forecast error remains a valid earlier empirical
result. Neither that result nor this failed certificate establishes a new
paper or a useful adapter. Cycle-10 absolute-loss risk remains unresolved.

The next bounded task returns to implementation relevance: audit whether
forward prediction-input evolution can be isolated while preserving the
official full input-gradient backward path. See NEXT_GRAPH_PROTOCOL.md.
No new performance claim or new experiments are authorized by that protocol.
