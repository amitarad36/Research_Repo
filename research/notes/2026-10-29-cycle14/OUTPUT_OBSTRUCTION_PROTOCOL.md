# Cycle 14: analytic certificate and output-budget obstruction

29 October 2026. Saved before calculating this cycle's certificate outcomes.
Execute the unchanged cycle-13 gate. First prove the proposed recurrence and
audit whether saved coefficient arrays are exact derivatives. Before spending
the 60 CPU second budget on upper bounds for the state map, test a necessary
condition for the prescribed separate-arm scalar-Hessian certificate.

For q_o(x)=sum_j V_jo tanh(w_j.x+b_j)+c_o, the mixed derivative in unit
directions V_jo and b_j is sech^2(w_j.x+b_j) in output o. For the RMS map
Q=vec(q)/sqrt(384), any valid bilinear Hessian norm bound H on a ball
containing the initial point obeys H^2 >= (25/64)^2 m/384 whenever m
evaluator inputs have |w_j.x+b_j|<=1. Use the largest such count among
the 16 hidden units, computed with exact rational arithmetic on stored
binary64 inputs/weights. The elementary cosh(1)<8/5 proves this bound.

Since state enclosures E and Jacobian-norm contributions are nonnegative,
the prescribed B obeys B >= H*(||u_stop||^2+||u_cached||^2)/2 at T=2.
Square and compare with (.20)^2 ||Delta_Q_hat||^2 using exact rational
arithmetic on the saved cycle-8 endpoints/forecast arrays. If already
larger, no tighter valid H or state remainder can make this prescribed
certificate pass for that saved forecast. A negative result concerns
this formula/norm, not every paired or direction-dependent bound.

All five seeds, both saved step sizes; seeds 0,2,3,4 primary, 1 sensitivity.
Verify source/checkpoint/trajectory/forecast manifests before use. No labels,
new neural trajectories, derivative finite differences, hyperparameter
changes or new source reading. Hash the cycle-13 protocol and this protocol.
If the obstruction does not reject the primary gate, proceed to valid
upper bounds within the remaining 60 CPU seconds, or report the specific
remaining obligation. Do not introduce a different bound after outcomes.

Also audit the norm convention: cycle 8 uses RMS of per-example output
Euclidean norm, whereas cycle 13 uses RMS per scalar output (sqrt(2) smaller).
Apply the same conversion to separation and its scale floor; relative
criteria do not change. Treat saved numerical forecast arrays as a fixed
forecast, not exact real-arithmetic propagation. Derivative approximation
errors would require extra nonnegative remainder terms; they cannot remove
an obstruction already present with those terms set to zero.
