# Fixed analytic upper-bound recipe (before upper-bound outcomes)

The first output-Hessian lower-bound test did not reject any seed. It is
inconclusive, not a successful certificate. Proceed once with the following
standard global derivative envelope; do not tune it after outcomes.

Reconstruct only the saved affine recurrence to T=2 at both steps. Fix R
as twice its maximum unscaled joint-state norm, per the parent protocol.
For constants, use an outward rational radius Rbar on the grid 1/1024;
the certified ball remains R, not Rbar. Treat the floating-point forecast
as the specified numerical forecast; no new nonlinear trajectory.

For each input let l=sqrt(1+||x||^2). Let V bound the output-weight
Frobenius norm and b bound output-bias norm on the Rbar ball. Let K bound
the fixed prediction-input weight Frobenius norm. Use rational upper square
roots on grid 1/1024 for initial norms, l, and a=sqrt(l^2+K^2).
For a tanh network with 16 units define

    q0=4V+b, q1=Vl+5, q2=Vl^2+2l, q3=2Vl^3+3l^2.

These use |tanh|<=1, |tanh'|<=1, |tanh''|<=1, |tanh'''|<=2;
qk bounds kth parameter derivative norm. The 5 bounds the output-weight
and bias derivative terms by 4+1. These are conservative, not optimized.
For the stopped teacher on joint (phi,z), use t1=Va+5, t2=Va^2+2a,
g1=max(1,q1), r0=2q0, r1=t1*g1+q1,
r2=t2*g1^2+(t1+1)*q2.
For cached input use r1=2q1, r2=2q2 and the same r0.
Then F2=q3*r0+2*q2*r1+q1*r2 bounds the second derivative of
F=mean Jq^T r. Use the maximum input l to bound the mean.
For the exact student-first EMA map, beta=eta/(1+eta), take
L=eta*(1+beta)*F2 >= eta*sqrt(1+beta^2)*F2.

For evaluation Q=vec(q)/sqrt(384), H=q2_eval is a valid conservative
Hessian bound (in fact q2_eval/sqrt(2) suffices). To avoid an avoidable
sqrt(2) penalty use H^2=q2_eval^2/2 in exact comparisons.

Before full propagation, test two necessary budget conditions for THIS
upper-bound construction with all state/coefficient errors set to zero:
(i) B>=H/2 sum_a ||u_N^a||^2;
(ii) E_N^a >= L_a/2 ||u_(N-1)^a||^2, hence
B>= (1/sqrt(2)) sum_a L_a/2 ||u_(N-1)^a||^2,
because output-bias directions give ||DQ(z0)||>=1/sqrt(2).
Both inequalities concern the computed bound, not actual nonlinear error.
If either exceeds 20% predicted separation, this recipe cannot pass even
with exact initial coefficients and no rounding error. That suffices to
reject this construction; it does NOT show optimal constants or paired
direction-sensitive enclosures must fail. Save all ten conditions.

No supremum is estimated from derivative samples. Additional terms needed
for finite-difference coefficients and numerical recurrence defects will
be derived in the theorem note; they cannot repair a failure already
present with those nonnegative terms dropped. Total computation remains
below 60 CPU seconds including the first test.
