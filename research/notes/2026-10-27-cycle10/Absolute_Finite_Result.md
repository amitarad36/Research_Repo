# Absolute-loss intervention: risk inference unresolved, separation onset checked

27 October 2026, cycle 10. New checked numerical result in the existing synthetic model. This is neither an official IT3 reproduction nor a novelty claim.

## Decision

The predeclared primary numerical gate fails: **15/16** primary stopped/cached fixed/EMA conditions change terminal NMSE by more than 5% when eta is halved. The correct classification is **numerically unresolved**, not a positive replication or a negative effect finding. All runs and thresholds are retained. No smaller step, smoothing, alternate horizon, averaging window, seed selection or rescaling was tried.

There is a narrower verified result. In all 20 stopped/cached pairs (five seeds, two reference schedules, two step sizes), the states remain exactly equal until different residual signs produce a nonzero weighted field difference. The first sign and velocity disagreements occur at the same recorded step; states separate on the following update. This is an onset description and exact discrete identity, not an explanation of terminal predictive risk.

## Unchanged protocol and provenance

`PROTOCOL.md` is byte-identical to cycle 9's NEXT_EXPERIMENT_PROTOCOL.md, SHA256 `081806a9e30ac8ec09466cc4f9a3448a2258ed696dfd51cc0cfb92061da110d1`. All five cycle-3 checkpoints, source code and data match their saved hashes. Primary seeds remain 0,2,3,4; seed 1 remains sensitivity-only. Initialization gradients match cycle 9 and initial NMSE matches cycle 7 exactly. The cycle-9 derivative audit is the mathematical/code source; the selected official tabular notebook is pinned there at commit `100524001d890efc08911f3cba8f45f51f4aa7d1`, cell 13 lines 34-38. No new literature or remote implementation was retrieved or executed this cycle.

Mean absolute residual over N*d entries, N=64,d=2, sign(0)=0, with the required 1/d field factor. Original stopped-output/full input-gradient/cached-input, each fixed reference or tau=1 EMA. Student-first GD/EMA, eta=.02/.01, T=20. All 60 trajectories and 90,000 updates completed in **48.046875 CPU seconds / 51.157833 wall seconds**. No training, installations, paid compute or GPU. NumPy/Python versions are recorded in the complete JSON.

The same 64 inputs are reused; 192 independent inputs and labels are evaluator-only. Each arm allocates the same initial 64x2 float64 prediction cache (1024 bytes); only cached-input updates use it. Two parameter copies; 32 inactive coordinates remain exactly fixed. Cache retention is per-input information, not a fresh-stream method. The original ReLU, k=50, one-output, Adam and episodic-reset implementation is not reproduced. Absolute speeds across the earlier differently scaled squared loss are not a controlled loss-only comparison.

## Primary endpoint record

NMSE at T=20, stopped/EMA (Re) versus cached/EMA (Rc):

| Seed | Initial R0 | Re eta=.02 | Rc eta=.02 | Re eta=.01 | Rc eta=.01 | Effect inequalities at both steps |
|---|---:|---:|---:|---:|---:|---|
| 0 | .012340 | 22.273436 | .736963 | 50.820505 | 3.544380 | Yes |
| 2 | .036731 | 161.210322 | 3.147092 | 160.904360 | 7.132950 | Yes |
| 3 | .017224 | 11.141349 | 1.693386 | 9.033429 | 4.148523 | No |
| 4 | .007521 | .830373 | .161839 | 6.068585 | 2.358623 | No |

The inequalities are Re>=2R0, Re-R0>=.01, max(Rc-R0,0)<=.25(Re-R0). All four meet them at eta=.02, but only two at eta=.01. This count is descriptive because the numerical gate already failed. It must not be relabeled a numerically established non-replication. Every cached/EMA endpoint is worse than its initial error, including sensitivity seed 1 at both steps.

Numerical discrepancies use abs(main-fine)/max(abs(fine),.01). Eight of eight primary fixed-reference stopped/cached checks fail; seven of eight EMA checks fail (only seed 2 original stopped/EMA passes). Maximum primary discrepancy is **152.2826%**; cached/EMA discrepancies range **55.8795%-93.1384%**. Thus the blocker is not confined to a secondary arm or to a near-zero denominator.

The full input-gradient arms are a separate secondary comparison, with different initial velocities. Six of eight primary-seed checks fail; the largest discrepancy is **1605.9326%**, seed 3 EMA (162.125606 versus 9.503635). Its broad risk comparison is also unresolved. Passing individual conditions do not rescue the all-condition gates. Sensitivity seed 1 remains excluded from primary counts; its complete outcomes are preserved.

## Sign events and exact first separation

The saved algorithm evaluates fields at state n, then updates to n+1. Let n* be the first sign disagreement at a common state, J_i the student output Jacobian there, and delta-sigma the original-minus-cached signs. At that state,

    F_stop - F_cached = (1/(N*d)) sum_i J_i^T delta-sigma_i
    theta_stop[n*+1] - theta_cached[n*+1] = eta*(F_stop-F_cached).

This equality is exact for the declared update. A sign disagreement could cancel under the weighted sum; it does not do so at the observed first events. Equal signs only ensure equal fields at equal student states; after separation, the Jacobians can differ even if signs later coincide.

| Primary seed | Fixed: sign time .02 / .01 | EMA: sign time .02 / .01 |
|---|---|---|
| 0 | .04 / .04 | .04 / .05 |
| 2 | .02 / .01 | .02 / .01 |
| 3 | .02 / .01 | .02 / .01 |
| 4 | .02 / .02 | .02 / .01 |

At eta=.02 this is after one or two updates; the finer runs can take up to five. In every case first state separation is one step later. These are discrete grid events, not continuous-time crossing estimates or certified event-time convergence. Shared states/signs produce exactly equal fields and next states numerically. No residual is exactly zero at any recorded per-step evaluation; this does not exclude crossings between steps. Frequent sign changes are recorded, but do not by themselves prove a sliding limit, numerical chaos, or the cause of terminal step sensitivity.

## Independent verification and artifacts

`verify_absolute_finite.py` reads saved results; it does not rerun trajectories. It independently recomputes all **3,060** metric records and saved targets exactly, checks **20** archive hashes and all **60** event snapshots. Explicit per-sample/output parameter-gradient assembly agrees with event fields within **3.886e-16**; the first parameter-separation identity agrees within **1.601e-16**. All fixed-weight cached targets remain exactly constant and all inactive coordinates remain fixed. The script rederives failed step checks and the effect counts; verification took .84375 CPU seconds.

`absolute_finite_results.json` SHA256: `fa6a5b83e6ff38d5d5e2066a58712b90cc582d9449eec1c179970f47048a56a9`. It contains complete histories, provenance and gates. Twenty NPZ archives store both step sizes, both parameter copies, reference targets and caches, per-step sign statistics and onset snapshots. `verification.json` records the independent checks. A permission-review timeout occurred while writing the verifier draft; a retry succeeded before verification, with no trajectory rerun or result modification.

## Consequence for thesis/paper scope

Keep the supported squared-loss finite-time result and its early affine forecast explicitly scoped to that smooth surrogate. The absolute-loss audit establishes local equality and checked sign-triggered onset, but not a robust accuracy effect. No retention, recovery, drift, safe-adaptation or novel method claim follows.

The candidate thesis question remains how loss derivatives and the backward rule determine feedback through the reference prediction input. A first-paper package still needs an informative predictive result across sign changes and relevance to one actual implementation, beyond elementary sign calculus or generic EMA failure. The present evidence does not meet that threshold.

Next single question: which nonsmooth mathematical evolution is well posed at an attracting residual-zero surface, and which claims about the exact discrete algorithm survive that distinction? The bounded analytic protocol in `NEXT_ANALYSIS_PROTOCOL.md` uses a transparent affine diagnostic, not a retuned neural experiment. It should either supply a qualified baseline for later Lyapunov reasoning or expose why the desired transfer cannot be claimed. Standard nonsmooth theory is background, not proposed novelty.
