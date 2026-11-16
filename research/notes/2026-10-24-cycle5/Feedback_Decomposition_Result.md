# Local feedback decomposition: a mismatch is not necessary for these modes

Completed 24 October 2026. Protocol prepared 23 October before the run. This is a checked project calculation on existing synthetic neural checkpoints, not a novelty claim or a reproduction of an official IT3 benchmark.

## Decision

Neither residual-weighted derivative terms nor unequal student/reference parameter Jacobians are necessary for the positive, prediction-observable initial modes in this diagnostic. Removing residual-weighted terms preserves the leading observable rates within 7.34% in every primary setting, passing the predeclared 10% gate. Additionally setting the reference parameter Jacobian L equal to the student Jacobian J preserves positive observable modes in all primary settings, but changes rates by up to 34.12%, failing the quantitative gate.

Reject the proposed necessity of parameter-Jacobian mismatch for this local signature. Do not infer that mismatch is irrelevant, that shared-J feedback explains the complete nonlinear trajectory, or that changing L alone defines an implementable neural method. A causal intervention on a learned network was not performed. The checkpoints remain nonstationary, so these are frozen-time derivatives, not equilibrium stability results.

## Precise model and derivation

Use cycle 3's 114-parameter tanh model and squared consistency surrogate. Restrict both parameter copies to the 82 non-fixed coordinates; the other 32 multiply the student's zero auxiliary input and remain fixed under matched initialization and EMA. This restriction is not a controllability claim.

At the matched initial checkpoint, q=f_theta(x,0), r=f_phi(x,q)-q, J=partial_theta f_theta(x,0), L=partial_phi f_phi(x,q) with q held fixed, A=partial_q f_phi(x,q), B=I-A. Identical weights do not imply L=J: the two calls have different auxiliary inputs. The residual differential is exactly

    dr = -B J dtheta + L dphi.

The student velocity is F=E[C r], with C=J^T B^T for the input-gradient rule and C=J^T for stopped output. Thus

    S = partial_theta F = S0 + Rs,
    T = partial_phi F   = T0 + Rt,
    Rs = E[(partial_theta C) r], Rt = E[(partial_phi C) r].

The product-rule blocks are

    input: S0=-E[J^T B^T B J], T0=E[J^T B^T L];
    stop:  S0=-E[J^T B J],     T0=E[J^T L].

For stopped output, Rt=0 exactly. The input-gradient S0 is negative semidefinite, but this fact alone does not settle the coupled dynamics. For each pair of blocks, evaluate

    M = [[S,T],[I/tau,-I/tau]].

The second diagnostic retains S0 and replaces L by J in T0. Neither deletion is asserted to be a physically realizable network change or to put the actual checkpoint at an equilibrium. In particular, "zero-residual blocks" names the product-rule component, not an achieved zero residual.

## Frozen protocol and results

All five cycle-3 checkpoints were reused without training. Seeds 0,2,3,4 remain primary; failed-fit seed 1 is retained as a sensitivity case. The same 64 inputs define the empirical field; the other 192 define output observability. No labels enter these calculations. Both rules, tau=1 and 20, and three matrix variants give 60 matrices of dimension 164. There is no additional trajectory sweep, distribution change, method search, or threshold tuning.

A qualifying mode has real part greater than 1e-6 and output observability ||J_eval u||/(||u|| ||J_eval||_2) at least 1e-3. All eigenvectors are checked. Across the 48 primary matrices, leading-mode observability lies between .356 and .930, well above the cutoff. Each variant preserves a positive observable mode in all four primary seeds for each rule/tau pair; the predeclared sign gate required at least three.

Leading observable continuous-time rates (rounded):

| Seed | Rule | tau | Full | Remove residual terms | Also set L=J |
|---|---|---:|---:|---:|---:|
| 0 | Input | 1 | .140613 | .138609 | .167082 |
| 0 | Input | 20 | .010801 | .010818 | .014339 |
| 0 | Stop | 1 | .148655 | .148579 | .186033 |
| 0 | Stop | 20 | .010992 | .011083 | .014743 |
| 2 | Input | 1 | .342612 | .341720 | .316908 |
| 2 | Input | 20 | .057287 | .057278 | .050333 |
| 2 | Stop | 1 | .630403 | .630612 | .565619 |
| 2 | Stop | 20 | .067007 | .067080 | .060097 |
| 3 | Input | 1 | .218159 | .214991 | .255205 |
| 3 | Input | 20 | .025129 | .024677 | .027452 |
| 3 | Stop | 1 | .294270 | .294337 | .343769 |
| 3 | Stop | 20 | .027897 | .027860 | .031893 |
| 4 | Input | 1 | .092546 | .091982 | .096679 |
| 4 | Input | 20 | .007726 | .007196 | .008023 |
| 4 | Stop | 1 | .099633 | .099511 | .104836 |
| 4 | Stop | 20 | .007991 | .007405 | .008253 |

The worst relative difference from removing residual terms is 7.3343%, seed 4/stopped/tau=20. The worst after also setting L=J is 34.1213%, seed 0/stopped/tau=20. The 10% gate compares the leading observable rate, not individual matched eigenvectors, a full spectrum, or trajectory response. Seed 1 also has positive observable modes throughout; its full precision results are retained in the JSON and archives, but it does not enter primary gates.

In primary seeds, ||L-J||_F/||J||_F is .3124-.5711. A substantial mismatch therefore exists, despite not being needed for mode existence under this matrix diagnostic. Student residual-term Frobenius ratios are .002843-.005518 relative to the full student block. Input-gradient teacher residual ratios are .001637-.002499; stopped-output teacher remainders are zero. Small matrix norms alone would not have certified small spectral changes; the rate comparison was computed separately.

The specified student-first discrete derivative at eta=.02 and alpha=tau/(tau+.02) also has spectral radius above one in all primary cases (range 1.000144-1.012700). This is a local frozen-time derivative observation, not an assertion of sustained discrete growth. No future-risk conclusion follows from these eigenvalues. Cycle 4 remains the separate evidence for finite-time prediction amplification and risk deterioration.

## Verification and reproducibility

- Independent complex-step reference-parameter Jacobians, holding q fixed: maximum absolute error 3.34e-16.
- Differentiate C(theta,phi)r0 while freezing the baseline residual r0; add these independently computed remainders to S0,T0 and compare with cycle 3's saved full central differences: maximum absolute error 1.575e-10 (protocol limit 1e-7).
- Stopped-output teacher remainder is exactly zero in the calculation (limit 1e-12).
- Maximum eigenpair backward residual: 1.541e-15 (limit 1e-10).
- An independent explicit sample-loop assembly matches the vectorized block construction within 1.222e-15.
- Model, data, and all five checkpoint hashes match cycle 4's recorded inputs. Protocol/script hashes match the results. Post-run verification did not change the protocol, script, eligibility, or gates.

Main calculation: 1.94 CPU wall seconds; NumPy 2.3.5 and Python 3.12.14. No training, GPU, package installation, paid compute, or cluster work. Reproduce from the proposal notes directory with `python decompose_feedback.py`, then `python verify_and_summarize.py`; these use the preserved sibling cycle-3 and cycle-4 artifacts. Scripts, protocol, results, five compressed matrix archives, and verification summary accompany this note.

Primary-source provenance remains the earlier audited IT3 squared-surrogate interpretation: [arXiv v2](https://arxiv.org/html/2410.04201v2), reference choices in Eqs. (6)-(7), and [official code revision 100524001d890efc08911f3cba8f45f51f4aa7d1](https://github.com/nikitadurasov/ittt/tree/100524001d890efc08911f3cba8f45f51f4aa7d1). This cycle adds no new literature result and did not execute that implementation. Its update ordering and squared loss remain project choices.

## Next decision gate

Audit the exact assumptions and conclusions of PeTTA's collapse model and RMT's consistency-gradient analysis, with the fixed-teacher comparison as an empirical boundary. Make a compact claim map against this specified dual-input squared-loss graph: teacher target, differentiated branches, parameter coupling, stationarity, and measured outcome. The question is whether any demonstrated effect or testable distinction remains beyond existing teacher-feedback results. Record exact theorem/equation/version locations. If the overlap covers the observed signature, demote generic instability and mismatch-based explanations rather than add a stabilizer or a larger benchmark. Only advance a distinct remaining claim with a qualitative prediction and falsification criterion fixed before another experiment. The constructed rotation result, failed learned-rotation gate, failed uniform rule ordering, and finite-time results remain useful thesis groundwork; a first-paper package is not yet justified.
