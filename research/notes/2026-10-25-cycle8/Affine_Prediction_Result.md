# Initial derivatives predict the early input-path effect, with a finite validity range

25 October 2026, afternoon cycle 8. **New checked calculation on existing synthetic trajectories.** The saved T=2 gate passes in all four primary seeds at both step sizes. This is an analysis of existing data after terminal results were known, not an independent confirmatory experiment. No new neural adaptation or training was run, and no labels were used in the calculation.

## Question and fixed prediction

Cycle 7 showed that holding the reference's prediction input fixed substantially reduces excess terminal error. Does an untuned approximation using only initial derivatives predict the early difference between the original stopped-output and cached-input trajectories?

Use the 82 non-fixed coordinates per parameter copy. For each arm, let u=theta-theta0 and v=phi-phi0 in these coordinates. The full initial derivatives S=d_theta F, T=d_phi F include residual-weighted terms; F0 is the common nonzero initial student velocity. The forecast is the affine student-first update

`u_next = u + eta (F0 + S u + T v)`

`v_next = alpha v + (1-alpha)u_next`, with `alpha=1/(1+eta)`.

It starts at u=v=0. No matrices are fitted to the nonlinear trajectories or updated along them. The initial evaluator Jacobian J0 predicts `Delta_q_hat = J0 (u_original-u_cached)`, compared with `Delta_q = q(theta_original)-q(theta_cached)` on the same 192 evaluation inputs. Their labels are not extracted. This forecasts a difference between two interventions, not a risk value or response to arbitrary perturbations.

## Derived map and ordering checks

Writing w=(u,v), the affine map is `w_next=M w+b`, with

```
M = [ I+eta S                 eta T                 ]
    [ (1-alpha)(I+eta S)      alpha I+(1-alpha)eta T ]
b = [ eta F0 ; (1-alpha)eta F0 ].
```

This is obtained by substituting the updated student into the reference step; replacing it by the old student changes the map. Both arms have the same first step. Since their initial teacher-derivative blocks agree, their second **affine** student step differs by exactly `eta^2 (S_original-S_cached) F0`. The equality is a check of the affine recurrence, not an exact identity for the nonlinear models at finite eta. In cycle 7, the full student-block difference was independently checked against `mean J^T A J`.

The forcing F0 cannot be discarded: these initial checkpoints are not equilibria. A homogeneous recurrence initialized at zero would predict no motion. These algebraic tools are standard linearization and chain-rule calculations; successful implementation is not a novelty claim.

## Unchanged analysis protocol

- Protocol was saved in cycle 7 before this approximation was calculated, after the original experiment's endpoint results were known. SHA256 `283c71ca44dcc84be95fce119044b5d4c51fd9e0b1be50693bb775a1406bc8ca`.
- All five existing checkpoints, primary 0,2,3,4; seed 1 remains sensitivity-only. Original stopped-output/EMA and cached-input/EMA at tau=1, eta=.02 and .01. No extra seed, perturbation, nonlinear trajectory, retraining, label use or parameter choice.
- All predictions through T=20 are retained at the 51 original record times. Primary time fixed at T=2. Error is RMS(Delta_q_hat-Delta_q)/max(RMS(Delta_q), 1e-8 max(RMS(q0),1)). Tiny separations do not count as passes.
- Required error <=25% in at least three of four primary seeds at BOTH step sizes, plus all numerical checks. No horizon or threshold changed. Long-horizon curves are descriptive and cannot replace the primary gate.

## Results

All four primary seeds pass at both steps, with primary prediction error **0.4517%-1.5078%**. The actual primary RMS separation at T=2 is .001308-.005074, well above the approximately 1e-8 scale floor. Direction cosines exceed .99995 in every primary comparison; this is descriptive, not a separate success gate.

| Seed | Actual separation RMS, eta=.02 | Predicted RMS, eta=.02 | T=2 error, .02 | T=2 error, .01 | First saved time exceeding 25% error, both steps |
|---|---:|---:|---:|---:|---|
| 0 | .00140336 | .00140144 | .5850% | .5836% | none through T=20 |
| 2 | .00507432 | .00505486 | .4531% | .4517% | 10.4 |
| 3 | .00131331 | .00129744 | 1.5078% | 1.5039% | none through T=20 |
| 4 | .00154679 | .00154815 | .6380% | .6356% | none through T=20 |
| 1, sensitivity only | .00696004 | .00696841 | .2714% | .2662% | 8.0 |

The first failure times are **sampled diagnostics at .4 spacing**, not continuous-time exit estimates or certified horizons. At T=20 the primary main-step errors are 8.53%, 24,391.80%, 15.58%, and 9.57% for seeds 0,2,3,4. Seed 2 is therefore a strong counterexample to extrapolating this accurate early forecast to all later times. The failed-fit sensitivity seed reaches approximately 2.50 million percent error at T=20. Both are retained; neither alters the fixed early gate. No risk-ranking or stability conclusion follows from the output-separation forecast.

## Numerical validation and reproducibility

- Checked model/data hashes, every source checkpoint, the cycle-7 result and script/protocol hashes, and all 35 cycle-7 numerical archives before analysis.
- Full initial reference-block equality, student-block identity and initial velocity equality pass. All inactive student velocity coordinates are exactly zero.
- Initial 192-input evaluator Jacobian agrees with central finite differences within 4.536e-11 (114 coordinates, step 1e-5).
- Direct student-first and independently assembled block-matrix propagation agree within 7.855e-14 relative state error across all steps, far below 1e-9. First affine steps agree exactly; maximum second-step identity error 2.849e-20.
- `verify_saved.py` independently recomputes all 510 output-comparison records from saved nonlinear states and per-input matrix products, checks block coefficients, and reproduces all gate decisions. Maximum actual-array difference is zero; predicted-array difference is 5.821e-11 at the very large late affine values, and relative-metric difference is <=7.276e-12. Block coefficient reconstruction error <=5.552e-15.
- Main calculation: 1.078125 CPU seconds, 1.82008 wall seconds, within the 30-second target and 2-minute cap. No nonlinear updates, new training, installs, GPU, paid or cluster compute.
- An initial implementation attempt supplied a 64-row zero auxiliary input to the 192-row evaluator Jacobian and stopped before any affine propagation or outcome calculation. Corrected only that array shape to zeros_like(xe). Preserved the failed source and JSON report as `initial_attempt.py` and `initial_attempt_failure.json`; no protocol, seed, threshold or model changed.
- Results SHA256: `6e1b291b6c1550a1378d02382407cfaeaad51121dd7ff931e5a801959d50f563`. Full source/protocol/environment/provenance and output archive hashes are in `affine_prediction_results.json`. `verification_summary.json` records the independent audit. Five seed archives contain the matrices, forcing, initial evaluator Jacobian, both affine propagations, and all actual/predicted output differences. The completed script refuses to overwrite results; run a reproduction in a fresh folder with PROTOCOL.md and the prior evidence accessible.

## Research decision

The narrow thesis question now has both a realizable intervention and a successful early forecast in the chosen smooth model. This supports studying input-mediated feedback separately from weight averaging. It does **not** establish a generally interpretable stability condition, current-risk improvement, useful adaptation under distribution drift, retention, recovery, or novelty. The predictor uses complete derivative matrices; a mechanistic theorem still needs explicit approximation assumptions and an error bound. Classical linearization and a retrospective explanation alone are insufficient for a first paper.

The most consequential next uncertainty is **loss transfer**. The audited IT3 regression paths use absolute discrepancy, whereas the completed dynamics use a squared surrogate. Before investing in a tighter smooth Taylor bound, check whether the same local input-path derivative mechanism exists for that nonsmooth loss. `NEXT_LOSS_AUDIT_PROTOCOL.md` fixes a bounded analytic/code check, including behavior inside residual-sign regions and at their boundaries. No absolute-loss trajectories or derivative results have been computed in this cycle. The existing next-stage requirement for a justified approximation bound remains; this audit determines which assumptions would be relevant.
