# Error changes during both updates

12 November 2026, cycle 32. Post-hoc accounting from the saved four-seed trajectories. No new model calls, training, adaptation, raw-data access or additional targets.

The sign of the mean error change is already present after the first update within each fresh/cached pair. Both first and second updates increase error in seed 0 and decrease it in seeds 1/2/3. This holds in all four graph arms. Each statement is about the mean over the 40 evaluated examples. Individual examples can move in the other direction.

| Seed | Arm | Reset MSE | After first | After second | First increment | Second increment |
|---|---|---:|---:|---:|---:|---:|
| 0 | current-connected | 0.072212617 | 0.074494638 | 0.076643114 | 0.002282021 | 0.002148476 |
| 0 | cached-connected | 0.072212617 | 0.074494638 | 0.076340957 | 0.002282021 | 0.001846319 |
| 0 | current-stopped | 0.072212617 | 0.074268671 | 0.076389611 | 0.002056054 | 0.002120940 |
| 0 | cached-stopped | 0.072212617 | 0.074268671 | 0.074749233 | 0.002056054 | 0.000480562 |
| 1 | current-connected | 0.078094989 | 0.076932704 | 0.075714854 | -0.001162285 | -0.001217850 |
| 1 | cached-connected | 0.078094989 | 0.076932704 | 0.075714854 | -0.001162285 | -0.001217850 |
| 1 | current-stopped | 0.078094989 | 0.077074061 | 0.076101143 | -0.001020928 | -0.000972918 |
| 1 | cached-stopped | 0.078094989 | 0.077074061 | 0.075913262 | -0.001020928 | -0.001160799 |
| 2 | current-connected | 0.101976398 | 0.101728712 | 0.101186549 | -0.000247687 | -0.000542162 |
| 2 | cached-connected | 0.101976398 | 0.101728712 | 0.101190784 | -0.000247687 | -0.000537928 |
| 2 | current-stopped | 0.101976398 | 0.101307550 | 0.100284553 | -0.000668848 | -0.001022997 |
| 2 | cached-stopped | 0.101976398 | 0.101307550 | 0.100284553 | -0.000668848 | -0.001022997 |
| 3 | current-connected | 0.086681056 | 0.082309450 | 0.078942968 | -0.004371606 | -0.003366482 |
| 3 | cached-connected | 0.086681056 | 0.082309450 | 0.078944196 | -0.004371606 | -0.003365255 |
| 3 | current-stopped | 0.086681056 | 0.082092030 | 0.078652926 | -0.004589026 | -0.003439104 |
| 3 | cached-stopped | 0.086681056 | 0.082092030 | 0.078652926 | -0.004589026 | -0.003439104 |

The source traces contain the student prediction before each of two updates. Therefore q at step index 1 is the prediction after the first update. Reshaping the stored order as batch, arm, step, example, output gives 20x4x2x2x1. All initial q arrays agree exactly with the stored no-adaptation output. First-update predictions agree exactly within arms 0/1 and within arms 2/3. Connected and stopped pairs are compared separately.

For each of the 640 seed-arm-example records, let d1=q1-q0 and d2=q2-q1. Verified the finite-step identities
E1-E0 = 2(q0-y)d1+d1^2,
E2-E1 = 2(q1-y)d2+d2^2,
E2-E0 = (E1-E0)+(E2-E1).
Maximum absolute discrepancy 2.246466901389965e-16. Independent Python scalar squares and math.fsum agree with mean errors within 2.7755575615628914e-17. All final MSEs reproduce the previously reported values within 1e-12. Every source evidence-manifest entry was checked before processing. All four seed traces and source manifest hashes are recorded in step_results.json.

Script output initially exceeded the tool response limit. Repeated the same read-only calculation with compact JSON serialization to retain the complete per-example record. Numeric computations and thresholds were identical. Retained the complete second output and executable. Reported CPU 0.03125 seconds after imports, under the fixed 10-second calculation cap.

The earlier statement that fresh/cached differences are small remains accurate. The first-step and second-step gains show that the larger adaptation benefit is distributed across both updates of the shared process. This arithmetic identifies when the mean error changes. It does not identify a cause for the cross-seed change in risk direction.

Crucial policy boundary: optimizer moments and counter before each batch came from the canonical current-connected two-update run. A recorded first-update score describes stopping locally at that point with inherited two-update history. It does not evaluate deploying one update for every batch. Using evaluation labels to choose which step to keep would also introduce label access. No stopping policy is proposed from these scores.

The source code remains IT3 revision 100524001d890efc08911f3cba8f45f51f4aa7d1, tabular cell 13 lines 34-58 for the loss, reset and updates, lines 61-64 and 90-95 for optimizer lifetime, line 103 for two updates. Our trace schema and writes are in cycle19/diagnostic.py and the cycle31 path-adjusted copy, whose archived hashes were verified. The per-step squared-error identity is reproduced elementary algebra. No novelty claim.
https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb

Next substantive question is whether carried Adam history changes the risk direction on this fixed cohort. Prepare a bounded matched-history intervention using saved per-batch starting states before launching any new adaptation. Compare the archived carried state with zeroed moments at the matched counter and a fresh optimizer at counter zero, with common restored weights, same batch, two updates, current-connected graph and no-adaptation control. Keep the archived history as a common external history for every fork so the contrast isolates the current-batch intervention. Predeclare all seeds, budgets, scoring and failure criteria. This would be a state intervention on the observed trajectories, with a separately stated boundary for any policy-level conclusion.
