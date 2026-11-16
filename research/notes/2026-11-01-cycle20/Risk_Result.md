# Saved-prediction accuracy decision

The cycle-19 protocol was executed once on the locked predictions for rows 455 through 494 inclusive. The scoring script was saved and hash-locked before any held-out target was parsed. All six cycle-19 manifest entries, training scaling and pinned dataset hashes were verified. Evaluation used only these 40 targets with training target normalization. Model calls and optimizer steps were zero.

| Arm | Normalized MSE | Original-unit MSE |
| --- | ---: | ---: |
| No adaptation | 0.072212616836457 | 6.509612374967083 |
| Current-connected | 0.07664311396811949 | 6.908999908877443 |
| Cached-connected | 0.07634095719944575 | 6.881761961733204 |
| Current-stopped | 0.07638961120814848 | 6.886147881410652 |
| Cached-stopped | 0.07474923312121733 | 6.738275861781515 |

The primary paired mean difference, current-connected minus no adaptation, is +0.00443049713166249 normalized squared units, a 6.13535% error increase. Current-connected minus cached-connected is +0.00030215676867374, a 0.395799% increase relative to cached-connected. The registered criterion required both differences to be negative. It fails. Every adaptation arm has higher MSE than the baseline.

All 40 examples enter every aggregate. The 15 batches with identical current/cached connected predictions give exactly zero paired difference. The two computations of all paired mean differences agree within 1.74e-17. A separate Python scalar-square and math.fsum verifier confirms every per-example error and the aggregate decision. Computation after loading was below the resolution of the process CPU timer, reported as 0.0 seconds, within the 10-second cap. The score file includes every prediction, normalized target, squared error and all paired aggregate differences.

The cycle-19 mechanistic observation persists. Five batches changed their gradient and final prediction when the reference input was refreshed. The cycle-20 score associates that intervention with higher aggregate error in the declared setting. This comparison is descriptive and conditional on one local checkpoint, the first 40 held-out rows, same-input post-update evaluation and matched canonical optimizer histories. It provides evidence against the registered benefit claim for this setting. Population effects, distribution drift, continual retention and recovery remain open.

Evidence resides in score_saved_predictions.py, registered.sha256, run/risk_results.json, verify_scores.py and verification_results.json. The prior NEXT_RISK_PROTOCOL.md is copied verbatim as PROTOCOL.md. Source architecture, official revision 100524001d890efc08911f3cba8f45f51f4aa7d1, notebook exps/tabular/vanilla_model.ipynb, cell 13 and training deviations are documented in cycles 15 through 19. No new literature claim is made.

The empirical branch is closed at its preset accuracy decision. Preserve the negative result and all earlier failures. The next task is a focused thesis/paper claim audit using the closest IT3 and consistency-adaptation analyses. Determine what the sign/Jacobian event analysis adds beyond standard gradient alignment, identify the assumptions needed to link an event to current error, and specify a minimum independent validation requirement. Publication viability remains provisional.
