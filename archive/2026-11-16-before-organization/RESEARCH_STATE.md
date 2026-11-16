# Research state

Updated 16 November 2026, cycle 34. Working source: ../research_proposal_main.tex. Title: **When Test-Time Consistency Changes Predictions**. Author Amit Arad, advisor Assaf Shocher, acknowledgment Shoham Grunblat. Existing twice-daily automation.

## Decision and ranked directions

1. **Adviser review.** Recommend the scoped thesis on attribution of IT3 updates and their measured predictive consequences. The toy optimizer analyses remain thesis controls. Close the current recovery-paper candidate. Kang and Lee already vary Adam moment policies and learning-rate adjustment; PeTTA already compares recurring-domain resets. Neither inspected comparison isolates the reciprocal domain-specific return-risk contrast. The current evidence supplies neither a positive prediction beyond recent gradients nor a qualified independent compatible stream. See notes/2026-11-05-cycle25/Adviser_Decision.md.
2. **Completed compatible-task baseline.** With protected rows alone, every update changing current predictions has unbounded worst-case excess squared error under the unrestricted shared linear model. With a valid compact convex feasible coefficient set, projection gives the attained minimax gain. A bounded extra source reading produces a positive example. SAFER already contains the projection/optimality principle. Close this rule as a stand-alone paper and keep its information accounting in the thesis.
3. **Completed IT3 case study and initialization replication.** The original seed-0 failure is retained. Fixed seeds 1/2/3 reduce error by 3.05%/.77%/8.93% against no adaptation; fresh and cached inputs are equal or nearly equal. Their qualifying events number 0/1/3 and arise through the derivative term with fixed residual signs. The original harm fails its persistence criterion. The decomposition remains thesis evidence. Moving-reference stability, PEIRA and the prior finite-time failures remain supporting evidence.

## Next action

Cycle 34 completes the first-update direction-versus-scale question. All 76 noninitial batches have carried updates outside the nonnegative fresh-step ray, including nonzero linearized-output residuals. Do not add more variants on the same 40 examples merely to seek an accuracy benefit. The user needs to submit soon: prioritize consolidating the existing TeX into a focused proposal with three objectives, a compact preliminary-results table, closest-prior-art positioning, and an independent validation plan. Preserve the long technical record and proofs in research backups/supporting notes. Page limit/template and deadline were requested but remain unknown; use a clearly stated concise working format if none arrives. Any later nonlinear rescaled-step claim requires direct forward evaluation. Evidence: notes/2026-11-16-cycle34/Direction_Result.md.

## Evidence and boundaries

- Cycle 34. Reproduced exact first-step Adam decomposition at fixed historical denominator and epsilon-aware counter scaling. All 76 noninitial batches reject a pure nonnegative fresh-step rescaling at the registered tolerance; four initial controls agree. Per-seed median parameter residuals 70.5%/52.5%/90.5%/55.0%; linearized-output residuals 1.30%/3.96%/16.19%/1.65%. Counter-only relative correction below .404%. Zero models/updates/targets, 1.015625 CPU seconds; algebra/state checks pass. Direction is established in the declared parameterization and probes, not accuracy attribution or a nonlinear scaled-step experiment.

- Cycle 33. Exactly 480 updates with common per-batch weights/input and archived canonical history. Carried replay is exact. Zero moments at matched counter yields baseline-relative MSE changes -.115%/-5.219%/-2.878%/-4.975% for seeds 0/1/2/3; fresh Adam +1.410%/-6.842%/-3.823%/-8.245%. Both resets improve versus carried for 0/1/2 and worsen for 3. Sign invariance and uniform-reset advantage fail. Independent Adam and forward checks within 6.67e-16; 5.109375 CPU seconds. Predictions locked before prior-target scoring. Conditional batch intervention, no independent stream or recovery result.

- Cycle 32. Saved scores at reset/first/second update reproduce every final MSE. Both increments have the same seed-level sign in all arms. Verified per-example finite-step expansion and telescoping for 640 records within 2.25e-16; scalar means within 2.78e-17. Zero model calls or new targets.

- Cycle 31. All three additional initializations improve current-connected MSE versus baseline. The mean contrasts are -.00363602430 versus baseline and -.00000182063 versus cached. Joint strict benefit passes for seeds 2/3 and ties caching for seed 1. None meets the 5/20 mechanism-event gate. All four new events are derivative-only. All 480 updates and independent prediction/scoring checks passed.

- Cycle 30. Fixed seeds 1/2/3 qualified with default training NMSE .17622/.15314/.16412. All three independent verifications passed within 7.78e-16. Reported summed CPU time 26.0625 seconds. Zero adaptation steps or new held-out numeric access. Checkpoints, hashes and the fixed evaluation protocol are retained.

- Cycle 29. All 442 accepted synthetic probes matched direct full gradients within 4.45e-16; 390 abstained. All accepted cases had equal fresh/cached coefficients. Region setup plus one query took 197.27 microseconds versus 133.53 for fresh forward/input-backward at batch 16. No one-query acceleration claim.

- Cycle 28. Three references share cached value, derivative and fresh value, yet give fresh-connected fields 1, 2 and -1. Twelve rational cases and 24 derivative checks passed. This closes the proposed summary-only shortcut at the stated information budget.

- Cycle 27. Established an exact within-region test for the second gradient. The rational check returned 42 equal, five different and 50 abstentions. Coverage and speed on a trained checkpoint remain open.

- Cycle 26. Active and neutral bridges produce the same matching advantages on both return domains (+.00010703629). All eight cells checked under a fixed compatible target. Reciprocal effect, bridge interaction and recovery now have separate criteria.

- Cycle 25. Two-primary-source comparison and bounded feasibility inventory complete. The opening decision section in the existing TeX states the thesis package, paper decision, matched controls and falsification criterion.

- Cycle 24. Best fresh scale .75204 leaves L2 residual .000186697 for A,B, fully accounted for by coordinate two from B in this fixture. Neutral-prefix and matched-gradient alias controls recorded. See notes/2026-11-04-cycle24.

- Cycle 23. A,B and C,B histories give opposite first return changes in a connected two-coordinate model, at common step five. Matched-counter cleared moments and fresh Adam recorded separately. A first-moment difference is sufficient. Scalar step size explains the A,B-versus-fresh difference on this probe. See notes/2026-11-03-cycle23.

- Cycle 22. Sharp projection proof, source-only unboundedness, and exact calibration example with worst excess -5/2 for projection, zero for restoration and center, +35/2 for a permitted wrong-direction step. Extra calibration row is outside the protected rows. All eight endpoint differences checked. See notes/2026-11-02-cycle22/Linear_Result.md.

- Cycle 21. Two-paper source audit, sharp finite-step accuracy condition and paired compatible linear witness. Identical source data, current inputs, old predictions and positive initial current error admit opposite accuracy effects. Exact rational example changes MSE by -5/32 or +15/32. Record, script and arithmetic in notes/2026-11-02-cycle21.
- Cycles 15-20. Pinned IT3 tabular graph; one locally trained 1,291-parameter checkpoint; 20 pairs, two Adam steps, fixed reference, reset student weights, carried canonical optimizer state, four graph variants and no adaptation. Five of 20 batches changed gradient and final prediction; 15 were identical. Across all 40 held-out examples current-connected raised MSE by 6.14% against the baseline and 0.40% against cached-connected. All four variants raised error. Held-out labels were evaluation-only. Evidence and implementation hashes in cycle 20 Risk_Result.md and cycle 19 Diagnostic_Result.md.
- Cycles 2-14. Constructed stability separation and affine compatible-task example. Strong rotation absent in trained tanh screen. Squared-loss caching reduced excess terminal error in all four primary seeds but lost initial accuracy. Short-horizon affine forecast passed and failed later in one seed. Absolute-loss finite-time risk failed step-size checks. Nonsmooth closure results remain supporting math. Finite-step certificate failed its nonvacuity gate. Full protocols, failures and source maps remain indexed in RESEARCH_LOG.md and the byte-preserved pre-cycle-21 state.
- Current claims cover the declared fixed graphs, model assumptions and measured split. No independent real-data validation of a safe correction, streaming risk guarantee, retention experiment, or verified novelty claim. The earlier requested ChatGPT shared-Sources upload remains incomplete.

Preserve the original editor/source, hash-check concurrent edits and compile with the editor after TeX changes. Synced chat-project sources are read-only. Keep unique notes, papers, code and results. No paid/cluster work, contact or publication.
