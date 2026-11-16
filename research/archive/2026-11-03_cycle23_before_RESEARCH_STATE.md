# Research state

Updated 2 November 2026, cycle 22. Working source: ../research_proposal_main.tex. Title: **When Test-Time Consistency Changes Predictions**. Author Amit Arad, advisor Assaf Shocher, acknowledgment Shoham Grunblat. Existing twice-daily automation.

## Decision and ranked directions

1. **Next feasibility question.** Can optimizer history distinguish a previously encountered task on return when the audited IT3 student weights reset? Compare identical weights, inputs and step counters with different moments. Pre-update predictions, post-update predictions and recovery risk must be measured separately. The next check is analytical and mechanism-only.
2. **Completed compatible-task baseline.** With protected rows alone, every update changing current predictions has unbounded worst-case excess squared error under the unrestricted shared linear model. With a valid compact convex feasible coefficient set, projection gives the attained minimax gain. A bounded extra source reading produces a positive example. SAFER already contains the projection/optimality principle. Close this rule as a stand-alone paper and keep its information accounting in the thesis.
3. **Completed IT3 case study.** The two-update diagnostic verified a gradient difference but failed the registered accuracy criterion. Its sign identity is closed as an independent paper claim. Moving-reference stability, PEIRA and the prior finite-time failures remain supporting evidence.

## Next action

Follow the last paragraph of notes/2026-11-02-cycle22/Linear_Result.md. Use a two-coordinate return construction, equal counters and explicit memory budgets, at most ten CPU seconds. Stop if the apparent difference is explained by the counter alone. Independent benefit and novelty evidence are prerequisites for a paper. No trained experiment scheduled.

## Evidence and boundaries

- Cycle 22. Sharp projection proof, source-only unboundedness, and exact calibration example with worst excess -5/2 for projection, zero for restoration and center, +35/2 for a permitted wrong-direction step. Extra calibration row is outside the protected rows. All eight endpoint differences checked. See notes/2026-11-02-cycle22/Linear_Result.md.

- Cycle 21. Two-paper source audit, sharp finite-step accuracy condition and paired compatible linear witness. Identical source data, current inputs, old predictions and positive initial current error admit opposite accuracy effects. Exact rational example changes MSE by -5/32 or +15/32. Record, script and arithmetic in notes/2026-11-02-cycle21.
- Cycles 15-20. Pinned IT3 tabular graph; one locally trained 1,291-parameter checkpoint; 20 pairs, two Adam steps, fixed reference, reset student weights, carried canonical optimizer state, four graph variants and no adaptation. Five of 20 batches changed gradient and final prediction; 15 were identical. Across all 40 held-out examples current-connected raised MSE by 6.14% against the baseline and 0.40% against cached-connected. All four variants raised error. Held-out labels were evaluation-only. Evidence and implementation hashes in cycle 20 Risk_Result.md and cycle 19 Diagnostic_Result.md.
- Cycles 2-14. Constructed stability separation and affine compatible-task example. Strong rotation absent in trained tanh screen. Squared-loss caching reduced excess terminal error in all four primary seeds but lost initial accuracy. Short-horizon affine forecast passed and failed later in one seed. Absolute-loss finite-time risk failed step-size checks. Nonsmooth closure results remain supporting math. Finite-step certificate failed its nonvacuity gate. Full protocols, failures and source maps remain indexed in RESEARCH_LOG.md and the byte-preserved pre-cycle-21 state.
- Current claims cover the declared fixed graphs, model assumptions and measured split. No independent real-data validation of a safe correction, streaming risk guarantee, retention experiment, or verified novelty claim. The earlier requested ChatGPT shared-Sources upload remains incomplete.

Preserve the original editor/source, hash-check concurrent edits and compile with the editor after TeX changes. Synced chat-project sources are read-only. Keep unique notes, papers, code and results. No paid/cluster work, contact or publication.
