# Research state

Updated 3 November 2026, cycle 23. Working source: ../research_proposal_main.tex. Title: **When Test-Time Consistency Changes Predictions**. Author Amit Arad, advisor Assaf Shocher, acknowledgment Shoham Grunblat. Existing twice-daily automation.

## Decision and ranked directions

1. **Optimizer history on return.** The matched-counter affine check showed a first-moment sign reversal even with equal second moments and identical pre-return predictions. For this scalar probe the repeated-task output can be matched by rescaling a fresh Adam step. The next mechanism question is whether the difference survives a scalar step-size control on multiple return probes. A task-specific benefit remains untested.
2. **Completed compatible-task baseline.** With protected rows alone, every update changing current predictions has unbounded worst-case excess squared error under the unrestricted shared linear model. With a valid compact convex feasible coefficient set, projection gives the attained minimax gain. A bounded extra source reading produces a positive example. SAFER already contains the projection/optimality principle. Close this rule as a stand-alone paper and keep its information accounting in the thesis.
3. **Completed IT3 case study.** The two-update diagnostic verified a gradient difference but failed the registered accuracy criterion. Its sign identity is closed as an independent paper claim. Moving-reference stability, PEIRA and the prior finite-time failures remain supporting evidence.

## Next action

Follow notes/2026-11-03-cycle23/Return_Moments.md, last paragraph. Check a multi-probe return with a matched non-A history, matched recent gradients and a fresh step scaled optimally without labels. Record pre-return predictions, post-update vector and its distance to the fresh-update ray. Ten CPU seconds for constructed arithmetic. Direction and useful task recovery are separate gates.

## Evidence and boundaries

- Cycle 23. A,B and C,B histories give opposite first return changes in a connected two-coordinate model, at common step five. Matched-counter cleared moments and fresh Adam recorded separately. A first-moment difference is sufficient. Scalar step size explains the A,B-versus-fresh difference on this probe. See notes/2026-11-03-cycle23.

- Cycle 22. Sharp projection proof, source-only unboundedness, and exact calibration example with worst excess -5/2 for projection, zero for restoration and center, +35/2 for a permitted wrong-direction step. Extra calibration row is outside the protected rows. All eight endpoint differences checked. See notes/2026-11-02-cycle22/Linear_Result.md.

- Cycle 21. Two-paper source audit, sharp finite-step accuracy condition and paired compatible linear witness. Identical source data, current inputs, old predictions and positive initial current error admit opposite accuracy effects. Exact rational example changes MSE by -5/32 or +15/32. Record, script and arithmetic in notes/2026-11-02-cycle21.
- Cycles 15-20. Pinned IT3 tabular graph; one locally trained 1,291-parameter checkpoint; 20 pairs, two Adam steps, fixed reference, reset student weights, carried canonical optimizer state, four graph variants and no adaptation. Five of 20 batches changed gradient and final prediction; 15 were identical. Across all 40 held-out examples current-connected raised MSE by 6.14% against the baseline and 0.40% against cached-connected. All four variants raised error. Held-out labels were evaluation-only. Evidence and implementation hashes in cycle 20 Risk_Result.md and cycle 19 Diagnostic_Result.md.
- Cycles 2-14. Constructed stability separation and affine compatible-task example. Strong rotation absent in trained tanh screen. Squared-loss caching reduced excess terminal error in all four primary seeds but lost initial accuracy. Short-horizon affine forecast passed and failed later in one seed. Absolute-loss finite-time risk failed step-size checks. Nonsmooth closure results remain supporting math. Finite-step certificate failed its nonvacuity gate. Full protocols, failures and source maps remain indexed in RESEARCH_LOG.md and the byte-preserved pre-cycle-21 state.
- Current claims cover the declared fixed graphs, model assumptions and measured split. No independent real-data validation of a safe correction, streaming risk guarantee, retention experiment, or verified novelty claim. The earlier requested ChatGPT shared-Sources upload remains incomplete.

Preserve the original editor/source, hash-check concurrent edits and compile with the editor after TeX changes. Synced chat-project sources are read-only. Keep unique notes, papers, code and results. No paid/cluster work, contact or publication.
