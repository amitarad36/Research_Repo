# Research state

Updated 2 November 2026, cycle 21. Working source: ../research_proposal_main.tex. Title: **When Test-Time Consistency Changes Predictions**. Author Amit Arad, advisor Assaf Shocher, acknowledgment Shoham Grunblat. Existing twice-daily automation.

## Decision and ranked directions

1. **Open thesis question.** Can a learner certify an update that improves current predictions while retaining exact predictions on an earlier compatible task? Start with a finite linear model, explicit source information budget and a sharp threshold for current squared error. Study whether any extra information needed for the threshold is available before current labels arrive. Compare against source restoration, feasible-set center and protected-gradient baselines. First-paper viability is conditional on a distinct, nonvacuous result and independent validation.
2. **Completed thesis case study.** The matched-state absolute-loss diagnostic explains how an update changes but fails the registered current-accuracy criterion. The cycle-21 IT3/RMT claim audit demotes this event identity as a stand-alone paper. Preserve this branch at its observed outcome.
3. **Longer-term question.** Drift, retention on recurring domains and recoverability require a justified useful update, separate metrics and matched access to old information. The PEIRA-inspired scalar tracking result remains supporting background.

## Next theorem and decision rule

Follow the last two paragraphs of notes/2026-11-02-cycle21/Thesis_Decision.md. Use source-compatible linear coefficients and a declared prior constraint to derive an attained worst-case finite-step MSE change. Test whether any nonzero retention-preserving correction can be certified without hidden target labels or a tautological assumption. Stop this route if it collapses to standard restoration/projection or requires unavailable information. Budget at most ten CPU seconds for small exact arithmetic. No new trained runs are pending.

## Evidence and boundaries

- Cycle 21. Two-paper source audit, sharp finite-step accuracy condition and paired compatible linear witness. Identical source data, current inputs, old predictions and positive initial current error admit opposite accuracy effects. Exact rational example changes MSE by -5/32 or +15/32. Record, script and arithmetic in notes/2026-11-02-cycle21.
- Cycles 15-20. Pinned IT3 tabular graph; one locally trained 1,291-parameter checkpoint; 20 pairs, two Adam steps, fixed reference, reset student weights, carried canonical optimizer state, four graph variants and no adaptation. Five of 20 batches changed gradient and final prediction; 15 were identical. Across all 40 held-out examples current-connected raised MSE by 6.14% against the baseline and 0.40% against cached-connected. All four variants raised error. Held-out labels were evaluation-only. Evidence and implementation hashes in cycle 20 Risk_Result.md and cycle 19 Diagnostic_Result.md.
- Cycles 2-14. Constructed stability separation and affine compatible-task example. Strong rotation absent in trained tanh screen. Squared-loss caching reduced excess terminal error in all four primary seeds but lost initial accuracy. Short-horizon affine forecast passed and failed later in one seed. Absolute-loss finite-time risk failed step-size checks. Nonsmooth closure results remain supporting math. Finite-step certificate failed its nonvacuity gate. Full protocols, failures and source maps remain indexed in RESEARCH_LOG.md and the byte-preserved pre-cycle-21 state.
- Current claims cover the declared fixed graphs, model assumptions and measured split. No independent real-data validation of a safe correction, streaming risk guarantee, retention experiment, or verified novelty claim. The earlier requested ChatGPT shared-Sources upload remains incomplete.

Preserve the original editor/source, hash-check concurrent edits and compile with the editor after TeX changes. Synced chat-project sources are read-only. Keep unique notes, papers, code and results. No paid/cluster work, contact or publication.
