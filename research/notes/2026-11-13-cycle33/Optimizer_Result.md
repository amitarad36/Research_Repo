# Adam state changes accuracy at fixed starting weights
13 November 2026, cycle33. New checked local experiment; retrospective fixed cohort, no novelty claim.

The original seed-zero harm is not invariant to optimizer state. Clearing both moments at the existing counter changes +6.13535% error to -0.11487% versus no adaptation. Starting Adam fully fresh gives +1.41046%. Seeds1/2 improve more with either reset; seed3 improves less. Neither reset uniformly beats carried history. This identifies optimizer state as a consequential part of the empirical result, not a reliable reset method or task-specific memory benefit.

| Seed | Baseline MSE | Carried MSE | Zero moments, matched counter MSE | Fresh MSE |
|---|---:|---:|---:|---:|
|0|.07221261683645701|.07664311396811949|.07212966327938986|.07323114694519257|
|1|.07809498939733860|.07571485384193946|.07401957679777736|.07275175066442980|
|2|.10197639804515209|.10118654902747573|.09904187482797092|.09807756499251274|
|3|.08668105647926636|.07894296816576832|.08236828991691550|.07953415914835738|

## Design and provenance
Protocol and script saved before new outputs. Registration SHA256 a804fed2b92b8aa55bb243c747a3d2c027e848ae9c88236209ee8b2b869cbaf3. All four seeds and20batches, same source-derived architecture and data scaling. Exactly480updates:160carried replay,320interventions;5.109375CPU seconds after imports including trace save,60second cap. One CPU thread torch2.9.0+cpu, double. No retraining, new targets, learning-rate selection, exclusions or tuning.

For each saved batch start, restore trained theta and fork three optimizer states. Carried m/v/counter; zero m/v at same counter; zero m/v/counter. Two current-connected mean-absolute-discrepancy steps lr1e-4 beta.9/.999 eps1e-8. Reference weights frozen but input gradient live. Archived canonical history is external to every fork; forks do not update the next batch's state. Input rows455:495 only features parsed. Read previous target arrays only in separate scorer after prediction lock. Independent NumPy scoring checks scalar sums within2.78e-17. Original risk scores reproduced.

Pinned primary implementation: https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb
Cell13 lines34-58 loss/reset/updates,61-64 and90-95 optimizer lifetime,103two steps. Source extraction retained cycle9/tabular_cells.txt. Local cycle19 diagnostic imported only definitions, no main execution. Traces, checkpoints and preprocessing paths/hashes in registered.json. All source manifest entries for cycles18/30/31/32 verified before setup. No fresh literature search or new prior-art claim.

## Validation
All checks passed. Initial weights and gradient identical for all three arms. State assignment exact; counters0,2,...,38 before each original batch; nonnegative v. Carried replay matches every initial and final theta/m/v/counter and both gradients exactly, as well as final predictions. All arms match on first batch. Independent NumPy Adam recurrence max theta5.55e-17,m4.44e-16,v2.22e-16,counter0. Independent NumPy forward maximum6.66e-16. Full state and gradient arrays retained in intervention_trace.npz. Prediction file SHA2560adecac616bbffb1278272e9dbe30b3710f6fc69f0b2c55e87067eedf78a2e6e,locked before scoring. Perexample scores, all firststep scores and noadaptation predictions retained.

## Interpretation and failure criteria
The predeclared preservation of each seed's baseline-relative sign across all arms fails for seed0. Both proposed uniform-reset advantages fail for seed3. Do not suppress either failure or call the small seed0 benefit statistically established. Same40inputs,4initializations, already observed labels; adaptation and evaluation inputs coincide. No independent test cohort or stream drift. No labels in updates; no labelled stopping rule.

Clearing both moments at matched counter estimates their joint effect; no separate m/v attribution. Changing counter with zero moments alters bias correction and subsequent gradients. This is an intervention conditional on archived starting histories, not evidence about general continual-policy performance. Full reset is also executable without historical moments, but this experiment is not a recurring-domain benchmark. Carrying history is better for seed3 and worse for0/1/2. No retention/recovery claim.

## Next bounded question
Use saved first-update gradients, m/v and displacements to determine whether the optimizer-state contrast can be accounted for by a scalar step change or includes an observable direction change. Before further model updates, derive and check the exact fixed-denominator decomposition of Adam's numerator into old first moment and current gradient; account separately for second-moment normalization and counter scaling. Evaluate best nonnegative scalar fit using unlabeled parameter/output displacements with residual norms reported for every batch and seed. This is attribution, not novelty or an accuracy guarantee. Avoid interpreting the post-update nonlinear output of a rescaled parameter step without an explicit forward evaluation. A full intervention separating first/second moments is a later question only if necessary.
