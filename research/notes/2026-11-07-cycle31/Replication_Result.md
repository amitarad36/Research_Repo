# Initialization changes the accuracy result and the gradient mechanism

7 November 2026, cycle 31. Completed the fixed cycle-30 evaluation of all three additional initializations. The original harmful adaptation contrast does not persist. The original five-batch mechanism criterion fails in each new seed.

| Seed | Event batches /20 | MSE baseline | Current-connected | Cached-connected | Current-stopped | Cached-stopped |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.0780949894 | 0.0757148538 | 0.0757148538 | 0.0761011432 | 0.0759132620 |
| 2 | 1 | 0.1019763980 | 0.1011865490 | 0.1011907835 | 0.1002845535 | 0.1002845535 |
| 3 | 3 | 0.0866810565 | 0.0789429682 | 0.0789441956 | 0.0786529260 | 0.0786529260 |

The seed-0 historical result remains +6.13535% current-connected MSE against no adaptation and +0.395799% against caching. The new seeds give -3.04774%, -0.77454% and -8.92708% against no adaptation. Against caching their changes are 0%, -0.00418466% and -0.00155477%. Seeds 2 and 3 pass the original strict two-comparison error-benefit criterion. Seed 1 ties caching and fails it. None passes the >=5/20 joint gradient/output event criterion. These two decisions measure different quantities and are both retained.

All four qualifying new events have unchanged discrepancy signs and changed frozen-reference input derivatives. They occur at batch 14 in seed 2 and batches 8,14,19 in seed 3. The sign term is zero. The derivative term accounts for the difference, with maximum decomposition error 1.06e-15. Seed 1 has exactly identical fresh/cached predictions in all 20 batches. Seeds 2 and 3 have 19 and 17 identical batches. There are no exact or near-zero residuals or reference preactivations at recorded evaluations.

The unweighted new-cohort means are baseline .0889174813073, current-connected .0852814570117, cached-connected .0852832776421, current-stopped .0850128742192 and cached-stopped .0849502471523. The new mean current-minus-baseline contrast is -.00363602429552 and current-minus-cached is -.000001820630400. Every new current-minus-baseline contrast is negative, so the protocol's positive-contrast persistence criterion is falsified. The cohort average is descriptive and does not substitute for seed-level outcomes.

## Verification and information budget

The protocol was fixed in cycle 30, SHA256 89c4f71213d40ddb446fc4de1abd649a5b2739764cf5858ec91178f38924f090. The path/provenance-adjusted diagnostic was saved and hashed before outputs as f5d4592dc4f1ee7bba937211553d7be8ff194c9b1d7306030c886ad5044ab829. Its diff from the source cycle-19 script is retained. Changes cover CLI input/output paths, cohort manifest verification, seed metadata and remaining aggregate CPU budget. The graph algorithm and numerical/effect thresholds were preserved.

All three original checkpoint manifests and qualification records were checked. The adapter read only 40 held-out feature rows, used no labels, and performed 480 total updates with matched shared optimizer histories. All predictions and results were hash-locked before scoring. The scorer was also saved and hashed beforehand, SHA256 6aa1556016b7fe64890a4d40c12597ef851ad7607399cbbdb29439b90f288034. It read target values only after the prediction lock. The same target rows had been examined in cycle 20, making this retrospective fixed-split sensitivity evidence.

All analytic-gradient and common-first-state checks pass. The scorer independently reconstructed every final student prediction from flat parameter vectors using NumPy, with maximum absolute error 2.23e-16. It verified equal initial parameters and moment/counter states across the four arms for every batch, scalar-square sums for every arm, paired mean identities, and zero error differences on identical-prediction batches. Full per-example predictions, normalized targets and squared errors are in risk_results.json. Raw traces contain every saved state, input derivative, sign, mask and gradient. The maximum formula error is 1.78e-15. All results are finite.

Diagnostic CPU time after model/input loading, including trace serialization, totals 11.703125 seconds against the aggregate 60-second cap. Scoring and independent verification use 3.234375 CPU seconds against the 10-second cap. Torch 2.9.0+cpu, float64, one CPU thread. No new training or tuning occurred.

## Source and decision

Official IT3 revision 100524001d890efc08911f3cba8f45f51f4aa7d1, tabular notebook cell 5 lines 82-121 supplies architecture and initialization; cell 13 lines 34-58 supplies connected discrepancy and student resets; lines 61-64,90-95 the optimizer lifetime; line 103 the two-step call. Cached-connected remains a controlled straight-through intervention. The previously audited source extraction and reproduction choices remain in cycles 15-18.
https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb

The proposal now presents the original negative result together with its fixed replication. Local initialization changes both the risk direction and which gradient term is active. The tiny or zero refresh benefit relative to caching does not establish a useful refresh method. A first-paper claim remains provisional and closed for this experiment alone.

Next, use only these saved trajectories to locate the accuracy change in the shared first step versus the second step. Score all 40 examples for each fixed seed at reset, after the first update and after the second. Check exact telescoping of error differences. Preserve all four arms and distinguish the common first step within each connected/stopped pair. No model calls, retraining, additional labels, seed selection or new input data are needed. Label this a post-hoc decomposition of the fixed cohort. It can clarify why overall adaptation helps when refreshing barely changes the result.
