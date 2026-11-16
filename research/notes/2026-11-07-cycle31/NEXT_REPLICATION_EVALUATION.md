# Fixed evaluation of the initialization cohort

7 November 2026, before any adaptation or held-out predictions from seeds 1, 2 and 3.

Question: does the original harmful adaptation effect persist across these three fixed initializations? Report all three seeds separately and their unweighted mean normalized MSE, together with seed 0 as a historical reference. No selection, retuning or replacement.

Verify registered.json and evidence_manifest.json in this folder, each checkpoint's recorded hash and independent verification, the original data hash baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a, and each saved preprocessing bundle. Qualified final checkpoints are
- Seed 1, SHA256 58cfefef22ede629b273afd972f9b192eaad9f3e2a663bad481748ea8f0ef1b1.
- Seed 2, SHA256 f35038b6c58d3bd2bce1826da86e0454d6ed322f225a8d1fe9baf6c275d106a6.
- Seed 3, SHA256 0207516486589da7853cff045b2ba9dc7a60864118b958d4b17e7790b3c0de17.

Reuse the cycle-19 four-graph diagnostic algorithm. Refactor only input/output paths and provenance checks needed to accept the new locked artifacts. Save and hash the executable before evaluating any of these checkpoints on held-out inputs. Preserve the original assertions for common first updates, analytic gradients, second-step decomposition, cache immutability, canonical optimizer history and nonfinite values. Preserve all failed outputs.

For each seed use first 40 held-out feature rows 455:495, their original order and saved training normalization, batch size 2, four graph arms and no adaptation. Reset weights after each batch. Start canonical Adam fresh at the call boundary and copy its history into each arm at every batch. Only current-connected advances the shared history to the next batch. Two steps at lr1e-4, beta(.9,.999), eps1e-8, no EMA. Four arms times two steps times 20 batches times three seeds gives 480 updates. Do not reuse pretraining moments or the historical seed-0 adaptation state.

Preserve the numerical threshold 1e-10*max(1,comparison scale). Per seed, the original mechanism criterion is second-gradient difference above 1e-4*max(norms,1e-8) AND final-output RMS difference above 1e-6*max(RMS(q0),1) in at least 5/20 batches. Report every seed's event count and magnitude, including zero. A numerical mismatch invalidates that seed's mechanistic interpretation and stops downstream scoring for it.

Use an aggregate 60 reported process-CPU-second cap for the three diagnostic computations, and a separate 10-second scoring cap. Include model work and serialization in the diagnostic time. Stop on budget failure and preserve incomplete evidence. No new training, network, installs, paid compute or GPU. No extension of batch count, horizon, seeds or stopping rule based on outcomes.

Save and hash all predictions before scoring. As in cycle 20, score all 40 targets using saved training target normalization, with labels restricted to the scoring process. Compute all five arms' MSE, per-example squared errors and current-connected minus baseline/cached-connected paired means. Independently recompute scalar sums and check paired mean identities at the original tolerance. Report absolute and relative changes for every seed. Joint benefit requires strictly lower current-connected MSE against BOTH no adaptation and cached-connected. Any failure of either comparison rejects the joint benefit for that seed. Preserve mixed results and the original seed-0 failure.

The descriptive sign-persistence criterion for the original harm is positive current-minus-baseline and current-minus-cached MSE in each of the three new seeds. Any zero or negative contrast falsifies sign persistence across this fixed cohort. Also report unweighted mean contrasts across the three new seeds, without replacing per-seed reporting. A positive mean does not establish uniformity. No significance or population-wide claim follows from three selected-in-advance seeds. Targets from this fixed split were observed earlier, so this is retrospective robustness evidence, with no retention or recovery claim.
