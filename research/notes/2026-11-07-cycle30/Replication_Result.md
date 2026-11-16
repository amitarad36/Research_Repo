# Initialization replication is feasible

7 November 2026, cycle 30. Three final checkpoints prepared and independently verified. The next scientific outcome is the sensitivity of adaptation error to initialization.

The original training report recorded 2.875 CPU seconds for seed 0, making a small fixed cohort feasible. Before new training, registered seeds 1, 2 and 3 with 20 CPU seconds per seed and 60 summed CPU seconds for training and independent verification. Held the dedicated minibatch-order generator at seed 0. Thus the intentional change is model initialization. The default float32 initialization, zero biases, subsequent float64 conversion, architecture, original split, training-only normalization, losses, 30 epochs and final checkpoint rule follow the locked recipe. The scheduler rule also follows the original recipe and retained learning rate .001 for all three runs.

| Initialization seed | Default training NMSE | Label-input training NMSE | Prediction/target std ratio | Qualification |
|---|---:|---:|---:|---|
| 1 | 0.176223 | 0.024459 | 0.879448 | Pass |
| 2 | 0.153137 | 0.026703 | 0.875007 | Pass |
| 3 | 0.164121 | 0.044361 | 0.877298 | Pass |

All three completed exactly 450 updates and pass the original gates <=.75, <=.25, >=.10. No seed was replaced or checkpoint selected. Independent NumPy reconstruction agrees with all saved training metrics within 7.78e-16 and verifies normalization, row indices, tensor shapes, hashes, qualification and the scheduler trajectory. Summed reported training/verification CPU time was 26.0625 seconds. This sum uses the training script's import-inclusive CPU timer and the verifier's existing after-import timer. Interpreter startup and verifier import overhead are excluded from that sum. No held-out numeric data was parsed in this cycle, and no adaptation was run.

Training-only fit does not decide whether adaptation helps. The earlier seed-0 MSE increases of 6.14% versus no adaptation and 0.40% versus cached-connected retain their original scope. The already observed first-40 evaluation targets mean the planned replication is a retrospective same-split sensitivity study. The cohort supports reporting initialization variation, without adding new domains or independent test samples.

Provenance is preserved in registered.json, the scripts, per-seed reports, NPZ arrays and verification records. Registration preceded execution. The source training executable had SHA256 cb8cdb7553b5d2e67595ffcee3ef3be09eb82313da37dbf2b30da5b23e491f7f. The new script hash is b0243508f38bbabf21743982eb4674c170715523db20bd72eda0d1f8386704cc, and the new protocol hash is 4c763ea326f6572b371e804ded4688227a65561a1d80ff3d78847df666fb7c27.

The primary source is IT3 revision 100524001d890efc08911f3cba8f45f51f4aa7d1, notebook cell 5 lines 82-121 for architecture and initialization, cell 6 lines 11-35 for dual losses and final-minibatch return, cell 6 lines 105-130 for scheduler use, and cell 12 lines 5-19 for the training settings. Cycle 17 Artifact_Decision.md records these locations and the full-data normalization difference. The saved training recipe intentionally uses training-only scaling.
https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/exps/tabular/vanilla_model.ipynb

No new thesis-level effect is claimed from checkpoint preparation. The proposal source was read and remains at the last verified scientific revision. Continue with NEXT_REPLICATION_EVALUATION.md, fixed before observing any new adaptation output.
