# Registered checkpoint qualifies for the separate two-update diagnostic

31 October 2026, cycle 18. Completed the unchanged cycle-17 training
protocol once. This is training-only checkpoint qualification, not a
held-out performance result, official benchmark reproduction or paper claim.

## Execution and provenance

Verified all 12 cycle-17 evidence files before execution, including frozen
training script cb8cdb7553b5d2e67595ffcee3ef3be09eb82313da37dbf2b30da5b23e491f7f
and protocol 2fbc25fe4814beb51305212823ced4ca146932ad8d1b5c52ed34b1e616018802.
Data SHA256 baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a,
from the pinned notebook-linked Zigzag file; source/data URLs and revisions
remain in cycle17/Artifact_Decision.md and data_manifest.json.

Ran the saved script unchanged with existing Anaconda Python/Torch2.9.0+cpu,
NumPy2.1.3, one CPU thread, float64 computation. Seed0, fixed455 training
rows and train-only sample normalization. ReLU30/20/10,13inputs,oneoutput,
auxiliary default50,1291 parameters. Thirty epochs with15 batches each:
450 Adam updates, dual squared loss, no seed/checkpoint selection. The
declared last-minibatch scheduler retained lr=.001 throughout; this is an
observed trace, not a scheduler modification. Training CPU2.875 seconds
including import/preprocessing/metric computation and checkpoint saving,
below the120-second cap. No installs, network, GPU or extra training.

The final epoch30 checkpoint was saved with its normalization and row-index
bundle, not the best observed epoch. Float32 default weight initialization
then conversion to double, zero biases and source last-minibatch scheduler
semantics were retained as registered. Train-only normalization, explicit
seed and omitted held-out evaluation remain declared source deviations.

## Frozen qualification gates

| Training-only quantity | Result | Prespecified requirement |
|---|---:|---:|
| Default-input NMSE | .19191456293492845 | <=.75 |
| Label-input NMSE | .055213527673392965 | <=.25 |
| Prediction/target population standard deviation | .8699048379181095 | >=.10 |
| Optimizer steps | 450 | exactly450 |

All parameters and metrics are finite; all gates pass. NMSE divides the
corresponding training squared error by the normalized training target's
constant-predictor MSE (.9978021978021976). The label-input branch consumes
TRAINING labels as specified by supervised idempotent pretraining. This is
not label-free training, and its fit is not an estimate of test accuracy.
Training self-consistency MSE is .014745796396934362, so the model is not
an exact idempotent map. No stability or useful-adaptation inference follows.

No numeric values from the51 held-out rows were parsed, including features.
Their row indices and raw-file bytes/hash were known, but there was no
held-out loss, output, label statistic, plot or adaptation computation.

## Independent saved-artifact check

verify_checkpoint.py uses a separate NumPy implementation of the saved
Linear/ReLU forward map, not the training module/Torch forward. It reloads
NPZ arrays with allow_pickle=False, verifies the eight tensor names/shapes,
float64 dtype,1291 finite parameters,455/51 row indices and all hashes.
It recalculates train-only means/sample stds and every reported metric on
training rows, and reconstructs the declared scheduler transitions from
the recorded last-minibatch losses. Metrics agree to <=3.34e-16 absolute;
scaling agrees within the fixed1e-12 absolute/relative comparison (largest
absolute feature-mean discrepancy6.26e-13). All qualification decisions
agree. Verification used .015625 CPU seconds, with no additional training
or adaptation. It validates saved outputs; it is not a second training run.

Checkpoint SHA256:
edefebb1b00b35ceaf10ee1cb2248594af9909c6ecc8933f0405814fd2669e0e.
Preprocessing SHA256:
7815dfb04a28e8c2929142e8dc95442fadff37c6501086b67b7475fd9685ec42.
Full epoch trace, hashes, runtime and metrics are in training_results.json;
independent details are in verification_results.json. Do not edit these
artifacts or reuse training optimizer history for adaptation.

## Decision

Checkpoint preparation is complete and the fixed suitability gate passes.
Proceed in a separate cycle to the existing graph diagnostic without
changing its thresholds, two-step horizon, first20 batches, batch size2 or
canonical Adam history policy. This removes the trained-artifact blocker
for the NEW local reproduction, not for the authors' exact result.
The next protocol binds the qualified hashes to the prior harness plan;
first40 held-out FEATURES will become available then, while targets remain
unparsed. No adaptation was performed this cycle. A zero or negligible
effect remains a valid possible outcome. All earlier failures and novelty
limits remain in force.
