# Registered local training reproduction — not yet run

31 October 2026, before any target values were parsed or model trained.
Purpose: obtain ONE qualified checkpoint for the already specified
two-update diagnostic, not improve a benchmark or establish a paper claim.
No seed search, checkpoint selection by adaptation or held-out labels.

## Fixed inputs and information boundary

Use bostonHousing.txt from cvlab-epfl/zigzag revision
50d43a94d46790627e781bfc2dd83d583421ab78, SHA256
baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a.
506 nonempty rows are recorded from opaque bytes. Require exactly 14 tokens
per row. The first floor(.9*506)=455 rows train; final 51 rows are held out,
matching the notebook's row-order split. In this training run parse ONLY
the first 455 rows into floating point. Do not parse held-out feature or
target values, evaluate held-out fit, plot or log held-out target statistics.
Hashing raw file bytes is allowed; it is not target inspection.

Fit feature and target means/stds ONLY on training rows, sample std
(correction=1). Reject zero/nonfinite scales; do not repair/drop columns
after seeing results. Store all scaling and exact row indices with weights.
This differs deliberately from the notebook's full-data normalization and
prevents held-out target statistics entering the model. Preserve the source's
13 inputs, one output, widths 30/20/10, Linear/ReLU, default auxiliary k=50.

## Fixed optimization and compute

Existing C:\Users\amita\anaconda3\python.exe, Torch 2.9.0+cpu, CPU only,
one Torch/BLAS thread. Set Python/NumPy/Torch seed 0; a dedicated DataLoader
generator seed 0; num_workers=0. Construct Linear layers with default
float32 initialization, zero all biases as the source class does, then
convert to float64 before data/optimization. This preserves a declared
initialization convention while using the planned diagnostic precision.
No other model, seed or dtype fallback.

Train 30 epochs, batch size 32, shuffle=True, drop_last=False: 15 batches
per epoch, exactly 450 optimizer steps. Adam lr=1e-3, betas=(.9,.999),
eps=1e-8, weight_decay=0, amsgrad=False, foreach=False.
Loss = mean((f(x,50)-y)^2) + mean((f(x,y)-y)^2), matching original
zigzag pretraining (new_zz=False), with normalized training y.
ReduceLROnPlateau mode=min, factor=.5, patience=10, threshold=1e-4,
threshold_mode=rel, cooldown=0, min_lr=0, eps=1e-8. Feed the final minibatch's
pre-update loss each epoch, as the inspected train_epoch/train_model code
does; do not silently replace it with the epoch mean. Also log epoch mean.

Hard CPU limit 120 seconds including import/preprocessing/training/metrics;
check between minibatches. Save an incomplete/failed artifact on timeout
or error where possible, do not extend the cap or resume automatically.
No GPU, installations, cluster, network during training or adaptation run.

## Checkpoint qualification fixed before outcomes

Use ONLY the final epoch-30 state. Save it even if qualification fails;
never choose a better earlier state. Require finite parameters/metrics and
all 450 steps. On training rows only, define baseline MSE as the variance
of normalized y about its training mean. Require:

1. default-input training MSE / baseline MSE <= .75;
2. label-input training MSE / baseline MSE <= .25;
3. population std(f(x,50)) / population std(y) >= .10.

These are prespecified suitability checks for a fitted nonconstant fixture,
not generalization or adaptation evidence. Report both constituent losses,
prediction dispersion and training-input self-consistency, including all
failures. Do not lower thresholds, increase epochs, choose another seed,
change normalization or select another checkpoint if a gate fails.

## Artifacts and next decision

Save state_dict tensors in an NPZ archive (no pickle), scaling/indices in
another NPZ, architecture/dtype/library/source/data/protocol/script hashes,
full epoch trace, timing, number of steps and qualification results as JSON.
Training optimizer state is NOT the adaptation state; the later canonical
adaptation Adam starts fresh at ttt-call entry as the source does.

If qualified, the next separate cycle may execute the existing cycle-16
20-batch/four-arm/two-step protocol after verifying provenance and locking
the harness code. Only first 40 held-out FEATURES are then needed; targets
remain unparsed. This is not authorization to merge training and adaptation
into an outcome-driven sweep. If qualification fails, save the negative
result and make a new research decision; do not automatically retune.
