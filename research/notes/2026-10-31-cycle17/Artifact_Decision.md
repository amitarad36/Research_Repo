# Pinned data and registered local training recipe

31 October 2026, cycle 17. Completed the bounded artifact decision saved
in cycle 16. No training, adaptation, target-value inspection or new
literature branch in this cycle. A prepared training script was syntax
parsed only; its execution and checkpoint qualification are still pending.

## Bounded official artifact check

Inspected the complete recursive Git tree at IT3 revision
100524001d890efc08911f3cba8f45f51f4aa7d1 and its README.
[Pinned tree](https://github.com/nikitadurasov/ittt/tree/100524001d890efc08911f3cba8f45f51f4aa7d1);
[pinned README](https://github.com/nikitadurasov/ittt/blob/100524001d890efc08911f3cba8f45f51f4aa7d1/README.md).
The API response has truncated=false and 24 entries. The only weight
path is exps/age/weights/pretrained_zigzag.pth, unrelated to this tabular
architecture; it was not downloaded. exps/tabular contains only
vanilla_model.ipynb. The README's Tabular Data (UCI) subsection links that
notebook without a tabular checkpoint or preprocessing bundle. The inspected
notebook trains in memory and does not save a tabular checkpoint. This is
a bounded finding about these pinned sources, not proof that the authors
have no weights elsewhere. No releases/other tasks/third-party mirrors
were searched and no authors contacted.

Source hashes and exact URLs are in fetch_manifest.json. Tree SHA256:
8277389e9857c9a8fd72a5dd8175e0988297ce0bc8ae1a08970b3085e6983068.
README SHA256:
386013f2b98f6ae77e902e586a97c7b53e1c1e088418ae1a3a9f1576b313c1b4.
Web open returned a cache miss; restricted shell retrieval failed DNS.
Approved read-only retrieval then succeeded. These access failures are
recorded, not treated as evidence of missing artifacts. No fetched code ran.

## Data provenance resolved

The notebook's cell 1 directly names the upstream Zigzag data file. Resolved
its file-history revision to 50d43a94d46790627e781bfc2dd83d583421ab78 and
saved the [pinned file](https://raw.githubusercontent.com/cvlab-epfl/zigzag/50d43a94d46790627e781bfc2dd83d583421ab78/exps/data/bostonHousing.txt)
as opaque bytes: 49,082 bytes, 506 nonempty lines, SHA256
baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a.
The API file-history response and data_manifest.json retain provenance.
No floating-point targets, features, fit statistics or examples were read.
Saving/hashing raw bytes does not mean target values were inspected.

## Registered reproduction, with explicit source deviations

There is now sufficient source and data to prepare a NEW local checkpoint,
not to claim recovery of the authors' checkpoint. TRAINING_PROTOCOL.md fixes
one seed, one final checkpoint, train-only qualification and a 120 CPU
second/30-epoch cap before any results. train_registered.py implements the
recipe without importing/executing the downloaded notebook. Its syntax was
parsed but it has NOT been executed.

Source locations reused: notebook cell 5 lines 82-121 (architecture/default
input/zero-bias initialization), cell 6 lines 11-35 (two squared losses,
last-minibatch return), cell 6 lines 105-130 (scheduler on returned last loss),
cell 12 lines 5-19 (one model, lr1e-3,30 epochs). Source notebook SHA256
6d583591e7433547c3b8fbb85f2177e51a933596f21709bc30b251be89c1e227;
exact extraction remains in cycle 9. Source scheduler is deliberately
interpreted as monitoring final-minibatch loss, not epoch-mean loss.

The same first-90% row split means 455 training and 51 held-out rows.
New feature and target statistics use ONLY training rows, unlike the
notebook's full-dataset normalization. CPU float64 computation, explicit
seed0, deterministic loader generator, omitted per-epoch held-out evaluation
and NPZ provenance artifacts are further declared reproduction choices.
Model dimensions/activation/k50, dual squared pretraining objective, batch32,
30 epochs, Adam1e-3 and scheduler semantics follow the inspected recipe.
Existing CPU Torch2.9.0+cpu is reused; no install or GPU work.

Qualification uses training rows only: default NMSE<=.75, label-input
NMSE<=.25, prediction std/target std>=.10, finite tensors and exactly450
updates. Save the final checkpoint even on failure; no alternate seed,
extra epochs, best-epoch selection or adaptation-outcome selection.
These are suitability checks for a fitted nonconstant diagnostic, not
generalization evidence. The script never parses held-out numeric values.

## Decision and next action

Proceed next cycle with the registered training run once, after checking
saved hashes, within its fixed budget. Store outputs in a fresh directory;
retain incomplete/failure artifacts. Do not run adaptation in that cycle.
If qualified, the existing separate 20-batch/four-arm/two-update plan can
then be made executable with this checkpoint and first40 held-out features
only. If qualification fails, retain the negative result and reconsider;
do not automatically tune. Source/data/runtime prerequisites are now
resolved for this NEW reproduction. Trained-fit suitability remains open,
and neither publication novelty nor useful adaptation is established.
