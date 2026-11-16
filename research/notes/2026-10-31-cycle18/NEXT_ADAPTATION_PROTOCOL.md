# Execute the unchanged two-update graph diagnostic once

The cycle-18 qualified final checkpoint and preprocessing are fixed:
checkpoint SHA256 edefebb1b00b35ceaf10ee1cb2248594af9909c6ecc8933f0405814fd2669e0e;
preprocessing SHA256 7815dfb04a28e8c2929142e8dc95442fadff37c6501086b67b7475fd9685ec42.
Verify cycle18 evidence manifest, qualification and independent verifier,
plus cycle17 raw data hash. No retraining or alternative checkpoint.

Implement the existing cycle16 HARNESS_PLAN.md exactly; an unchanged copy
is included here. Save the executable and its hash BEFORE evaluating held-out
model outputs or inspecting event/effect results. Then one run is authorized
under that plan's CPU60-second computation cap; this is a separate cycle
from training. No extra steps, batches, seeds, feature masks or outcome-driven
debugging that silently overwrites failed artifacts. Preserve any failure.

Parse only feature tokens0:13 from raw rows455:495 (40inputs); do not convert
token13 (target) to float or compute any held-out target statistic. Normalize
features using the saved TRAINING statistics. Keep file order,20 batches
of2, no corruption, CPUfloat64, source-sized ReLU/k50 architecture. NPZ
arrays are loaded with allow_pickle=False and mapped to named parameters.

Four arms: current-connected, cached-connected straight-through,
current-stopped, cached-stopped; plus no adaptation. Fixed reference,
two Adam steps at1e-4, no EMA. Fresh canonical Adam at the ttt-call start;
reset weights each batch but preserve canonical moments/counter, then
deep-copy that common state into all arms. Advance only the canonical
current-connected state to the next batch. Cache after reset, two scalars
allocated per arm, expired after batch. Total160 updates, not200.

Follow the already registered numerical tolerance1e-10*max(1,comparison
scale), sign/Jacobian decomposition at the connected pair's common second
state, immutable cache checks and full state/moment evidence. A mismatch
in first updates invalidates the shared-state analysis; never hide it by
averaging states. Log exact/near-zero residuals and reference preactivations,
all masks and input Jacobians; do not exclude event-free or boundary cases.

The original nontrivial-effect gate is unchanged: BOTH second-gradient
difference >1e-4*max(||g_current||,||g_cached||,1e-8) AND post-update student
output RMS difference >1e-6*max(RMS(q0),1) in >=5/20 batches. Save all raw
magnitudes including zeros, cancellation contributions and pass/fail
decisions. A correct harness with no/insufficient effect rejects the effect
for this declared setting; it is not a cue to extend the horizon or add
corruption. A numerical mismatch is unresolved/invalid, not positive or
negative evidence. No risk, forgetting, recovery or novelty claim follows
from this unlabeled diagnostic alone.
