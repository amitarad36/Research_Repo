"""Post-hoc stepwise error accounting from saved predictions only."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3] / "research/notes"
start = time.process_time()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(folder):
    manifest = json.loads((folder/"evidence_manifest.json").read_text())
    for name, digest in manifest.items():
        assert sha(folder/name) == digest, str(folder/name)
    return sha(folder/"evidence_manifest.json")
sources = {name:verify(ROOT/name) for name in
           ("2026-11-01-cycle19", "2026-11-01-cycle20", "2026-11-07-cycle31")}
cohort = json.loads((ROOT/"2026-11-07-cycle31/risk_results.json").read_text())
original = json.loads((ROOT/"2026-11-01-cycle20/run/risk_results.json").read_text())
arms = ["current-connected", "cached-connected", "current-stopped", "cached-stopped"]
results = []
max_identity_error = max_scalar_error = 0.
for seed in (0,1,2,3):
    path = (ROOT/"2026-11-01-cycle19/run/trace.npz" if seed == 0 else
            ROOT/f"2026-11-07-cycle31/seed_{seed}/run/trace.npz")
    with np.load(path, allow_pickle=False) as trace:
        reset = trace["noadapt"].reshape(40)
        before = trace["q"].reshape(20,4,2,2,1)
        first = before[:,:,1].transpose(1,0,2,3).reshape(4,40)
        final = trace["final_output"].reshape(20,4,2,1).transpose(1,0,2,3).reshape(4,40)
        for arm in range(4):
            assert np.array_equal(before[:,arm,0].reshape(40),reset)
        assert np.array_equal(first[0],first[1])
        assert np.array_equal(first[2],first[3])
    saved = original if seed == 0 else next(r for r in cohort["cohort"] if r["seed"]==seed)
    target = np.asarray(saved["normalized_targets"] if seed == 0 else saved["targets"]).reshape(40)
    previous = saved["mse_normalized"] if seed == 0 else saved["mse"]
    reset_error = (reset-target)**2
    first_error = (first-target)**2
    final_error = (final-target)**2
    assert np.isfinite(first_error).all() and np.isfinite(final_error).all()
    assert abs(float(reset_error.mean())-previous["no_adaptation"]) < 1e-12
    records = []
    for a,name in enumerate(arms):
        d1 = first[a]-reset
        d2 = final[a]-first[a]
        change1 = first_error[a]-reset_error
        change2 = final_error[a]-first_error[a]
        total = final_error[a]-reset_error
        # Independent finite-step squared-error identity and telescoping.
        for actual,formula in ((change1,2*(reset-target)*d1+d1*d1),
                               (change2,2*(first[a]-target)*d2+d2*d2),
                               (total,change1+change2)):
            error = float(np.max(np.abs(actual-formula)))
            max_identity_error = max(max_identity_error,error)
            assert error < 1e-12
        mse1 = float(first_error[a].mean())
        mse2 = float(final_error[a].mean())
        assert abs(mse2-previous[name]) < 1e-12
        scalar1 = math.fsum((float(q)-float(y))**2 for q,y in zip(first[a],target))/40
        scalar2 = math.fsum((float(q)-float(y))**2 for q,y in zip(final[a],target))/40
        max_scalar_error = max(max_scalar_error,abs(mse1-scalar1),abs(mse2-scalar2))
        assert abs(mse1-scalar1)<1e-12 and abs(mse2-scalar2)<1e-12
        records.append(dict(arm=name,mse_after_first=mse1,mse_after_second=mse2,
            first_change=float(change1.mean()),second_change=float(change2.mean()),
            total_change=float(total.mean()),
            first_improved_examples=int((change1<0).sum()),
            second_improved_examples=int((change2<0).sum()),
            first_improved_batches=int((change1.reshape(20,2).mean(1)<0).sum()),
            second_improved_batches=int((change2.reshape(20,2).mean(1)<0).sum()),
            per_example_first_prediction=first[a].tolist(),
            per_example_first_change=change1.tolist(),per_example_second_change=change2.tolist()))
    results.append(dict(seed=seed,baseline_mse=float(reset_error.mean()),arms=records,
                        trace_sha256=sha(path)))
    assert time.process_time()-start < 10
report=dict(status="complete",results=results,source_manifests=sources,
            max_identity_error=max_identity_error,max_scalar_mean_error=max_scalar_error,
            new_model_calls=0,new_training_steps=0,additional_targets=0,
            cpu_seconds=time.process_time()-start)
print(json.dumps(report,separators=(",",":")))
