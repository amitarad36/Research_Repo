"""Score locked predictions with scalar-sum and NumPy forward verification."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import hashlib
import math
import time
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
RESEARCH = Path("C:\\Users\\amita\\source\\repos\\Detach Proposal Presentation\\research\\notes")
COHORT = RESEARCH / "2026-11-07-cycle30"
DATA = RESEARCH / "2026-10-31-cycle17/bostonHousing.txt"
ARMS = ["no_adaptation", "current-connected", "cached-connected",
        "current-stopped", "cached-stopped"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def forward(flat_weights, x):
    shapes = [(30,14), (30,), (20,30), (20,), (10,20), (10,), (1,10), (1,)]
    arrays = []
    offset = 0
    for shape in shapes:
        size = math.prod(shape)
        arrays.append(flat_weights[offset:offset+size].reshape(shape))
        offset += size
    assert offset == 1291
    h = np.concatenate([x, np.full((len(x),1), 50.)], axis=1)
    for layer in range(4):
        h = h @ arrays[2*layer].T + arrays[2*layer+1]
        if layer < 3:
            h = np.maximum(h, 0)
    return h

start = time.process_time()
assert sha(D/"score.py") == (D/"score.sha256").read_text().strip()
locked = json.loads((D/"locked_predictions.json").read_text())
for name, digest in locked["files"].items():
    assert sha(D/name) == digest
assert sha(DATA) == "baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a"
raw = [line for line in DATA.read_bytes().splitlines() if line.strip()]
selected = np.array([[float(token) for token in line.split()] for line in raw[455:495]])
results = []
for seed in (1,2,3):
    saved = json.loads((D/f"seed_{seed}/run/results.json").read_text())
    assert saved["status"] == "complete" and saved["updates"] == 160
    trace = np.load(D/f"seed_{seed}/run/trace.npz", allow_pickle=False)
    prep = np.load(COHORT/f"seed_{seed}/preprocessing.npz", allow_pickle=False)
    target = ((selected[:,13:] - prep["y_mean"]) / prep["y_std"]).reshape(40)
    x = (selected[:,:13] - prep["x_mean"]) / prep["x_std"]
    final = trace["final_output"].reshape(20,4,2,1)
    predictions = np.concatenate([trace["noadapt"].reshape(1,40),
        np.stack([final[:,i].reshape(40) for i in range(4)])])
    assert predictions.shape == (5,40) and np.isfinite(predictions).all()
    assert np.isfinite(target).all()
    max_forward_error = 0.
    for batch in range(20):
        states = trace["before_theta"][batch*8:batch*8+8]
        for arm in range(4):
            assert np.array_equal(states[0], states[2*arm])
            after_index = batch*8+arm*2+1
            independent = forward(trace["after_theta"][after_index], x[2*batch:2*batch+2])
            err = float(np.max(np.abs(independent-final[batch,arm])))
            max_forward_error = max(max_forward_error, err)
            assert err <= 1e-10 * max(1., float(np.max(np.abs(independent))))
        baseline = forward(states[0], x[2*batch:2*batch+2])
        assert np.allclose(baseline.reshape(2), predictions[0,2*batch:2*batch+2], atol=1e-10, rtol=1e-10)
        # Same optimizer history at the start of every fork.
        for field in ("before_m", "before_v", "before_step"):
            for arm in (1,2,3):
                assert np.array_equal(trace[field][batch*8], trace[field][batch*8+2*arm])
    errors = (predictions-target[None,:])**2
    means = errors.mean(1)
    independent_means = [math.fsum((float(pred)-float(y))**2 for pred,y in zip(row,target))/40
                         for row in predictions]
    assert np.allclose(means,independent_means,atol=1e-10,rtol=1e-10)
    deltas = means[1]-means[[0,2]]
    paired = (errors[1]-errors[[0,2]]).mean(1)
    assert np.allclose(deltas,paired,atol=1e-10,rtol=1e-10)
    unchanged = [b for b in range(20) if np.array_equal(final[b,0],final[b,1])]
    for b in unchanged:
        assert np.array_equal(errors[1,2*b:2*b+2], errors[2,2*b:2*b+2])
    results.append(dict(seed=seed,event_batches=saved["count"],mechanism_pass=saved["count"]>=5,
        mse=dict(zip(ARMS,means.tolist())),current_minus_baseline=float(deltas[0]),
        current_minus_cached=float(deltas[1]),
        percent_vs_baseline=100*float(deltas[0]/means[0]),
        percent_vs_cached=100*float(deltas[1]/means[2]),
        joint_benefit=bool(np.all(deltas<0)),
        predictions=predictions.tolist(),targets=target.tolist(),squared_errors=errors.tolist(),
        independent_mse=independent_means,max_forward_error=max_forward_error,
        identical_connected_batches=unchanged,
        max_formula_error=float(trace["formula_err"].max()),
        max_decomposition_error=max(b["decomposition_error"] for b in saved["batches"]),
        residual_near_zero_count=int(trace["r_near"].sum()),
        preactivation_near_zero_count=int(trace["pre_near"].sum())))
    assert time.process_time()-start < 10
cohort_mean = {arm:math.fsum(r["mse"][arm] for r in results)/3 for arm in ARMS}
report = dict(status="complete",cohort=results,mean_mse=cohort_mean,
    mean_current_minus_baseline=cohort_mean["current-connected"]-cohort_mean["no_adaptation"],
    mean_current_minus_cached=cohort_mean["current-connected"]-cohort_mean["cached-connected"],
    harm_sign_persists=all(r["current_minus_baseline"]>0 and r["current_minus_cached"]>0 for r in results),
    cpu_seconds=time.process_time()-start,script_sha256=sha(D/"score.py"))
assert report["cpu_seconds"]<10
with (D/"risk_results.json").open("x",encoding="utf-8") as handle:
    json.dump(report,handle,indent=2,allow_nan=False)
print(json.dumps({**report,"cohort":[{k:v for k,v in r.items() if k not in ("predictions","targets","squared_errors")} for r in results]},indent=2))
