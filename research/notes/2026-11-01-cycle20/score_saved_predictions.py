"""Score cycle-19 saved predictions under the previously registered protocol."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
NOTES = DIRECTORY.parent
PREVIOUS = NOTES / "2026-11-01-cycle19"
TRAINING = NOTES / "2026-10-31-cycle18"
DATA = NOTES / "2026-10-31-cycle17" / "bostonHousing.txt"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    assert digest(Path(__file__)) == (DIRECTORY / "registered.sha256").read_text().strip().lower()
    assert digest(PREVIOUS / "NEXT_RISK_PROTOCOL.md") == "ef7c15589c779c4e95e9feec69c124bb537b59bff849572925aa7728f8bfdbda"
    for name, expected in load_json(PREVIOUS / "evidence_manifest.json").items():
        assert digest(PREVIOUS / name) == expected
    assert digest(TRAINING / "preprocessing.npz") == "7815dfb04a28e8c2929142e8dc95442fadff37c6501086b67b7475fd9685ec42"
    assert digest(DATA) == "baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a"
    metadata = load_json(PREVIOUS / "run" / "results.json")
    assert metadata["status"] == "complete" and metadata["updates"] == 160
    arm_names = ["no_adaptation"] + metadata["arm_order"]
    assert arm_names[1:] == ["current-connected", "cached-connected", "current-stopped", "cached-stopped"]
    with np.load(PREVIOUS / "run" / "trace.npz", allow_pickle=False) as trace:
        assert trace["final_output"].shape == (80, 2, 1)
        assert trace["noadapt"].shape == (20, 2, 1)
        adapted = trace["final_output"].reshape(20, 4, 2).transpose(1, 0, 2).reshape(4, 40)
        predictions = np.vstack((trace["noadapt"].reshape(1, 40), adapted))
    with np.load(TRAINING / "preprocessing.npz", allow_pickle=False) as scaling:
        target_mean = float(scaling["y_mean"].item())
        target_std = float(scaling["y_std"].item())
    raw_lines = [line for line in DATA.read_bytes().splitlines() if line.strip()]
    assert len(raw_lines) == 506 and target_std > 0
    output = DIRECTORY / "run"
    output.mkdir()  # Refuse to overwrite a previous scoring run.
    started = time.process_time()
    report = {"status": "running", "new_model_calls": 0, "new_optimizer_steps": 0,
              "script_sha256": digest(Path(__file__)), "arm_order": arm_names}
    try:
        targets = np.array([float(line.split()[13]) for line in raw_lines[455:495]])
        normalized_targets = (targets - target_mean) / target_std
        assert targets.shape == (40,)
        assert np.isfinite(predictions).all() and np.isfinite(normalized_targets).all()
        squared_errors = (predictions - normalized_targets[None, :]) ** 2
        mse = squared_errors.mean(axis=1)
        paired = squared_errors[:, None, :] - squared_errors[None, :, :]
        paired_means = paired.mean(axis=2)
        difference_of_means = mse[:, None] - mse[None, :]
        errors = np.abs(paired_means - difference_of_means)
        tolerance = 1e-10 * max(1., np.abs(paired_means).max(), np.abs(difference_of_means).max())
        assert errors.max() <= tolerance
        identical_batches = np.all(adapted[0].reshape(20, 2) == adapted[1].reshape(20, 2), axis=1)
        assert identical_batches.sum() == 15
        assert np.all(paired[1, 2].reshape(20, 2)[identical_batches] == 0)
        benefit = bool(mse[1] < mse[0] and mse[1] < mse[2])
        report.update(
            status="complete",
            evaluated_rows=list(range(455, 495)),
            mse_normalized=dict(zip(arm_names, mse.tolist())),
            mse_original_units=dict(zip(arm_names, (mse * target_std ** 2).tolist())),
            paired_difference_means=paired_means.tolist(),
            maximum_mean_identity_error=float(errors.max()),
            identical_batch_count=int(identical_batches.sum()),
            current_vs_noadapt_relative_change=float(mse[1] / mse[0] - 1),
            current_vs_cached_relative_change=float(mse[1] / mse[2] - 1),
            joint_benefit_gate=benefit,
            decision="passes_this_setting" if benefit else "reject_joint_benefit_this_setting",
            per_example_squared_errors=squared_errors.tolist(),
            per_example_predictions=predictions.tolist(),
            normalized_targets=normalized_targets.tolist(),
            target_mean=target_mean,
            target_std=target_std,
            numpy_version=np.__version__,
        )
    except Exception as error:
        report.update(status="invalid", error=f"{type(error).__name__}: {error}")
    report["cpu_seconds_after_loading"] = time.process_time() - started
    if report["cpu_seconds_after_loading"] >= 10:
        report.update(status="invalid", error="Registered CPU limit exceeded")
    with (output / "risk_results.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ["per_example_squared_errors", "per_example_predictions", "normalized_targets", "evaluated_rows", "paired_difference_means"]}, indent=2))
    assert report["status"] == "complete"


if __name__ == "__main__":
    main()
