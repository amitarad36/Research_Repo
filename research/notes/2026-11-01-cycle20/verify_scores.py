import hashlib
import json
import math
from pathlib import Path

directory = Path(__file__).resolve().parent
results_path = directory / "run" / "risk_results.json"
report = json.loads(results_path.read_text())
assert report["status"] == "complete"
verified = {}
for name, predictions, stored_errors in zip(report["arm_order"], report["per_example_predictions"], report["per_example_squared_errors"]):
    errors = [(prediction - target) ** 2 for prediction, target in zip(predictions, report["normalized_targets"])]
    assert errors == stored_errors
    mean_error = math.fsum(errors) / 40
    assert abs(mean_error - report["mse_normalized"][name]) <= 1e-14
    verified[name] = mean_error
assert verified["current-connected"] > verified["no_adaptation"]
assert verified["current-connected"] > verified["cached-connected"]
assert all(verified[name] > verified["no_adaptation"] for name in report["arm_order"][1:])
result = {
    "status": "passed",
    "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    "method": "Python scalar squares and math.fsum on all saved prediction/target pairs",
    "verified_mse": verified,
    "joint_benefit_gate": False,
    "all_four_arms_worse_than_baseline": True,
}
with (directory / "verification_results.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
