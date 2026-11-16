"""Exact squared errors on the constructed two-feature calibration example."""
from fractions import Fraction as F
import json
import time

def predictions(u):
    return (1 + u, 2 * u)

def risk(update, truth):
    errors = [a - b for a, b in zip(predictions(update), predictions(truth))]
    return sum(e * e for e in errors) / 2

start = time.process_time()
updates = {"baseline_or_reset": 0, "projection": 1, "interval_center": 2,
           "protected_wrong_direction": -1}
endpoints = (F(1), F(3))
result = {}
for name, update in updates.items():
    differences = [risk(update, u) - risk(0, u) for u in endpoints]
    exact = [F(5, 2) * (update ** 2 - 2 * update * u) for u in endpoints]
    assert differences == exact
    result[name] = {"at_lower": str(exact[0]), "at_upper": str(exact[1]),
                    "maximum": str(max(exact))}
assert [result[name]["maximum"] for name in updates] == ["0", "-5/2", "0", "35/2"]
elapsed = time.process_time() - start
assert elapsed < 10
print(json.dumps({"endpoints": result, "cpu_seconds": elapsed}, indent=2))

