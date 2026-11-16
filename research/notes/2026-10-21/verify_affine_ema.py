"""Deterministic checks of the affine EMA diagnostic model; standard library only.

Run: python verify_affine_ema.py
This checks closed-form identities, not a general neural-network theorem.
"""
import json
import math
from pathlib import Path


def state(t, beta, k, tau, u0):
    total = beta * k + 1.0 / tau
    delta = math.sqrt((beta * k - 1.0 / tau) ** 2 + 4.0 * k / tau)
    fast = (total + delta) / 2.0
    slow = k * (beta - 1.0) / tau / fast
    weight = (fast - k * (beta - 1.0)) / (fast - slow)
    es, ef = math.exp(-slow * t), math.exp(-fast * t)
    u = u0 * (weight * es + (1.0 - weight) * ef)
    du = -u0 * (weight * slow * es + (1.0 - weight) * fast * ef)
    ddu = u0 * (weight * slow**2 * es + (1.0 - weight) * fast**2 * ef)
    residual = du / k
    v = residual + beta * u
    dv = ddu / k + beta * du
    return u, v, du, dv, residual, slow, fast


results = []
max_ode_error = 0.0
for beta in (1.01, 1.2, 2.0, 5.0):
    for k in (1.0, beta):
        for tau in (0.01, 0.1, 1.0, 10.0, 100.0):
            u0, c = 0.6, 0.6
            *_, slow, fast = state(0, beta, k, tau, u0)
            times = [0.0] + [10.0 ** (-8 + i * 12 / 500) / slow for i in range(501)]
            previous = None
            for t in times:
                u, v, du, dv, residual, _, _ = state(t, beta, k, tau, u0)
                error = max(abs(du - k * (v - beta * u)), abs(dv - (u - v) / tau))
                max_ode_error = max(max_ode_error, error)
                assert error < 1e-9
                discrepancy = residual**2 / 2
                new_risk = u**2 / 2
                old_risk = c**2 * (u0 - u)**2 / 2
                if previous:
                    assert discrepancy <= previous[0] + 1e-11
                    assert new_risk <= previous[1] + 1e-11
                    assert old_risk >= previous[2] - 1e-11
                previous = discrepancy, new_risk, old_risk
            results.append({"beta": beta, "k": k, "tau": tau, "slow_rate": slow,
                            "old_risk_limit": c**2 * u0**2 / 2,
                            "last_old_risk": previous[2]})

summary = {"model_combinations": len(results), "times_per_combination": len(times),
           "max_closed_form_ode_error": max_ode_error, "all_checks_passed": True,
           "scope": "Closed-form deterministic affine ODE checks; no discrete or neural guarantee.",
           "results": results}
output = Path(__file__).with_name("affine_ema_verification.json")
output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
