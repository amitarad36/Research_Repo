"""Exact counterexample to output-only second-gradient identification."""
from fractions import Fraction as F
import json

def relu(z):
    return max(F(0), z)

def reference(z, displacement, coefficient):
    first = z - displacement / 3
    second = z - 2 * displacement / 3
    value = 1 + coefficient * relu(first) - 2 * coefficient * relu(second)
    slope = coefficient * int(first > 0) - 2 * coefficient * int(second > 0)
    return value, slope

def sign(z):
    return int(z > 0) - int(z < 0)

rows = []
eta, epsilon = F(1, 10000), F(1, 10**8)
displacements = [F(1, 8), F(1, 10000), F(1, 10**9), eta / (1 + epsilon)]
for delta in displacements:
    signatures = []
    for coefficient in (F(0), F(1), F(-2)):
        cached_value, cached_slope = reference(F(0), delta, coefficient)
        fresh_value, fresh_slope = reference(delta, delta, coefficient)
        signatures.append((cached_value, cached_slope, delta, fresh_value))
        cached_field = (1 - cached_slope) * sign(cached_value - delta)
        fresh_field = (1 - fresh_slope) * sign(fresh_value - delta)
        assert cached_field == 1
        assert fresh_field == 1 + coefficient
        # Independent exact finite difference at both differentiable endpoints.
        h = delta / 100
        for point, derivative in ((F(0), cached_slope), (delta, fresh_slope)):
            difference = (reference(point+h, delta, coefficient)[0] -
                          reference(point-h, delta, coefficient)[0]) / (2*h)
            assert difference == derivative
        # Local affine prediction fails when coefficient is nonzero.
        rows.append(dict(delta=str(delta), coefficient=str(coefficient),
                         cached_field=str(cached_field), fresh_field=str(fresh_field)))
    assert len(set(signatures)) == 1
print(json.dumps(dict(cases=len(rows), endpoint_derivative_checks=2*len(rows),
                     all_assertions_passed=True, results=rows), indent=2))
