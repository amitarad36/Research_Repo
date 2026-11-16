"""Data-free conformance and marginal cost of a ReLU region diagnostic."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
START = time.process_time()
TOL = 1e-10
MARGIN = 1e-12
BATCH = 16

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(14, 30)
        self.hidden1 = nn.Linear(30, 20)
        self.hidden2 = nn.Linear(20, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x, z=None):
        if z is None:
            z = x.new_full((len(x), 1), 50.)
        h = torch.relu(self.input(torch.cat((x, z), dim=1)))
        h = torch.relu(self.hidden1(h))
        h = torch.relu(self.hidden2(h))
        return self.output(h)

@torch.no_grad()
def cache_forward(model, x, c):
    # First-layer x contribution is reusable by both compared methods.
    base = model.input(torch.cat((x, torch.zeros_like(c)), dim=1))
    u1 = base + c * model.input.weight[:, -1]
    u2 = model.hidden1(torch.relu(u1))
    u3 = model.hidden2(torch.relu(u2))
    return base, [u1, u2, u3], model.output(torch.relu(u3))

@torch.no_grad()
def region_setup(model, units):
    slopes = [model.input.weight[:, -1].expand(len(units[0]), -1)]
    for layer, u in zip((model.hidden1, model.hidden2), units[:2]):
        slopes.append((slopes[-1] * (u > 0)) @ layer.weight.T)
    a = (slopes[-1] * (units[-1] > 0)) @ model.output.weight.T
    u = torch.cat(units, dim=1)
    v = torch.cat(slopes, dim=1)
    # Distances to the first positive and negative wall.
    roots = -u / torch.where(v != 0, v, torch.ones_like(v))
    infinity = torch.full_like(roots, float("inf"))
    right = torch.where((v != 0) & (roots > 0), roots, infinity).amin(1, keepdim=True)
    left = torch.where((v != 0) & (roots < 0), -roots, infinity).amin(1, keepdim=True)
    valid = (u.abs() > MARGIN).all(1, keepdim=True)
    return a, left, right, valid

@torch.no_grad()
def predict(summary, cached_output, c, q):
    a, left, right, valid = summary
    delta = q - c
    radius = torch.where(delta >= 0, right, left)
    inside = valid & (delta.abs() < radius - MARGIN * (1 + radius.abs()))
    # Handle infinite radii separately to avoid inf-inf.
    inside |= valid & torch.isinf(radius)
    inside |= delta == 0
    cached_residual = cached_output - q
    fresh_residual = cached_output + a * delta - q
    accepted = inside & (cached_residual.abs() > MARGIN) & (fresh_residual.abs() > MARGIN)
    current = (1-a) * torch.sign(fresh_residual)
    cached = (1-a) * torch.sign(cached_residual)
    return accepted, current, cached

def direct_reference(model, base, z):
    u1 = base + z * model.input.weight[:, -1]
    u2 = model.hidden1(torch.relu(u1))
    return model.output(torch.relu(model.hidden2(torch.relu(u2))))

def direct_coefficient(model, base, q):
    z = q.detach().clone().requires_grad_(True)
    value = direct_reference(model, base, z)
    slope = torch.autograd.grad(value.sum(), z)[0]
    return (1-slope) * torch.sign(value.detach()-q), slope

def flat(values):
    return torch.cat([value.reshape(-1) for value in values])

report = {"seeds": list(range(4)), "batch_size": BATCH, "dtype": "float64",
          "torch_version": torch.__version__, "training_steps": 0, "labels_used": False,
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "tolerance": TOL, "margin": MARGIN, "case_groups": {}}
max_coefficient_error = max_parameter_error = 0.
accepted_count = abstained_count = equal_count = different_count = 0
for seed in report["seeds"]:
    torch.manual_seed(seed)
    student = Model()
    reference = copy.deepcopy(student).requires_grad_(False)
    assert sum(p.numel() for p in student.parameters()) == 1291
    x = torch.randn(BATCH, 13)
    c = student(x).detach()
    base, units, output = cache_forward(reference, x, c)
    summary = region_setup(reference, units)
    a, left, right, _ = summary
    cases = [(str(d), torch.full_like(c, d)) for d in (0., -1e-4, 1e-4, -.01, .01, -1., 1.)]
    for direction, radius in ((-1, left), (1, right)):
        for factor in (.5, 1., 1.01):
            # Infinite wall distances cannot define this probe; mark and exclude.
            delta = direction * factor * radius
            cases.append((f"wall_{direction}_{factor}", delta))
    for name, delta in cases:
        finite = torch.isfinite(delta)
        safe_delta = torch.where(finite, delta, torch.zeros_like(delta))
        q = c + safe_delta
        accepted, predicted, cached_predicted = predict(summary, output, c, q)
        accepted &= finite
        selected = accepted[:, 0]
        direct, actual_slope = direct_coefficient(reference, base, q)
        n = int(accepted.sum())
        m = int(finite.sum()) - n
        accepted_count += n
        abstained_count += m
        item = report["case_groups"].setdefault(name, {"accepted": 0, "abstained": 0, "excluded": 0})
        item["accepted"] += n
        item["abstained"] += m
        item["excluded"] += BATCH-int(finite.sum())
        if n:
            err = float((direct[accepted]-predicted[accepted]).abs().max())
            max_coefficient_error = max(max_coefficient_error, err)
            assert err <= TOL, (seed, name, err)
            assert torch.allclose(actual_slope[accepted], a[accepted], atol=TOL, rtol=0)
            same = int((predicted[accepted] == cached_predicted[accepted]).sum())
            equal_count += same
            different_count += n-same
            # Full parameter gradients of shifted student predictions, fresh and cached.
            for cached_mode, coefficient in ((False, predicted), (True, cached_predicted)):
                student_q = student(x) + safe_delta
                z = c + (student_q-student_q.detach()) if cached_mode else student_q
                target = reference(x, z)
                loss = ((target-student_q).abs() * accepted).sum()/BATCH
                actual_gradient = flat(torch.autograd.grad(loss, tuple(student.parameters())))
                check_q = student(x) + safe_delta
                expected_gradient = -flat(torch.autograd.grad(
                    check_q, tuple(student.parameters()),
                    grad_outputs=coefficient * accepted / BATCH))
                err = float((actual_gradient-expected_gradient).abs().max())
                max_parameter_error = max(max_parameter_error, err)
                assert err <= TOL, (seed, name, cached_mode, err)
        assert time.process_time()-START < 60
    if seed == 0:
        timing_fixture = (reference, base, units, output, c)
# A unit exactly on its cache boundary forces abstention for a nonzero step.
modified = [u.clone() for u in units]
modified[0][:, 0] = 0
bad_summary = region_setup(reference, modified)
assert not predict(bad_summary, output, c, c+1e-4)[0].any()
report["zero_preactivation_abstention_passed"] = True

reference, base, units, output, c = timing_fixture
q = c + 1e-4
def region_route():
    return predict(region_setup(reference, units), output, c, q)

def direct_route():
    return direct_coefficient(reference, base, q)

# The common cache forward and student work are excluded from both routes.
for _ in range(10):
    region_route()
    direct_route()
measurements = {"region_setup_and_query": [], "fresh_forward_and_input_backward": []}
for block in range(5):
    routes = [("region_setup_and_query", region_route),
              ("fresh_forward_and_input_backward", direct_route)]
    if block % 2:
        routes.reverse()
    for name, fn in routes:
        begin = time.perf_counter()
        for _ in range(100):
            fn()
        measurements[name].append((time.perf_counter()-begin)/100)
        assert time.process_time()-START < 60
report.update(accepted=accepted_count, abstained=abstained_count,
              accepted_equal=equal_count, accepted_different=different_count,
              max_coefficient_abs_error=max_coefficient_error,
              max_parameter_gradient_abs_error=max_parameter_error,
              timing_seconds_per_call=measurements,
              median_seconds_per_call={k: statistics.median(v) for k,v in measurements.items()},
              cpu_seconds=time.process_time()-START,
              status="passed_numerical_conformance")
assert report["cpu_seconds"] < 60
print(json.dumps(report, indent=2))
