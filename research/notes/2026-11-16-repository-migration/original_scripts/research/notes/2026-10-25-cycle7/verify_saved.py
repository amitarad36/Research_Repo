"""Independent saved-state/metric audit. Does not run adaptation updates."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / '2026-10-22-cycle3'
if not SRC.exists():
    SRC = Path(r'C:\Users\amita\source\repos\Detach Proposal Presentation\research\notes\2026-10-22-cycle3')
spec = importlib.util.spec_from_file_location('cycle3', SRC / 'neural_geometry_gate.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pred(p, x): return model.forward(p, x, np.zeros_like(x))[0]

report = json.loads((ROOT / 'frozen_input_results.json').read_text(encoding='utf-8'))
assert report['status'] == 'complete' and report['completed_updates'] == 90000
assert report['script_sha256'] == sha(ROOT / 'frozen_input_control.py')
assert report['protocol_sha256'] == sha(ROOT / 'PROTOCOL.md')
data = np.load(SRC / 'shared_data.npz')
x, xe, ye = data['x_eval'][:64], data['x_eval'][64:], data['y_eval'][64:]
variance = np.mean(np.sum((ye-ye.mean(0))**2, axis=1))
max_metric_abs = 0.
max_inactive = 0.
table = []
record_count = 0
allruns = []
for s in report['seeds']:
    p = np.load(SRC / f"seed_{s['seed']}.npz")['parameters']
    check = s['initial_checks']
    assert sha(ROOT / check['archive']) == check['archive_sha256']
    d = np.load(ROOT / check['archive'])
    assert np.max(np.abs(d['stop_student']-d['cached_student']-d['expected_student_difference'])) <= 1e-7
    assert np.max(np.abs(d['stop_teacher']-d['cached_teacher'])) <= 1e-7
    cache0 = pred(p, x)
    for run in s['runs']:
        allruns.append(run)
        assert sha(ROOT / run['archive']) == run['archive_sha256']
        arrays = np.load(ROOT / run['archive'])
        for prefix, rr in [('', run), ('fine_', run['halved_step'])]:
            states, targets, cache = arrays[prefix+'state'], arrays[prefix+'target'], arrays[prefix+'cached']
            assert states.shape == (51,2,114) and targets.shape == (51,64,2)
            assert cache.nbytes == 1024 and np.array_equal(cache, cache0)
            assert np.array_equal(states[0,0], p) and np.array_equal(states[0,1], p)
            max_inactive = max(max_inactive, float(np.max(np.abs(states[:,:,32:64]-p[32:64]))))
            if run['tau'] is None:
                assert np.array_equal(states[:,1], np.broadcast_to(p, (51,114)))
            for state, saved_target, h in zip(states, targets, rr['history']):
                student, teacher = state
                q = pred(student, x)
                target = model.forward(teacher, x, cache if run['rule'] == 'cached' else q)[0]
                assert np.array_equal(target, saved_target)
                metrics = {
                    'nmse': float(np.mean(np.sum((pred(student, xe)-ye)**2, axis=1))/variance),
                    'discrepancy': float(.5*np.mean(np.sum((target-q)**2, axis=1))),
                    'reference_lag': float(np.linalg.norm(student-teacher)),
                    'reference_target_change_rms': float(np.sqrt(np.mean(np.sum((target-targets[0])**2, axis=1)))),
                    'reference_target_change_max_abs': float(np.max(np.abs(target-targets[0]))),
                }
                for k, val in metrics.items():
                    max_metric_abs = max(max_metric_abs, abs(val-h[k]))
                record_count += 1
        if run['tau'] == 1:
            table.append({'seed': s['seed'], 'primary': s['primary'], 'rule': run['rule'],
                          'initial_nmse': s['no_adaptation_nmse'],
                          'main_nmse': run['final']['nmse'], 'fine_nmse': run['halved_step']['final']['nmse'],
                          'initial_discrepancy': run['initial']['discrepancy'],
                          'final_discrepancy': run['final']['discrepancy'],
                          'target_change_rms': run['final']['reference_target_change_rms'],
                          'reference_lag': run['final']['reference_lag']})
assert record_count == 3060 and max_metric_abs < 1e-12 and max_inactive == 0.
assert len(list(ROOT.glob('seed*_initial_checks.npz'))) == 5
assert len(list(ROOT.glob('seed*_fixed.npz'))) == 15
assert len(list(ROOT.glob('seed*_tau1.npz'))) == 15
summary = {
    'verifier_sha256': sha(Path(__file__)), 'results_sha256': sha(ROOT / 'frozen_input_results.json'),
    'records_recomputed': record_count, 'max_metric_absolute_difference': max_metric_abs,
    'max_fixed_coordinate_error': max_inactive, 'parameter_target_archives': 30, 'derivative_archives': 5,
    'max_gradient_error': max(s['initial_checks']['control_loss_gradient_max_abs'] for s in report['seeds']),
    'max_student_identity_error': max(s['initial_checks']['student_block_identity_max_abs'] for s in report['seeds']),
    'max_baseline_endpoint_abs_difference': max(max(r.get('baseline_absolute_differences', [0])) for r in allruns),
    'max_primary_step_difference': max(r['final_nmse_step_difference'] for s in report['seeds'] if s['primary'] for r in s['runs']),
    'max_all_step_difference': max(r['final_nmse_step_difference'] for r in allruns),
    'max_fixed_cached_target_difference': max(r.get('fixed_target_max_abs', 0) for r in allruns),
    'max_original_fixed_target_rms_change': max(h['reference_target_change_rms'] for r in allruns if r['tau'] is None and r['rule'] != 'cached' for h in r['history']),
    'gates': report['gates'], 'ema_table': table,
    'cpu_seconds': report['cpu_seconds'], 'wall_seconds': report['wall_seconds'],
}
(ROOT / 'verification_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
