"""Cycle 7: run the unchanged cycle-6 frozen-input protocol; NumPy/CPU only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
NOTES = (Path(__file__).resolve().parents[3] / 'research/notes')
SRC = ROOT.parent / '2026-10-22-cycle3'
if not SRC.exists():
    SRC = NOTES / '2026-10-22-cycle3'
BASE = SRC.parent / '2026-10-22-cycle4' / 'finite_response_results.json'
PROTOCOL_SHA = '86eca22e2ec0a9c4a1e8ef85b46ff7edb136a77593dce321733e6c60966a0a86'
BASE_SHA = '463e17630e992d318abbd26a8376b79cd62b55b85fcafa25138c1b5c57066c89'
spec = importlib.util.spec_from_file_location('model_cycle3', SRC / 'neural_geometry_gate.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
INACTIVE = np.arange(32, 64)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pred(p, x):
    return model.forward(p, x, np.zeros_like(x))[0]


def field(s, t, x, cached, rule):
    if rule != 'cached':
        return model.velocity(s, t, x, rule)
    q, cq = model.forward(s, x, np.zeros_like(x))
    target = model.forward(t, x, cached)[0]
    return model.vjp(cq, target - q)


def central_jacobian(fun, p, h=1e-5):
    columns = []
    for j in range(len(p)):
        d = np.zeros_like(p)
        d[j] = h
        columns.append((fun(p + d) - fun(p - d)) / (2*h))
    return np.stack(columns, axis=-1)


def initial_checks(p, x, cached, seed):
    target, ct = model.forward(p, x, cached)
    q = pred(p, x)
    def loss(s):
        return .5*np.mean(np.sum((target - pred(s, x))**2, axis=1))
    loss_grad = central_jacobian(loss, p)
    sf = central_jacobian(lambda s: field(s, p, x, cached, 'stop'), p)
    sc = central_jacobian(lambda s: field(s, p, x, cached, 'cached'), p)
    tf = central_jacobian(lambda t: field(p, t, x, cached, 'stop'), p)
    tc = central_jacobian(lambda t: field(p, t, x, cached, 'cached'), p)
    j = model.parameter_jacobian(p, x, np.zeros_like(x))
    a = model.jac_z(p, ct)
    explicit = sum(ji.T @ ai @ ji for ji, ai in zip(j, a)) / len(x)
    checks = {
        'control_loss_gradient_max_abs': float(np.max(np.abs(loss_grad + field(p, p, x, cached, 'cached')))),
        'initial_target_max_abs': float(np.max(np.abs(target - model.forward(p, x, q)[0]))),
        'initial_velocity_max_abs': float(np.max(np.abs(field(p, p, x, cached, 'stop') - field(p, p, x, cached, 'cached')))),
        'student_block_identity_max_abs': float(np.max(np.abs(sf - sc - explicit))),
        'teacher_block_identity_max_abs': float(np.max(np.abs(tf - tc))),
        'fd_step': 1e-5,
    }
    checks['pass'] = bool(checks['control_loss_gradient_max_abs'] <= 1e-7
                         and checks['initial_target_max_abs'] <= 1e-12
                         and checks['initial_velocity_max_abs'] <= 1e-12
                         and checks['student_block_identity_max_abs'] <= 1e-7
                         and checks['teacher_block_identity_max_abs'] <= 1e-7)
    out = ROOT / f'seed{seed}_initial_checks.npz'
    np.savez_compressed(out, stop_student=sf, cached_student=sc,
                        stop_teacher=tf, cached_teacher=tc, expected_student_difference=explicit,
                        loss_gradient_fd=loss_grad, cached=cached)
    checks['archive'] = out.name
    checks['archive_sha256'] = digest(out)
    return checks


def one_run(p, x, xe, ye, variance, rule, tau, eta, check_budget):
    # Allocate the same cache in every arm. No labels enter it or field().
    cached = pred(p, x).copy()
    cached.setflags(write=False)
    initial_target = model.forward(p, x, cached)[0]
    s, t = p.copy(), p.copy()
    alpha = 1. if tau is None else tau/(tau+eta)
    steps, every = round(20/eta), round(.4/eta)
    history, states, targets = [], [], []
    max_fixed_coordinate_error = 0.
    for step in range(steps+1):
        if step % every == 0:
            check_budget()
            qa = pred(s, x)
            target = model.forward(t, x, cached if rule == 'cached' else qa)[0]
            qe = pred(s, xe)
            history.append({
                'time': step*eta,
                'nmse': float(np.mean(np.sum((qe-ye)**2, axis=1))/variance),
                'discrepancy': float(.5*np.mean(np.sum((target-qa)**2, axis=1))),
                'reference_lag': float(np.linalg.norm(s-t)),
                'reference_target_change_rms': float(np.sqrt(np.mean(np.sum((target-initial_target)**2, axis=1)))),
                'reference_target_change_max_abs': float(np.max(np.abs(target-initial_target))),
            })
            states.append(np.stack([s, t]))
            targets.append(target.copy())
        if step == steps:
            break
        new_s = s + eta*field(s, t, x, cached, rule)
        new_t = alpha*t + (1-alpha)*new_s
        s, t = new_s, new_t
        s[INACTIVE] = p[INACTIVE]
        t[INACTIVE] = p[INACTIVE]
        if not (np.isfinite(s).all() and np.isfinite(t).all()):
            raise FloatingPointError(f'Nonfinite parameters: {rule}, {tau}, {eta}, {step}')
        max_fixed_coordinate_error = max(max_fixed_coordinate_error,
                                        float(np.max(np.abs(s[INACTIVE]-p[INACTIVE]))),
                                        float(np.max(np.abs(t[INACTIVE]-p[INACTIVE]))))
    return {'rule': rule, 'tau': tau, 'eta': eta, 'steps': steps, 'cache_scalars': int(cached.size),
            'cache_bytes': cached.nbytes, 'history': history, 'initial': history[0], 'final': history[-1],
            'fixed_coordinate_error': max_fixed_coordinate_error}, {
                'state': np.array(states), 'target': np.array(targets), 'cached': cached,
                'time': np.array([h['time'] for h in history])}


def main():
    if (ROOT / 'frozen_input_results.json').exists():
        raise RuntimeError('Results already exist; refusing to overwrite a confirmatory run.')
    start, cpu_start = time.perf_counter(), time.process_time()
    def budget():
        if time.process_time()-cpu_start > 600 or time.perf_counter()-start > 600:
            raise TimeoutError('10-minute CPU or conservative wall-time cap reached.')
    assert digest(ROOT / 'PROTOCOL.md') == PROTOCOL_SHA
    assert digest(BASE) == BASE_SHA
    prior = json.loads(BASE.read_text(encoding='utf-8'))
    assert digest(SRC / 'neural_geometry_gate.py') == prior['source_model_sha256']
    assert digest(SRC / 'shared_data.npz') == prior['source_data_sha256']
    data = np.load(SRC / 'shared_data.npz')
    x, xe, ye = data['x_eval'][:64], data['x_eval'][64:], data['y_eval'][64:]
    variance = float(np.mean(np.sum((ye-ye.mean(axis=0))**2, axis=1)))
    assert variance == prior['evaluation_label_variance']
    report = {'status': 'running', 'protocol_sha256': PROTOCOL_SHA, 'script_sha256': digest(Path(__file__)),
              'baseline_sha256': digest(BASE), 'source_model_sha256': digest(SRC / 'neural_geometry_gate.py'),
              'source_data_sha256': digest(SRC / 'shared_data.npz'),
              'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                              'openblas_num_threads': os.environ['OPENBLAS_NUM_THREADS'],
                              'omp_num_threads': os.environ['OMP_NUM_THREADS']},
              'evaluation_label_variance': variance, 'seeds': [], 'completed_updates': 0,
              'target_rms_convention': 'sqrt(mean over inputs of squared Euclidean output difference)'}
    def save():
        report['wall_seconds'] = time.perf_counter()-start
        report['cpu_seconds'] = time.process_time()-cpu_start
        (ROOT / 'frozen_input_results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    try:
        # All source/derivative checks before interpreting any trajectories.
        for seed in range(5):
            budget()
            old = next(s for s in prior['seeds'] if s['seed'] == seed)
            cp = SRC / f'seed_{seed}.npz'
            assert digest(cp) == old['checkpoint_sha256']
            p = np.load(cp)['parameters']
            check = initial_checks(p, x, pred(p, x).copy(), seed)
            r0 = float(np.mean(np.sum((pred(p, xe)-ye)**2, axis=1))/variance)
            assert r0 == old['no_adaptation_nmse']
            report['seeds'].append({'seed': seed, 'primary': seed != 1, 'checkpoint_sha256': digest(cp),
                                    'no_adaptation_nmse': r0, 'initial_checks': check, 'runs': []})
            print(json.dumps({'seed': seed, 'checks': check}), flush=True)
        save()
        if not all(s['initial_checks']['pass'] for s in report['seeds']):
            raise ArithmeticError('Initial derivative/equality validation failed; trajectories not started.')
        for seed_record in report['seeds']:
            seed = seed_record['seed']
            p = np.load(SRC / f'seed_{seed}.npz')['parameters']
            old = next(s for s in prior['seeds'] if s['seed'] == seed)
            for rule in ['stop', 'input', 'cached']:
                for tau in [None, 1]:
                    main_run, arrays = one_run(p, x, xe, ye, variance, rule, tau, .02, budget)
                    report['completed_updates'] += main_run['steps']
                    fine, fa = one_run(p, x, xe, ye, variance, rule, tau, .01, budget)
                    report['completed_updates'] += fine['steps']
                    out = ROOT / (f'seed{seed}_{rule}_' + ('fixed' if tau is None else 'tau1') + '.npz')
                    np.savez_compressed(out, **arrays, **{'fine_'+k: v for k, v in fa.items()})
                    main_run['halved_step'] = fine
                    main_run['archive'] = out.name
                    main_run['archive_sha256'] = digest(out)
                    main_run['final_nmse_step_difference'] = abs(main_run['final']['nmse']-fine['final']['nmse'])/max(abs(fine['final']['nmse']), .01)
                    main_run['step_check_pass'] = main_run['final_nmse_step_difference'] <= .05
                    if rule != 'cached':
                        oldrun = next(r for r in old['runs'] if r['rule'] == rule and r['tau'] == tau)
                        refs = [oldrun['final']['nmse'], oldrun['halved_step']['final']['nmse']]
                        vals = [main_run['final']['nmse'], fine['final']['nmse']]
                        main_run['baseline_absolute_differences'] = [abs(a-b) for a,b in zip(vals, refs)]
                        main_run['baseline_reproduction_pass'] = all(abs(a-b) <= max(1e-8, 1e-6*abs(b)) for a,b in zip(vals, refs))
                    if rule == 'cached' and tau is None:
                        err = max(h['reference_target_change_max_abs'] for rr in [main_run, fine] for h in rr['history'])
                        main_run['fixed_target_max_abs'] = err
                        main_run['fixed_target_pass'] = err <= 1e-12
                    seed_record['runs'].append(main_run)
                    save()
                    print(json.dumps({'seed': seed, 'rule': rule, 'tau': tau,
                                      'main_nmse': main_run['final']['nmse'], 'fine_nmse': fine['final']['nmse'],
                                      'step_difference': main_run['final_nmse_step_difference']}), flush=True)
        allruns = [r for s in report['seeds'] for r in s['runs']]
        def get(s, rule):
            return next(r for r in s['runs'] if r['rule'] == rule and r['tau'] == 1)
        gate_rows = []
        for s in report['seeds']:
            r0 = s['no_adaptation_nmse']
            original, control = get(s, 'stop'), get(s, 'cached')
            outcomes = []
            for oe, ce in [(original, control), (original['halved_step'], control['halved_step'])]:
                re, rc = oe['final']['nmse'], ce['final']['nmse']
                outcomes.append({'eta': oe['eta'], 'R0': r0, 'Re': re, 'Rc': rc,
                                 'nonnegative_excess_ratio': max(rc-r0, 0)/(re-r0) if re > r0 else None,
                                 'original_material': bool(re >= 2*r0 and re-r0 >= .01),
                                 'success': bool(re >= 2*r0 and re-r0 >= .01 and max(rc-r0, 0) <= .25*(re-r0))})
            gate_rows.append({'seed': s['seed'], 'primary': s['primary'], 'outcomes': outcomes,
                              'both_step_sizes_success': all(o['success'] for o in outcomes)})
        initial_ok = all(s['initial_checks']['pass'] for s in report['seeds'])
        reproduction_ok = all(r.get('baseline_reproduction_pass', True) for r in allruns)
        target_ok = all(r.get('fixed_target_pass', True) for r in allruns)
        step_ok = all(r['step_check_pass'] for s in report['seeds'] if s['primary'] for r in s['runs'])
        passes = [s['seed'] for s in gate_rows if s['primary'] and s['both_step_sizes_success']]
        numeric = initial_ok and reproduction_ok and target_ok and step_ok
        report['gates'] = {'individual_outcomes': gate_rows, 'initial_checks_pass': initial_ok,
                           'baseline_reproduction_pass': reproduction_ok, 'fixed_target_pass': target_ok,
                           'primary_step_checks_pass': step_ok,
                           'all_seed_step_checks_pass': all(r['step_check_pass'] for r in allruns),
                           'numerical_pass': numeric, 'primary_success_seeds': passes,
                           'primary_effect_count_pass': len(passes) >= 3,
                           'primary_gate_pass': numeric and len(passes) >= 3,
                           'decision': ('numerically_unresolved' if not numeric else
                                        ('supports_dominant_input_effect_in_this_test' if len(passes) >= 3 else
                                         'reject_dominant_input_effect_in_this_test'))}
        assert report['completed_updates'] == 90000
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'incomplete'
        report['failure'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    save()
    print(json.dumps({'gates': report['gates'], 'cpu_seconds': report['cpu_seconds'],
                      'wall_seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
