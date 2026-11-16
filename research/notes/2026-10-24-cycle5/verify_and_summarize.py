"""Post-run algebra/provenance checks; no changed experimental gates."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import numpy as np
import decompose_feedback as d

root = Path(__file__).resolve().parent
r = json.loads((root / 'decomposition_results.json').read_text())
prior_path = d.SRC.parent / '2026-10-22-cycle4/finite_response_results.json'
prior = json.loads(prior_path.read_text())
assert r['protocol_sha256'] == d.digest(root / 'PROTOCOL.md')
assert r['script_sha256'] == d.digest(root / 'decompose_feedback.py')
assert r['source_script_sha256'] == prior['source_model_sha256']
assert r['source_data_sha256'] == prior['source_data_sha256']
errors = []
for s, old in zip(r['seeds'], prior['seeds']):
    assert s['seed'] == old['seed']
    assert s['checkpoint_sha256'] == old['checkpoint_sha256']
    a = np.load(root / f"seed_{s['seed']}_decomposition.npz")
    # An explicit sample loop checks the matrix assembly independently of einsum.
    for rule in ['input', 'stop']:
        ss = np.zeros((82, 82)); tt = ss.copy(); tj = ss.copy()
        for j, l, az in zip(a['J'], a['L'], a['A']):
            bj = (np.eye(2) - az) @ j
            c = bj if rule == 'input' else j
            ss -= c.T @ bj / len(a['J'])
            tt += c.T @ l / len(a['J'])
            tj += c.T @ j / len(a['J'])
        err = max(np.max(np.abs(ss - a[rule + '_S0'])),
                  np.max(np.abs(tt - a[rule + '_T0'])),
                  np.max(np.abs(tj - a[rule + '_Tshared'])))
        assert err < 1e-12
        errors.append(float(err))
primary = [s for s in r['seeds'] if s['primary']]
rows = [v for s in primary for rule in s['rules'].values()
        for group in rule['variants'].values() for v in group]
all_rows = [v for s in r['seeds'] for rule in s['rules'].values()
            for group in rule['variants'].values() for v in group]
summary = {
    'result_sha256': d.digest(root / 'decomposition_results.json'),
    'verification_script_sha256': d.digest(Path(__file__)),
    'prior_cycle4_result_sha256': d.digest(prior_path),
    'prior_cycle4_code_data_and_all_checkpoints_match': True,
    'matrix_count': len(all_rows),
    'primary_matrix_count': len(rows),
    'loop_assembly_max_abs': max(errors),
    'reference_parameter_jacobian_max_abs': max(s['reference_parameter_jacobian_check_abs'] for s in r['seeds']),
    'decomposition_max_abs': max(v['decomposition_check_abs'] for s in r['seeds'] for v in s['rules'].values()),
    'eigenpair_backward_residual_max': max(v['eigenpair_backward_residual'] for v in all_rows),
    'primary_leading_observability_range': [min(v['leading_mode_observability_ratio'] for v in rows), max(v['leading_mode_observability_ratio'] for v in rows)],
    'primary_discrete_radius_range': [min(v['discrete_spectral_radius'] for v in rows), max(v['discrete_spectral_radius'] for v in rows)],
    'primary_student_remainder_ratio_range': [min(v['student_remainder_relative_fro'] for s in primary for v in s['rules'].values()), max(v['student_remainder_relative_fro'] for s in primary for v in s['rules'].values())],
    'primary_input_teacher_remainder_ratio_range': [min(s['rules']['input']['teacher_remainder_relative_fro'] for s in primary), max(s['rules']['input']['teacher_remainder_relative_fro'] for s in primary)],
    'primary_L_J_difference_range': [min(s['relative_L_J_difference'] for s in primary), max(s['relative_L_J_difference'] for s in primary)],
    'gates': r['gates'],
}
(root / 'verification_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
