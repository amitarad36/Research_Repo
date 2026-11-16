"""Independent closed-form Hessian and saved-evidence check; no neural updates."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'2026-10-22-cycle3'
if not SRC.exists():SRC=Path(r'C:\Users\amita\source\repos\Detach Proposal Presentation\research\notes\2026-10-22-cycle3')
spec=importlib.util.spec_from_file_location('model',SRC/'neural_geometry_gate.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((ROOT/'absolute_loss_results.json').read_text())
assert r['status']=='complete' and r['gates']['all_numerical_checks_pass']
assert r['script_sha256']==sha(ROOT/'absolute_loss_audit.py') and r['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
manifest=json.loads((ROOT/'source_manifest.json').read_text())
assert sha(ROOT/manifest['file'])==manifest['sha256'] and sha(ROOT/manifest['text_file'])==manifest['text_sha256']
x=np.load(SRC/'shared_data.npz')['x_eval'][:64]
rows=[];max_hessian=max_velocity=max_symmetry=0.
for s in r['seeds']:
    p=np.load(SRC/f"seed_{s['seed']}.npz")['parameters']
    assert sha(ROOT/s['archive'])==s['archive_sha256']
    a=np.load(ROOT/s['archive']);sigma=a['sign'];n,d=sigma.shape
    _,(u,h,w2)=model.forward(p,x,np.zeros_like(x))
    manual=np.zeros((114,114))
    for row in range(n):
        augmented=np.r_[u[row],1.]
        for hidden in range(16):
            ids=np.r_[np.arange(4)*16+hidden,64+hidden]
            weight=float(w2[hidden]@sigma[row])
            second=-2*h[row,hidden]*(1-h[row,hidden]**2)*weight
            manual[np.ix_(ids,ids)]+=second*np.outer(augmented,augmented)/(n*d)
            for output in range(d):
                idx=80+2*hidden+output
                cross=sigma[row,output]*(1-h[row,hidden]**2)*augmented/(n*d)
                manual[ids,idx]+=cross;manual[idx,ids]+=cross
    max_hessian=max(max_hessian,float(np.max(np.abs(manual-a['stop_student_predicted']))))
    j=model.parameter_jacobian(p,x,np.zeros_like(x))
    stopped=sum(ji.T@si for ji,si in zip(j,sigma))/(n*d)
    full=sum(ji.T@(np.eye(d)-ai).T@si for ji,ai,si in zip(j,a['A'],sigma))/(n*d)
    max_velocity=max(max_velocity,float(np.max(np.abs(stopped-a['stop_velocity']))),
                     float(np.max(np.abs(full-a['input_velocity']))))
    max_symmetry=max(max_symmetry,float(np.max(np.abs(a['input_student_predicted']-a['input_student_predicted'].T))))
    assert np.array_equal(a['stop_student_fd'],a['cached_student_fd'])
    assert np.max(np.abs(a['stop_teacher_fd']))==0 and np.max(np.abs(a['cached_teacher_fd']))==0
    rows.append({'seed':s['seed'],'primary':s['primary'],'min_initial_abs_residual':s['residual_min_abs'],
                 'fd_step':s['fd_step'],'halvings':len(s['step_attempts'])-1,
                 'minimum_probe_margin':s['minimum_checked_probe_margin'],
                 'checked_fd_residual_evaluations':s['checked_fd_residual_evaluations'],
                 'max_check_error':max(s['checks'].values()),**s['diagnostics']})
assert max_hessian<1e-12 and max_velocity<1e-12 and max_symmetry<1e-12
summary={'results_sha256':sha(ROOT/'absolute_loss_results.json'),'verifier_sha256':sha(Path(__file__)),
         'closed_form_hessian_max_abs_error':max_hessian,'explicit_mean_velocity_max_abs_error':max_velocity,
         'full_input_student_hessian_symmetry_error':max_symmetry,
         'max_all_derivative_gradient_errors':max(max(s['checks'].values()) for s in r['seeds']),
         'minimum_initial_margin':min(s['residual_min_abs'] for s in r['seeds']),
         'minimum_checked_probe_margin':min(s['minimum_checked_probe_margin'] for s in r['seeds']),
         'total_checked_fd_residual_evaluations':sum(s['checked_fd_residual_evaluations'] for s in r['seeds']),
         'rows':rows,'cpu_seconds':r['cpu_seconds'],'wall_seconds':r['wall_seconds']}
(ROOT/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
