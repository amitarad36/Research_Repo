"""Cycle 9: local absolute-loss fields in verified residual-sign regions.
No training, adaptation trajectories, or label access. Mean over N*d entries.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
NOTES=ROOT.parent
if not (NOTES/'2026-10-25-cycle8').exists():
    NOTES=Path(r'C:\Users\amita\source\repos\Detach Proposal Presentation\research\notes')
SRC=NOTES/'2026-10-22-cycle3'
PREV=NOTES/'2026-10-25-cycle7'
PROTOCOL_SHA='8bcf5082b1f334bbb94e7cf9ed9a23ce52427844a2553ef6887d118195f94ff6'
PRIOR_SHA='bec2eb7dab4713c4ba7910d5ec5bf40642e4346af7e816af01b2c584faf496ac'
spec=importlib.util.spec_from_file_location('model',SRC/'neural_geometry_gate.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
CS=1e-20
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prediction(p,x):return model.forward(p,x,np.zeros_like(x))[0]
def residual(s,t,x,c,rule):
    q=prediction(s,x)
    return model.forward(t,x,c if rule=='cached' else q)[0]-q
def field(s,t,x,c,rule):
    q,cq=model.forward(s,x,np.zeros_like(x))
    target,ct=model.forward(t,x,c if rule=='cached' else q)
    sigma=np.sign(target-q) # sgn(0)=0 is the declared zero-residual convention.
    v=sigma
    if rule=='input':v=sigma-np.einsum('nok,no->nk',model.jac_z(t,ct),sigma)
    return model.vjp(cq,v)/q.shape[1]
def cs_jac(fun,p):
    columns=[]
    for k in range(len(p)):
        d=np.zeros_like(p,dtype=complex);d[k]=1j*CS
        columns.append(np.imag(fun(p+d))/CS)
    return np.stack(columns,axis=-1)
def central(fun,p,h):
    cols=[]
    for k in range(len(p)):
        d=np.zeros_like(p);d[k]=h
        cols.append((fun(p+d)-fun(p-d))/(2*h))
    return np.stack(cols,axis=-1)
def select_step(p,x,c,sigma,budget):
    attempts=[]
    for reduction in range(13):
        h=1e-5/2**reduction
        mismatch=0;lowest=float('inf');evaluations=0
        for axis in ['student','teacher']:
            for k in range(len(p)):
                for sign in [-1,1]:
                    dp=np.zeros_like(p);dp[k]=sign*h
                    s,t=(p+dp,p) if axis=='student' else (p,p+dp)
                    for rule in ['stop','cached']:
                        r=residual(s,t,x,c,rule)
                        mismatch+=int(np.count_nonzero(np.sign(r)!=sigma))
                        lowest=min(lowest,float(np.min(np.abs(r))))
                        evaluations+=1
                if k%32==0:budget()
        attempts.append({'h':h,'halvings':reduction,'sign_mismatches':mismatch,
                         'minimum_probe_residual_abs':lowest,'residual_evaluations':evaluations})
        if mismatch==0:return h,attempts
    return None,attempts
def main():
    assert not (ROOT/'absolute_loss_results.json').exists(),'Refusing to overwrite results'
    start,cpu=time.perf_counter(),time.process_time()
    def budget():
        if time.process_time()-cpu>120 or time.perf_counter()-start>120:raise TimeoutError('Two-minute local audit cap')
    assert sha(ROOT/'PROTOCOL.md')==PROTOCOL_SHA
    assert sha(PREV/'frozen_input_results.json')==PRIOR_SHA
    prior=json.loads((PREV/'frozen_input_results.json').read_text())
    assert sha(SRC/'neural_geometry_gate.py')==prior['source_model_sha256']
    assert sha(SRC/'shared_data.npz')==prior['source_data_sha256']
    x=np.load(SRC/'shared_data.npz')['x_eval'][:64] # No label array is extracted.
    report={'status':'running','protocol_sha256':PROTOCOL_SHA,'script_sha256':sha(Path(__file__)),
            'previous_results_sha256':PRIOR_SHA,'source_model_sha256':prior['source_model_sha256'],
            'source_data_sha256':prior['source_data_sha256'],
            'loss':'mean over all N*d absolute residual entries','outputs':2,'zero_convention':'sign(0)=0',
            'source_manifest_sha256':sha(ROOT/'source_manifest.json'),
            'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
                           'openblas_num_threads':os.environ['OPENBLAS_NUM_THREADS'],'omp_num_threads':os.environ['OMP_NUM_THREADS']},
            'labels_used':False,'new_adaptation_updates':0,'seeds':[]}
    def save():
        report['cpu_seconds']=time.process_time()-cpu;report['wall_seconds']=time.perf_counter()-start
        (ROOT/'absolute_loss_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        for old in prior['seeds']:
            budget();seed=old['seed'];cp=SRC/f'seed_{seed}.npz'
            assert sha(cp)==old['checkpoint_sha256']
            p=np.load(cp)['parameters'];c=prediction(p,x).copy();c.setflags(write=False)
            r0=residual(p,p,x,c,'stop');sigma=np.sign(r0);d=r0.shape[1]
            rec={'seed':seed,'primary':old['primary'],'checkpoint_sha256':old['checkpoint_sha256'],
                 'residual_min_abs':float(np.min(np.abs(r0))),
                 'exact_zero_count':int(np.count_nonzero(r0==0)),
                 'near_zero_count_at_1e_minus_10':int(np.count_nonzero(np.abs(r0)<1e-10))}
            if rec['exact_zero_count']:
                rec['status']='unresolved_zero_residual';report['seeds'].append(rec);save();continue
            h,attempts=select_step(p,x,c,sigma,budget);rec['step_attempts']=attempts
            if h is None:
                rec['status']='unresolved_sign_region';report['seeds'].append(rec);save();continue
            rec['fd_step']=h
            checks={};arrays={'residual':r0,'sign':sigma,'cached':c}
            checked_probes=0;min_checked=float('inf')
            def checked(s,t,rule):
                nonlocal checked_probes,min_checked
                rr=residual(s,t,x,c,rule)
                assert np.array_equal(np.sign(rr),sigma),'Finite-difference probe left sign region'
                checked_probes+=1;min_checked=min(min_checked,float(np.min(np.abs(rr))))
                return field(s,t,x,c,rule)
            q,cq=model.forward(p,x,np.zeros_like(x));target,ct=model.forward(p,x,q)
            a0=model.jac_z(p,ct);arrays['A']=a0
            fixed_target=target.copy()
            for rule in ['stop','cached','input']:
                def loss_student(s):
                    nonlocal checked_probes,min_checked
                    qq=prediction(s,x)
                    rr=(model.forward(p,x,qq)[0] if rule=='input' else fixed_target)-qq
                    assert np.array_equal(np.sign(rr),sigma),'Loss probe left sign region'
                    checked_probes+=1;min_checked=min(min_checked,float(np.min(np.abs(rr))))
                    return np.mean(np.abs(rr))
                grad=central(loss_student,p,h)
                fs=central(lambda s:checked(s,p,rule),p,h)
                ft=central(lambda t:checked(p,t,rule),p,h)
                arrays[rule+'_student_fd']=fs;arrays[rule+'_teacher_fd']=ft
                arrays[rule+'_velocity']=field(p,p,x,c,rule)
                arrays[rule+'_loss_gradient_fd']=grad
                checks[rule+'_loss_gradient_max_abs']=float(np.max(np.abs(grad+arrays[rule+'_velocity'])))
            def fixed_student_prefactor(s,v):
                return model.vjp(model.forward(s,x,np.zeros_like(x))[1],v)/d
            hsign=cs_jac(lambda s:fixed_student_prefactor(s,sigma),p)
            bsign=sigma-np.einsum('nok,no->nk',a0,sigma)
            hb=cs_jac(lambda s:fixed_student_prefactor(s,bsign),p)
            # Independent product-rule pieces: derivative of J and derivative of A.
            def a_contribution(s,t):
                qq=prediction(s,x)
                _,reference_cache=model.forward(t,x,qq)
                aa=model.jac_z(t,reference_cache)
                return -model.vjp(cq,np.einsum('nok,no->nk',aa,sigma))/d
            atheta=cs_jac(lambda s:a_contribution(s,p),p)
            aphi=cs_jac(lambda t:a_contribution(p,t),p)
            expected={'stop_student':hsign,'cached_student':hsign,
                      'stop_teacher':np.zeros((114,114)),'cached_teacher':np.zeros((114,114)),
                      'input_student':hb+atheta,'input_teacher':aphi}
            arrays['input_J_derivative_piece']=hb;arrays['input_A_student_piece']=atheta
            for name,value in expected.items():
                arrays[name+'_predicted']=value
                checks[name+'_block_max_abs']=float(np.max(np.abs(arrays[name+'_fd']-value)))
            checks['stopped_cached_student_difference']=float(np.max(np.abs(arrays['stop_student_fd']-arrays['cached_student_fd'])))
            checks['stopped_cached_teacher_difference']=float(np.max(np.abs(arrays['stop_teacher_fd']-arrays['cached_teacher_fd'])))
            checks['stopped_cached_velocity_difference']=float(np.max(np.abs(arrays['stop_velocity']-arrays['cached_velocity'])))
            rec['checks']=checks;rec['all_checks_pass']=all(v<=1e-7 for v in checks.values())
            rec['checked_fd_residual_evaluations']=checked_probes;rec['minimum_checked_probe_margin']=min_checked
            rec['diagnostics']={'A_max_abs':float(np.max(np.abs(a0))),
                                'input_teacher_block_frobenius':float(np.linalg.norm(aphi)),
                                'input_student_A_piece_frobenius':float(np.linalg.norm(atheta)),
                                'stopped_student_block_frobenius':float(np.linalg.norm(hsign)),
                                'input_minus_stopped_velocity_norm':float(np.linalg.norm(arrays['input_velocity']-arrays['stop_velocity']))}
            out=ROOT/f'seed{seed}_absolute_blocks.npz';np.savez_compressed(out,**arrays)
            rec['archive']=out.name;rec['archive_sha256']=sha(out);rec['status']='checked'
            report['seeds'].append(rec);save()
            print(json.dumps({'seed':seed,'min_residual':rec['residual_min_abs'],'h':h,
                              'halvings':len(attempts)-1,'checks':checks,'diagnostics':rec['diagnostics']}),flush=True)
        resolved=all(s['status']=='checked' for s in report['seeds'])
        numeric=resolved and all(s['all_checks_pass'] for s in report['seeds'])
        report['gates']={'all_seeds_resolved':resolved,'all_numerical_checks_pass':numeric,
                         'decision':'local_stopped_cached_identity_verified_for_absolute_loss' if numeric else 'audit_unresolved'}
        report['status']='complete'
    except Exception as exc:
        report['status']='incomplete';report['failure']=f'{type(exc).__name__}: {exc}';save();raise
    save();print(json.dumps({'gates':report['gates'],'cpu_seconds':report['cpu_seconds'],'wall_seconds':report['wall_seconds']}),flush=True)
if __name__=='__main__':main()
