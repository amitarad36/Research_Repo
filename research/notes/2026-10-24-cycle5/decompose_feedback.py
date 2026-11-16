"""Cycle-5 preregistered diagnostic; no training or network mutation on disk."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import hashlib
import json
import time
import platform
import numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'2026-10-22-cycle3'
if not SRC.exists():SRC=(Path(__file__).resolve().parents[3] / 'research/notes/2026-10-22-cycle3')
spec=importlib.util.spec_from_file_location('model_cycle3',SRC/'neural_geometry_gate.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
ACTIVE=np.r_[0:32,64:114]
CS=1e-20

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def frozen_residual_velocity(s,t,x,r0,rule):
    q,cq=model.forward(s,x,np.zeros_like(x))
    if rule=='input':
        _,ct=model.forward(t,x,q)
        v=r0-np.einsum('nok,no->nk',model.jac_z(t,ct),r0)
    else:v=r0
    return model.vjp(cq,v)

def direct_remainder(p,x,r0,rule):
    rs=np.zeros((82,82));rt=np.zeros((82,82))
    for k,i in enumerate(ACTIVE):
        d=np.zeros(114,dtype=complex);d[i]=1j*CS
        rs[:,k]=frozen_residual_velocity(p+d,p,x,r0,rule).imag[ACTIVE]/CS
        rt[:,k]=frozen_residual_velocity(p,p+d,x,r0,rule).imag[ACTIVE]/CS
    return rs,rt

def spectrum(s,t,tau,je,je_norm):
    eye=np.eye(82)
    m=np.block([[s,t],[eye/tau,-eye/tau]])
    eig,v=np.linalg.eig(m)
    u=v[:82,:]
    obs=np.linalg.norm(je@u,axis=0)/np.maximum(np.linalg.norm(u,axis=0)*je_norm,1e-30)
    selected=(eig.real>1e-6)&(obs>=1e-3)
    lead=int(np.argmax(eig.real))
    eligible=np.where(selected)[0]
    best=int(eligible[np.argmax(eig.real[eligible])]) if len(eligible) else None
    eigerr=float(np.max(np.linalg.norm(m@v-v*eig[None,:],axis=0))/max(1,np.linalg.norm(m,'fro')))
    assert eigerr<=1e-10,eigerr
    eta=.02;alpha=tau/(tau+eta)
    jd=np.block([[eye+eta*s,eta*t],[(1-alpha)*(eye+eta*s),alpha*eye+(1-alpha)*eta*t]])
    ed=np.linalg.eigvals(jd)
    out={'max_real':float(eig[lead].real),'leading_imag':float(eig[lead].imag),
         'leading_mode_observability_ratio':float(obs[lead]),
         'positive_mode_count':int(np.sum(eig.real>1e-6)),
         'observable_positive_mode_count':int(np.sum(selected)),
         'max_observable_positive_real':float(eig[best].real) if best is not None else None,
         'observable_positive_mode_exists':bool(best is not None),
         'eigenpair_backward_residual':eigerr,
         'discrete_spectral_radius':float(np.max(np.abs(ed)))}
    return out,eig,obs

def main():
    begin=time.perf_counter()
    previous=json.loads((SRC/'neural_gate_results.json').read_text())
    assert digest(SRC/'neural_geometry_gate.py')==previous['script_sha256']
    data=np.load(SRC/'shared_data.npz')
    x=data['x_eval'][:64];xe=data['x_eval'][64:]
    report={'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'script_sha256':digest(Path(__file__)),
            'source_script_sha256':digest(SRC/'neural_geometry_gate.py'),
            'source_data_sha256':digest(SRC/'shared_data.npz'),
            'environment':{'numpy':np.__version__,'python':sys.version,'platform':platform.platform()},'seeds':[]}
    for seed in range(5):
        saved=np.load(SRC/f'seed_{seed}.npz');p=saved['parameters']
        q,_=model.forward(p,x,np.zeros_like(x))
        tq,ct=model.forward(p,x,q);r0=tq-q
        j=model.parameter_jacobian(p,x,np.zeros_like(x))[:,:,ACTIVE]
        l=model.parameter_jacobian(p,x,q)[:,:,ACTIVE]
        a=model.jac_z(p,ct);b=np.eye(2)[None,:,:]-a
        bj=np.einsum('nok,nkp->nop',b,j)
        je=model.parameter_jacobian(p,xe,np.zeros_like(xe))[:,:,ACTIVE].reshape(-1,82)
        je_norm=float(np.linalg.norm(je,2))
        # Holding q fixed is crucial when checking the reference-parameter derivative.
        lcs=np.empty_like(l)
        for k,i in enumerate(ACTIVE):
            d=np.zeros(114,dtype=complex);d[i]=1j*CS
            lcs[:,:,k]=model.forward(p+d,x,q)[0].imag/CS
        lerr=float(np.max(np.abs(l-lcs)));assert lerr<=1e-7,lerr
        row={'seed':seed,'primary':seed!=1,'checkpoint_sha256':digest(SRC/f'seed_{seed}.npz'),
             'reference_parameter_jacobian_check_abs':lerr,
             'relative_L_J_difference':float(np.linalg.norm(l-j)/np.linalg.norm(j)),
             'residual_rms':float(np.linalg.norm(r0)/np.sqrt(len(x))),
             'rules':{}}
        arrays={'J':j,'L':l,'A':a,'J_eval':je}
        for rule in ['input','stop']:
            c=bj if rule=='input' else j
            s0=-np.einsum('nop,noq->pq',c,bj)/len(x)
            t0=np.einsum('nop,noq->pq',c,l)/len(x)
            tj=np.einsum('nop,noq->pq',c,j)/len(x)
            sf=saved[rule+'_dstudent'][np.ix_(ACTIVE,ACTIVE)]
            tf=saved[rule+'_dteacher'][np.ix_(ACTIVE,ACTIVE)]
            rs,rt=direct_remainder(p,x,r0,rule)
            err=float(max(np.max(np.abs(sf-s0-rs)),np.max(np.abs(tf-t0-rt))))
            assert err<=1e-7,err
            if rule=='stop':assert np.max(np.abs(rt))<=1e-12
            rr={'decomposition_check_abs':err,
                'student_remainder_relative_fro':float(np.linalg.norm(rs)/np.linalg.norm(sf)),
                'teacher_remainder_relative_fro':float(np.linalg.norm(rt)/np.linalg.norm(tf)),
                'variants':{}}
            for variant,sm,tm in [('full',sf,tf),('zero_residual',s0,t0),('shared_J',s0,tj)]:
                rows=[]
                for tau in [1,20]:
                    out,eig,obs=spectrum(sm,tm,tau,je,je_norm)
                    out['tau']=tau;rows.append(out)
                    arrays[f'{rule}_{variant}_tau{tau}_eigen']=eig
                    arrays[f'{rule}_{variant}_tau{tau}_observability']=obs
                rr['variants'][variant]=rows
            for variant in ['zero_residual','shared_J']:
                for i in range(2):
                    full=rr['variants']['full'][i]['max_observable_positive_real']
                    ablated=rr['variants'][variant][i]['max_observable_positive_real']
                    rr['variants'][variant][i]['relative_rate_difference_from_full']=float(abs(ablated-full)/max(abs(full),1e-6)) if full is not None and ablated is not None else None
            row['rules'][rule]=rr
            arrays.update({rule+'_Sfull':sf,rule+'_Tfull':tf,rule+'_S0':s0,
                           rule+'_T0':t0,rule+'_Tshared':tj,rule+'_Rstudent':rs,rule+'_Rteacher':rt})
        np.savez_compressed(ROOT/f'seed_{seed}_decomposition.npz',**arrays)
        report['seeds'].append(row)
        print(json.dumps({'seed':seed,'L_J_relative':row['relative_L_J_difference'],
               'input_t1_rates':[row['rules']['input']['variants'][v][0]['max_real'] for v in ['full','zero_residual','shared_J']],
               'stop_t1_rates':[row['rules']['stop']['variants'][v][0]['max_real'] for v in ['full','zero_residual','shared_J']]}),flush=True)
    gate={}
    eligible=[s for s in report['seeds'] if s['primary']]
    for variant in ['zero_residual','shared_J']:
        surviving={}
        for rule in ['input','stop']:
            for i,tau in enumerate([1,20]):
                surviving[f'{rule}_tau{tau}']=[s['seed'] for s in eligible if s['rules'][rule]['variants'][variant][i]['observable_positive_mode_exists']]
        diff=[s['rules'][rule]['variants'][variant][i]['relative_rate_difference_from_full'] for s in eligible for rule in ['input','stop'] for i in range(2)]
        gate[variant]={'surviving_seeds':surviving,'signature_survives':all(len(v)>=3 for v in surviving.values()),
                       'all_primary_rates_within_10_percent':all(d is not None and d<=.1 for d in diff),
                       'max_primary_relative_rate_difference':max(d for d in diff if d is not None)}
    report['gates']=gate;report['wall_seconds']=time.perf_counter()-begin
    (ROOT/'decomposition_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'gates':gate,'wall_seconds':report['wall_seconds']}),flush=True)

if __name__=='__main__':main()
