"""Cycle-4 bounded finite-time test; see PROTOCOL.md written before results."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
import json
import hashlib
import time
import platform
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'2026-10-22-cycle3'
if not SRC.exists():
    SRC=(Path(__file__).resolve().parents[3] / 'research/notes/2026-10-22-cycle3')
spec=importlib.util.spec_from_file_location('model_cycle3',SRC/'neural_geometry_gate.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
ACTIVE=np.r_[0:32,64:114]
INACTIVE=np.arange(32,64)
EPS=[1e-4,1e-5]
CS=1e-20

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rms(v):return float(np.linalg.norm(v)/np.sqrt(len(v)))
def pred(p,x):return model.forward(p,x,np.zeros_like(x))[0]
def pred_jvp(p,d,x):return pred(p+1j*CS*d,x).imag/CS
def field_jvp(s,t,ds,dt,x,rule):
    return model.velocity(s+1j*CS*ds,t+1j*CS*dt,x,rule).imag/CS

def directions(n,seed):
    ds=n['stop_dstudent'][np.ix_(ACTIVE,ACTIVE)]
    dt=n['stop_dteacher'][np.ix_(ACTIVE,ACTIVE)]
    m=np.block([[ds,dt],[np.eye(82),-np.eye(82)]])
    ev,vec=np.linalg.eig(m); idx=np.argmax(ev.real)
    u=vec[:82,idx];u=u*np.exp(-1j*np.angle(u[np.argmax(np.abs(u))]))
    d=np.zeros(114);d[ACTIVE]=u.real;d/=np.linalg.norm(d)
    r=np.zeros(114);r[ACTIVE]=np.random.default_rng(40000+seed).normal(size=82);r/=np.linalg.norm(r)
    return np.array([d,r]),{'selected_eigenvalue_real':float(ev[idx].real),
                             'selected_eigenvalue_imag':float(ev[idx].imag)}

def verify_jvp(n,p,x,dirs):
    checks={}
    ds=dirs[0];dt=dirs[1]
    for rule in ['input','stop']:
        exact=n[rule+'_dstudent']@ds+n[rule+'_dteacher']@dt
        cs=field_jvp(p,p,ds,dt,x,rule)
        err=float(np.max(np.abs(cs-exact)))
        assert err<1e-7,(rule,err)
        checks[rule]=err
    return checks

def one_run(p,dirs,x,xe,ye,variance,rule,tau,eta,paired):
    nsteps=round(20/eta);every=round(.4/eta)
    s=p.copy();t=p.copy()
    ds=dirs.copy();dt=np.zeros_like(ds)
    alpha=1 if tau is None else tau/(tau+eta)
    initial=np.array([rms(pred_jvp(p,d,xe)) for d in dirs])
    assert initial.min()>1e-6,initial
    ps=np.empty((2,2,2,114));pt=np.empty_like(ps)
    if paired:
        for k in range(2):
            for e,eps in enumerate(EPS):
                for h,sgn in enumerate([1,-1]):
                    ps[k,e,h]=p+sgn*eps*dirs[k];pt[k,e,h]=p
    records=[];states=[];tangents=[]
    check_directions=np.array([dirs[0],dirs[1]])
    directional_checks=[]
    for step in range(nsteps+1):
        if step%every==0:
            q=pred(s,xe);qa=pred(s,x)
            residual=model.forward(t,x,qa)[0]-qa
            gains=[];errors=[]
            for k in range(2):
                dq=pred_jvp(s,ds[k],xe)
                gains.append(rms(dq)/initial[k])
                errs=[]
                if paired:
                    for e,eps in enumerate(EPS):
                        fd=(pred(ps[k,e,0],xe)-pred(ps[k,e,1],xe))/(2*eps)
                        errs.append(float(np.linalg.norm(fd-dq)/max(np.linalg.norm(dq),1e-12)))
                errors.append(errs)
            records.append({'time':step*eta,'nmse':float(np.mean(np.sum((q-ye)**2,axis=1))/variance),
              'discrepancy':float(.5*np.mean(np.sum(residual**2,axis=1))),
              'reference_lag':float(np.linalg.norm(s-t)), 'gain':gains,'relative_tangent_error':errors})
            states.append(np.stack([s,t]));tangents.append(np.stack([ds,dt]))
        if step in [0,nsteps//2,nsteps]:
            d1,d2=check_directions
            cs=field_jvp(s,t,d1,d2,x,rule)
            h=1e-5
            fd=(model.velocity(s+h*d1,t+h*d2,x,rule)-model.velocity(s-h*d1,t-h*d2,x,rule))/(2*h)
            err=float(np.max(np.abs(cs-fd)));assert err<1e-7,err
            directional_checks.append({'time':step*eta,'max_abs':err})
        if step==nsteps:break
        # Tangent of the actual map at old (s,t), including student-first EMA.
        next_ds=np.array([ds[k]+eta*field_jvp(s,t,ds[k],dt[k],x,rule) for k in range(2)])
        next_dt=alpha*dt+(1-alpha)*next_ds
        new_s=s+eta*model.velocity(s,t,x,rule)
        new_t=alpha*t+(1-alpha)*new_s
        if paired:
            for k in range(2):
                for e in range(2):
                    for h in range(2):
                        ps[k,e,h]+=eta*model.velocity(ps[k,e,h],pt[k,e,h],x,rule)
                        pt[k,e,h]=alpha*pt[k,e,h]+(1-alpha)*ps[k,e,h]
        s,t=new_s,new_t;ds,dt=next_ds,next_dt
        # Keep exactly fixed coordinates fixed rather than introducing roundoff by EMA.
        s[INACTIVE]=p[INACTIVE];t[INACTIVE]=p[INACTIVE]
        if paired:ps[:,:,:,INACTIVE]=p[INACTIVE];pt[:,:,:,INACTIVE]=p[INACTIVE]
        assert np.isfinite(s).all() and np.isfinite(ds).all()
    assert np.max(np.abs(ds[:,INACTIVE]))==0 and np.max(np.abs(dt[:,INACTIVE]))==0
    result={'rule':rule,'tau':tau,'eta':eta,'steps':nsteps,'initial_prediction_sensitivity':initial.tolist(),
            'history':records,'final':records[-1],'directional_checks':directional_checks}
    return result,{'state':np.array(states),'tangent':np.array(tangents),'directions':dirs}

def main():
    start=time.perf_counter()
    previous=json.loads((SRC/'neural_gate_results.json').read_text())
    assert digest(SRC/'neural_geometry_gate.py')==previous['script_sha256']
    data=np.load(SRC/'shared_data.npz')
    x=data['x_eval'][:64];xe=data['x_eval'][64:];ye=data['y_eval'][64:]
    variance=float(np.mean(np.sum((ye-ye.mean(axis=0))**2,axis=1)))
    report={'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'script_sha256':digest(Path(__file__)),
            'source_model_sha256':digest(SRC/'neural_geometry_gate.py'),
            'source_data_sha256':digest(SRC/'shared_data.npz'),
            'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()},
            'evaluation_label_variance':variance,'seeds':[]}
    for seed in range(5):
        n=np.load(SRC/f'seed_{seed}.npz');p=n['parameters']
        dirs,meta=directions(n,seed)
        checks=verify_jvp(n,p,x,dirs)
        baseline=float(np.mean(np.sum((pred(p,xe)-ye)**2,axis=1))/variance)
        s={'seed':seed,'primary':seed!=1,'checkpoint_sha256':digest(SRC/f'seed_{seed}.npz'),
           'direction':meta,'initial_derivative_checks':checks,'no_adaptation_nmse':baseline,
           'no_adaptation_gains':[1.,1.],'runs':[]}
        for rule in ['input','stop']:
            for tau in [None,1,20]:
                t0=time.perf_counter()
                base,arrays=one_run(p,dirs,x,xe,ye,variance,rule,tau,.02,True)
                fine,farrays=one_run(p,dirs,x,xe,ye,variance,rule,tau,.01,False)
                tag=f'seed{seed}_{rule}_'+('fixed' if tau is None else f'tau{tau}')
                np.savez_compressed(ROOT/(tag+'.npz'),**arrays,
                      fine_state=farrays['state'],fine_tangent=farrays['tangent'])
                base['halved_step']=fine
                base['final_nmse_relative_step_difference']=float(abs(base['final']['nmse']-fine['final']['nmse'])/max(fine['final']['nmse'],1e-8))
                base['final_gain_relative_step_difference']=(np.abs(np.array(base['final']['gain'])-fine['final']['gain'])/np.maximum(fine['final']['gain'],1e-12)).tolist()
                base['material_risk_deterioration']=bool(base['final']['nmse']>=1.25*baseline and base['final']['nmse']-baseline>=.005)
                base['max_tangent_error_by_epsilon']=[max(rec['relative_tangent_error'][k][e] for rec in base['history'] for k in range(2)) for e in range(2)]
                base['wall_seconds']=time.perf_counter()-t0
                s['runs'].append(base)
                print(json.dumps({'seed':seed,'rule':rule,'tau':tau,'gain':base['final']['gain'],
                    'nmse':base['final']['nmse'],'nmse0':baseline,'tangent_error':base['max_tangent_error_by_epsilon'],
                    'seconds':base['wall_seconds']}),flush=True)
        report['seeds'].append(s)
        (ROOT/'finite_response_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    eligible=[s for s in report['seeds'] if s['primary']]
    def get(s,rule,tau):return next(r for r in s['runs'] if r['rule']==rule and r['tau']==tau)
    amp=[s['seed'] for s in eligible if get(s,'stop',1)['final']['gain'][0]>=2 and get(s,'stop',None)['final']['gain'][0]<=1.25]
    sep=[s['seed'] for s in eligible if get(s,'stop',1)['final']['gain'][0]/get(s,'input',1)['final']['gain'][0]>=1.25]
    allruns=[r for s in report['seeds'] for r in s['runs']]
    report['gates']={'amplification_seeds':amp,'amplification_pass':len(amp)>=3,
                    'rule_separation_seeds':sep,'rule_separation_pass':len(sep)>=3,
                    'numerical_pass':all(r['max_tangent_error_by_epsilon'][1]<=.01 for r in allruns),
                    'discretization_pass':all(r['final_nmse_relative_step_difference']<=.05 and r['final_gain_relative_step_difference'][0]<=.05 for r in allruns)}
    report['wall_seconds']=time.perf_counter()-start
    (ROOT/'finite_response_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'gates':report['gates'],'wall_seconds':report['wall_seconds']}),flush=True)

if __name__=='__main__':main()
