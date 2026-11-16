"""Cycle 8: fixed initial-affine output-separation analysis, no neural updates."""
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
NOTES = ROOT.parent
if not (NOTES/'2026-10-25-cycle7').exists():
    NOTES = Path(r'C:\Users\amita\source\repos\Detach Proposal Presentation\research\notes')
SRC, PREV = NOTES/'2026-10-22-cycle3', NOTES/'2026-10-25-cycle7'
PROTOCOL_SHA = '283c71ca44dcc84be95fce119044b5d4c51fd9e0b1be50693bb775a1406bc8ca'
RESULT_SHA = 'bec2eb7dab4713c4ba7910d5ec5bf40642e4346af7e816af01b2c584faf496ac'
ACTIVE = np.r_[0:32,64:114]
spec = importlib.util.spec_from_file_location('cycle3', SRC/'neural_geometry_gate.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pred(p,x): return model.forward(p,x,np.zeros_like(x))[0]
def rms(q): return float(np.sqrt(np.mean(np.sum(q*q,axis=-1))))

def finite_jacobian(p,x,h=1e-5):
    cols = []
    for j in range(len(p)):
        d = np.zeros_like(p); d[j] = h
        cols.append((pred(p+d,x)-pred(p-d,x))/(2*h))
    return np.stack(cols,axis=-1)

def affine_map(s,t,f,eta):
    n=len(f); a=1/(1+eta); eye=np.eye(n)
    m=np.block([[eye+eta*s,eta*t],[(1-a)*(eye+eta*s),a*eye+(1-a)*eta*t]])
    b=np.concatenate([eta*f,(1-a)*eta*f])
    return m,b

def propagate(s,t,f,eta,budget):
    """Direct student-first recurrence, independently assembled block recurrence."""
    m,b=affine_map(s,t,f,eta)
    n=len(f); steps=round(20/eta); every=round(.4/eta)
    u=np.zeros(n); v=np.zeros(n); z=np.zeros(2*n)
    history=[]; independent=[]; first_two=[]; max_relative=0.; max_absolute=0.
    for step in range(steps+1):
        combined=np.concatenate([u,v])
        absolute=float(np.linalg.norm(combined-z))
        max_absolute=max(max_absolute,absolute)
        max_relative=max(max_relative,absolute/max(float(np.linalg.norm(combined)),1e-12))
        if step<=2: first_two.append(combined.copy())
        if step%every==0:
            budget(); history.append(combined.copy()); independent.append(z.copy())
        if step==steps: break
        next_u=u+eta*(f+s@u+t@v)
        next_v=(v+eta*next_u)/(1+eta)
        # Algebraically same EMA, computed independently of block-matrix construction.
        u,v=next_u,next_v
        z=m@z+b
        if not (np.isfinite(u).all() and np.isfinite(z).all()):
            raise FloatingPointError('Nonfinite affine propagation')
    return {'state':np.array(history),'independent_state':np.array(independent),
            'first_two':np.array(first_two),'matrix':m,'forcing':b}, {
                'max_relative_state_difference':max_relative,'max_absolute_state_difference':max_absolute,
                'pass':max_relative<=1e-9}

def main():
    assert not (ROOT/'affine_prediction_results.json').exists(), 'Refusing to overwrite an analysis'
    start,cpu=time.perf_counter(),time.process_time()
    def budget():
        if time.process_time()-cpu>120 or time.perf_counter()-start>120:
            raise TimeoutError('Two-minute cap reached')
    assert sha(ROOT/'PROTOCOL.md')==PROTOCOL_SHA
    assert sha(PREV/'NEXT_ANALYSIS_PROTOCOL.md')==PROTOCOL_SHA
    assert sha(PREV/'frozen_input_results.json')==RESULT_SHA
    previous=json.loads((PREV/'frozen_input_results.json').read_text())
    assert sha(PREV/'frozen_input_control.py')==previous['script_sha256']
    assert sha(PREV/'PROTOCOL.md')==previous['protocol_sha256']
    assert sha(SRC/'neural_geometry_gate.py')==previous['source_model_sha256']
    assert sha(SRC/'shared_data.npz')==previous['source_data_sha256']
    # Check all 35 archived numerical artifacts, including the unused descriptive arms.
    provenance={}
    for seed in previous['seeds']:
        cp=SRC/f"seed_{seed['seed']}.npz"
        assert sha(cp)==seed['checkpoint_sha256']
        for item in [seed['initial_checks']]+seed['runs']:
            digest=sha(PREV/item['archive'])
            assert digest==item['archive_sha256']
            provenance[item['archive']]=digest
    data=np.load(SRC/'shared_data.npz')
    x,xe=data['x_eval'][:64],data['x_eval'][64:]
    # No labels are extracted or used anywhere in this analysis.
    report={'status':'running','protocol_sha256':PROTOCOL_SHA,'script_sha256':sha(Path(__file__)),
            'previous_results_sha256':RESULT_SHA,'source_model_sha256':previous['source_model_sha256'],
            'source_data_sha256':previous['source_data_sha256'],'input_archive_hashes':provenance,
            'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
                           'openblas_num_threads':os.environ['OPENBLAS_NUM_THREADS'],
                           'omp_num_threads':os.environ['OMP_NUM_THREADS']},
            'labels_used':False,'new_nonlinear_updates':0,'seeds':[]}
    def save():
        report['cpu_seconds']=time.process_time()-cpu
        report['wall_seconds']=time.perf_counter()-start
        (ROOT/'affine_prediction_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        for old in previous['seeds']:
            budget(); seed=old['seed']
            p=np.load(SRC/f'seed_{seed}.npz')['parameters']
            d=np.load(PREV/old['initial_checks']['archive'])
            blocks={k:d[k][np.ix_(ACTIVE,ACTIVE)] for k in
                    ['stop_student','stop_teacher','cached_student','cached_teacher']}
            q0=pred(p,xe); cache=pred(p,x)
            ffull=model.velocity(p,p,x,'stop'); f=ffull[ACTIVE]
            q,c=model.forward(p,x,np.zeros_like(x))
            fc=model.vjp(c,model.forward(p,x,cache)[0]-q)
            jfull=model.parameter_jacobian(p,xe,np.zeros_like(x))
            fd=finite_jacobian(p,xe)
            jac_error=float(np.max(np.abs(jfull-fd)))
            checks={'evaluator_jacobian_fd_max_abs':jac_error,
                    'initial_velocity_max_abs':float(np.max(np.abs(ffull-fc))),
                    'student_identity_max_abs':float(np.max(np.abs(d['stop_student']-d['cached_student']-d['expected_student_difference']))),
                    'reference_identity_max_abs':float(np.max(np.abs(d['stop_teacher']-d['cached_teacher']))),
                    'inactive_field_max_abs':float(np.max(np.abs(ffull[32:64]))),
                    'fd_step':1e-5}
            checks['pass']=bool(jac_error<=1e-7 and checks['initial_velocity_max_abs']<=1e-12
                                and checks['student_identity_max_abs']<=1e-7
                                and checks['reference_identity_max_abs']<=1e-7
                                and checks['inactive_field_max_abs']==0.)
            if not checks['pass']: raise ArithmeticError(f'Initial numerical checks failed: {seed}, {checks}')
            j=jfull[:,:,ACTIVE]
            arrays={**blocks,'initial_velocity':f,'evaluator_jacobian':j,'q0':q0,'active':ACTIVE}
            rec={'seed':seed,'primary':old['primary'],'checkpoint_sha256':old['checkpoint_sha256'],
                 'initial_checks':checks,'runs':[]}
            for eta,prefix in [(.02,''),(.01,'fine_')]:
                tag='main' if not prefix else 'fine'
                out={}; pc={}
                for arm in ['stop','cached']:
                    out[arm],pc[arm]=propagate(blocks[arm+'_student'],blocks[arm+'_teacher'],f,eta,budget)
                first_error=float(np.max(np.abs(out['stop']['first_two'][1]-out['cached']['first_two'][1])))
                expected=eta*eta*(blocks['stop_student']-blocks['cached_student'])@f
                observed=out['stop']['first_two'][2,:82]-out['cached']['first_two'][2,:82]
                second_error=float(np.max(np.abs(observed-expected)))
                second_scale=max(float(np.max(np.abs(expected))),1e-12)
                map_checks={'propagation':pc,'first_step_max_abs':first_error,
                            'second_step_identity_max_abs':second_error,
                            'second_step_identity_scaled_error':second_error/second_scale,
                            'pass':bool(first_error<=1e-14 and second_error<=1e-12
                                        and all(v['pass'] for v in pc.values()))}
                states={}; times=None
                for arm in ['stop','cached']:
                    saved=next(r for r in old['runs'] if r['rule']==arm and r['tau']==1)
                    a=np.load(PREV/saved['archive'])
                    states[arm]=a[prefix+'state']
                    tt=a[prefix+'time']
                    if times is not None: assert np.array_equal(tt,times)
                    times=tt
                    assert states[arm].shape==(51,2,114)
                    assert np.array_equal(states[arm][0,0],p)
                    for key,val in out[arm].items(): arrays[tag+'_'+arm+'_'+key]=val
                actual=np.array([pred(so[0],xe)-pred(sc[0],xe) for so,sc in zip(states['stop'],states['cached'])])
                u_difference=out['stop']['state'][:,:82]-out['cached']['state'][:,:82]
                predicted=np.einsum('nop,tp->tno',j,u_difference)
                floor=1e-8*max(rms(q0),1.)
                history=[]
                for tm,real,estimate in zip(times,actual,predicted):
                    magnitude=rms(real); error=rms(estimate-real)/max(magnitude,floor)
                    norm_product=float(np.linalg.norm(real)*np.linalg.norm(estimate))
                    history.append({'time':float(tm),'actual_separation_rms':magnitude,
                                    'predicted_separation_rms':rms(estimate),'error_rms':rms(estimate-real),
                                    'relative_prediction_error':error,'above_floor':magnitude>floor,
                                    'cosine':float(np.sum(real*estimate)/norm_product) if norm_product>0 else None,
                                    'projection_ratio':float(np.sum(real*estimate)/np.sum(real*real)) if magnitude>floor else None})
                primary=next(h for h in history if abs(h['time']-2)<1e-12)
                first_failure=next((h['time'] for h in history if h['above_floor'] and h['relative_prediction_error']>.25),None)
                result={'eta':eta,'alpha':1/(1+eta),'checks':map_checks,'scale_floor':floor,
                        'primary_T2':primary,'primary_effect_pass':bool(primary['above_floor'] and primary['relative_prediction_error']<=.25),
                        'first_saved_time_above_25_percent_error':first_failure,'history':history}
                rec['runs'].append(result)
                arrays[tag+'_actual_output_difference']=actual
                arrays[tag+'_predicted_output_difference']=predicted
                arrays[tag+'_time']=times
            rec['both_steps_effect_pass']=all(r['primary_effect_pass'] for r in rec['runs'])
            archive=ROOT/f'seed{seed}_affine_prediction.npz'
            np.savez_compressed(archive,**arrays)
            rec['archive']=archive.name; rec['archive_sha256']=sha(archive)
            report['seeds'].append(rec); save()
            print(json.dumps({'seed':seed,'primary':rec['primary'],
                              'T2':[r['primary_T2'] for r in rec['runs']],
                              'first_saved_failure':[r['first_saved_time_above_25_percent_error'] for r in rec['runs']],
                              'both_steps_pass':rec['both_steps_effect_pass']}),flush=True)
        numeric=all(s['initial_checks']['pass'] and all(r['checks']['pass'] for r in s['runs']) for s in report['seeds'])
        passed=[s['seed'] for s in report['seeds'] if s['primary'] and s['both_steps_effect_pass']]
        report['gates']={'numerical_pass':numeric,'primary_success_seeds':passed,'effect_count_pass':len(passed)>=3,
                         'primary_gate_pass':numeric and len(passed)>=3,
                         'decision':'numerically_unresolved' if not numeric else
                         ('supports_initial_affine_prediction_at_T2' if len(passed)>=3 else 'reject_initial_affine_prediction_at_T2')}
        report['status']='complete'
    except Exception as exc:
        report['status']='incomplete'; report['failure']=f'{type(exc).__name__}: {exc}'
        save(); raise
    save()
    print(json.dumps({'gates':report['gates'],'cpu_seconds':report['cpu_seconds'],'wall_seconds':report['wall_seconds']}),flush=True)

if __name__=='__main__': main()
