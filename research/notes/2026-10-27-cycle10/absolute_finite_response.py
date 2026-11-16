"""Cycle 10: unchanged absolute-loss finite-time protocol, NumPy/CPU only."""
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
if not (NOTES/'2026-10-26-cycle9').exists():
    NOTES=(Path(__file__).resolve().parents[3] / 'research/notes')
AUDIT=NOTES/'2026-10-26-cycle9';SRC=NOTES/'2026-10-22-cycle3';PRIOR=NOTES/'2026-10-25-cycle7'
PROTOCOL_SHA='081806a9e30ac8ec09466cc4f9a3448a2258ed696dfd51cc0cfb92061da110d1'
AUDIT_SHA='2e6be33181896316189efae3829abdf352f492402e734c084bdfed8c15ee2043'
PRIOR_SHA='bec2eb7dab4713c4ba7910d5ec5bf40642e4346af7e816af01b2c584faf496ac'
spec=importlib.util.spec_from_file_location('absolute_audit',AUDIT/'absolute_loss_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
model=audit.model
INACTIVE=np.arange(32,64)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pred(p,x):return model.forward(p,x,np.zeros_like(x))[0]
def quantities(s,t,x,c,rule):
    q,cq=model.forward(s,x,np.zeros_like(x))
    target,ct=model.forward(t,x,c if rule=='cached' else q)
    r=target-q;sig=np.sign(r);v=sig
    if rule=='input':v=sig-np.einsum('nok,no->nk',model.jac_z(t,ct),sig)
    f=model.vjp(cq,v)/q.shape[1]
    return f,r,sig,target

def run_group(p,x,xe,ye,variance,rules,tau,eta,budget,progress):
    count=len(rules);pair=count==2
    state=np.broadcast_to(p,(count,2,len(p))).copy()
    caches=np.array([pred(p,x).copy() for _ in rules])
    caches.setflags(write=False)
    target0=model.forward(p,x,caches[0])[0]
    initial_fixed=p[INACTIVE].copy()
    alpha=1. if tau is None else tau/(tau+eta)
    steps,every=round(20/eta),round(.4/eta)
    histories=[[] for _ in rules];states=[];targets=[];times=[]
    perstep=[];old_sign=None
    events={k:None for k in ['first_sign_disagreement','first_velocity_disagreement','first_state_disagreement']} if pair else {}
    event_arrays={};common_count=0;common_sign_count=0;max_common_velocity=0.;max_common_next=0.
    max_fixed_target=np.zeros(count);zero_occurrences=np.zeros(count,dtype=int)
    for step in range(steps+1):
        info=[quantities(state[k,0],state[k,1],x,caches[k],rule) for k,rule in enumerate(rules)]
        fields=np.array([v[0] for v in info]);residuals=np.array([v[1] for v in info])
        sig=np.array([v[2] for v in info],dtype=np.int8);target=np.array([v[3] for v in info])
        zeros=np.count_nonzero(sig==0,axis=(1,2));zero_occurrences+=zeros
        changes=np.zeros(count,dtype=int) if old_sign is None else np.count_nonzero(sig!=old_sign,axis=(1,2))
        max_fixed_target=np.maximum(max_fixed_target,np.max(np.abs(target-target0),axis=(1,2)))
        if pair:
            state_gap=float(np.max(np.abs(state[0]-state[1])))
            velocity_gap=float(np.max(np.abs(fields[0]-fields[1])))
            sign_gap=int(np.count_nonzero(sig[0]!=sig[1]))
            common=state_gap<=1e-12
            if common:common_count+=1
            if common and sign_gap==0:
                common_sign_count+=1;max_common_velocity=max(max_common_velocity,velocity_gap)
            row=[step*eta,*changes.tolist(),sign_gap,*zeros.tolist(),state_gap,velocity_gap,int(common)]
            for key,trigger in [('first_sign_disagreement',sign_gap>0),('first_velocity_disagreement',velocity_gap>1e-12),('first_state_disagreement',state_gap>1e-12)]:
                if events[key] is None and trigger:
                    events[key]={'step':step,'time':step*eta,'state_max_abs_difference':state_gap,
                                 'velocity_max_abs_difference':velocity_gap,'sign_disagreement_count':sign_gap,
                                 'common_state_within_1e_minus_12':common}
                    event_arrays[key+'_state']=state.copy();event_arrays[key+'_field']=fields.copy()
                    event_arrays[key+'_residual']=residuals.copy();event_arrays[key+'_sign']=sig.copy()
        else:row=[step*eta,int(changes[0]),int(zeros[0])]
        perstep.append(row);old_sign=sig.copy()
        if step%every==0:
            budget();states.append(state.copy());targets.append(target.copy());times.append(step*eta)
            for k in range(count):
                qe=pred(state[k,0],xe)
                histories[k].append({'time':step*eta,'nmse':float(np.mean(np.sum((qe-ye)**2,axis=1))/variance),
                                     'discrepancy':float(np.mean(np.abs(residuals[k]))),
                                     'reference_lag':float(np.linalg.norm(state[k,0]-state[k,1])),
                                     'reference_target_change_rms':float(np.sqrt(np.mean(np.sum((target[k]-target0)**2,axis=1)))),
                                     'reference_target_change_max_abs':float(np.max(np.abs(target[k]-target0))),
                                     'zero_residual_count':int(zeros[k])})
        if step==steps:break
        new_s=state[:,0]+eta*fields
        new_t=alpha*state[:,1]+(1-alpha)*new_s
        new_s[:,INACTIVE]=initial_fixed;new_t[:,INACTIVE]=initial_fixed
        next_state=np.stack([new_s,new_t],axis=1)
        if pair and common and sign_gap==0:
            max_common_next=max(max_common_next,float(np.max(np.abs(next_state[0]-next_state[1]))))
        state=next_state;progress(count)
        if not np.isfinite(state).all():raise FloatingPointError(f'Nonfinite parameters: {rules}, {tau}, {eta}, {step}')
    saved_states=np.array(states);saved_targets=np.array(targets)
    runs=[];arrays={'time':np.array(times),'per_step':np.array(perstep),**event_arrays}
    for k,rule in enumerate(rules):
        arrays[rule+'_state']=saved_states[:,k];arrays[rule+'_target']=saved_targets[:,k]
        arrays[rule+'_cache']=caches[k]
        run={'rule':rule,'tau':tau,'eta':eta,'steps':steps,'cache_scalars':128,'cache_bytes':1024,
             'history':histories[k],'initial':histories[k][0],'final':histories[k][-1],
             'zero_residual_observations':int(zero_occurrences[k]),
             'fixed_coordinate_error':float(np.max(np.abs(saved_states[:,k][:,:,INACTIVE]-initial_fixed)))}
        if rule=='cached' and tau is None:
            run['fixed_target_max_abs']=float(max_fixed_target[k]);run['fixed_target_pass']=bool(max_fixed_target[k]<=1e-12)
        runs.append(run)
    if pair:
        sign_event,velocity_event,state_event=[events[k] for k in events]
        ordered=(velocity_event is None or (sign_event is not None and sign_event['step']<=velocity_event['step']))
        ordered=ordered and (state_event is None or (velocity_event is not None and velocity_event['step']<state_event['step']))
        checks={'shared_state_samples':common_count,'shared_state_and_sign_samples':common_sign_count,
                'max_velocity_difference_under_shared_state_and_sign':max_common_velocity,
                'max_next_state_difference_under_shared_state_and_sign':max_common_next,
                'onset_ordering_pass':ordered,'pass':bool(ordered and max_common_velocity<=1e-12 and max_common_next<=1e-12)}
        pair_record={'eta':eta,'tau':tau,'events':events,'checks':checks,
                     'per_step_columns':['time','stop_sign_change_count','cached_sign_change_count','between_arm_sign_disagreement_count',
                                         'stop_exact_zero_count','cached_exact_zero_count','state_max_abs_difference','velocity_max_abs_difference','shared_state_flag']}
    else:pair_record={'eta':eta,'tau':tau,'per_step_columns':['time','input_sign_change_count','input_exact_zero_count']}
    return runs,arrays,pair_record

def main():
    assert not (ROOT/'absolute_finite_results.json').exists(),'Refusing to overwrite a fixed-protocol run'
    start,cpu=time.perf_counter(),time.process_time()
    def budget():
        if time.process_time()-cpu>600 or time.perf_counter()-start>600:raise TimeoutError('Ten-minute compute/wall cap reached')
    assert sha(ROOT/'PROTOCOL.md')==PROTOCOL_SHA
    assert sha(AUDIT/'absolute_loss_results.json')==AUDIT_SHA
    assert sha(PRIOR/'frozen_input_results.json')==PRIOR_SHA
    audited=json.loads((AUDIT/'absolute_loss_results.json').read_text())
    prior=json.loads((PRIOR/'frozen_input_results.json').read_text())
    assert sha(AUDIT/'absolute_loss_audit.py')==audited['script_sha256']
    assert sha(SRC/'neural_geometry_gate.py')==audited['source_model_sha256']
    assert sha(SRC/'shared_data.npz')==audited['source_data_sha256']
    assert audited['gates']['all_numerical_checks_pass']
    data=np.load(SRC/'shared_data.npz');x=data['x_eval'][:64];xe=data['x_eval'][64:];ye=data['y_eval'][64:]
    variance=float(np.mean(np.sum((ye-ye.mean(0))**2,axis=1)))
    assert variance==prior['evaluation_label_variance']
    report={'status':'running','protocol_sha256':PROTOCOL_SHA,'script_sha256':sha(Path(__file__)),
            'audit_results_sha256':AUDIT_SHA,'audit_script_sha256':audited['script_sha256'],
            'source_model_sha256':audited['source_model_sha256'],'source_data_sha256':audited['source_data_sha256'],
            'prior_baseline_sha256':PRIOR_SHA,'evaluation_label_variance':variance,
            'loss':'mean absolute residual over N*d entries, sign(0)=0; no rescaling',
            'completed_updates':0,'completed_trajectories':0,'seeds':[],
            'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
                           'openblas_num_threads':os.environ['OPENBLAS_NUM_THREADS'],'omp_num_threads':os.environ['OMP_NUM_THREADS']}}
    def progress(n):report['completed_updates']+=n
    def save():
        report['cpu_seconds']=time.process_time()-cpu;report['wall_seconds']=time.perf_counter()-start
        (ROOT/'absolute_finite_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    try:
        # Finish all provenance/initial checks before adaptation.
        for old in audited['seeds']:
            seed=old['seed'];cp=SRC/f'seed_{seed}.npz';assert sha(cp)==old['checkpoint_sha256']
            assert sha(AUDIT/old['archive'])==old['archive_sha256']
            p=np.load(cp)['parameters'];cache=pred(p,x);blocks=np.load(AUDIT/old['archive'])
            errs={}
            for rule in ['stop','cached','input']:
                f=quantities(p,p,x,cache,rule)[0]
                errs[rule+'_saved_velocity_error']=float(np.max(np.abs(f-blocks[rule+'_velocity'])))
                errs[rule+'_audit_function_error']=float(np.max(np.abs(f-audit.field(p,p,x,cache,rule))))
            errs['initial_target_error']=float(np.max(np.abs(quantities(p,p,x,cache,'stop')[3]-quantities(p,p,x,cache,'cached')[3])))
            errs['initial_velocity_error']=float(np.max(np.abs(quantities(p,p,x,cache,'stop')[0]-quantities(p,p,x,cache,'cached')[0])))
            r0=float(np.mean(np.sum((pred(p,xe)-ye)**2,axis=1))/variance)
            oldbase=next(s for s in prior['seeds'] if s['seed']==seed)['no_adaptation_nmse']
            errs['no_adaptation_nmse_error']=abs(r0-oldbase)
            assert max(errs.values())<=1e-12,errs
            report['seeds'].append({'seed':seed,'primary':old['primary'],'checkpoint_sha256':old['checkpoint_sha256'],
                                    'initial_checks':errs,'no_adaptation_nmse':r0,'runs':[],'pair_records':[]})
        save()
        for s in report['seeds']:
            seed=s['seed'];p=np.load(SRC/f'seed_{seed}.npz')['parameters']
            for rules,label in [(['stop','cached'],'pair'),(['input'],'input')]:
                for tau in [None,1]:
                    base,arrays,pair=run_group(p,x,xe,ye,variance,rules,tau,.02,budget,progress)
                    report['completed_trajectories']+=len(rules)
                    fine,fa,fp=run_group(p,x,xe,ye,variance,rules,tau,.01,budget,progress)
                    report['completed_trajectories']+=len(rules)
                    out=ROOT/(f'seed{seed}_{label}_'+('fixed' if tau is None else 'tau1')+'.npz')
                    np.savez_compressed(out,**arrays,**{'fine_'+k:v for k,v in fa.items()})
                    for b,f in zip(base,fine):
                        b['halved_step']=f;b['archive']=out.name;b['archive_sha256']=sha(out)
                        b['final_nmse_step_difference']=abs(b['final']['nmse']-f['final']['nmse'])/max(abs(f['final']['nmse']),.01)
                        b['step_check_pass']=b['final_nmse_step_difference']<=.05
                        s['runs'].append(b)
                    if label=='pair':
                        s['pair_records'].append({'tau':tau,'main':pair,'fine':fp,'archive':out.name})
                    save()
                    print(json.dumps({'seed':seed,'rules':rules,'tau':tau,
                                      'main_nmse':[v['final']['nmse'] for v in base],
                                      'fine_nmse':[v['final']['nmse'] for v in fine],
                                      'step_difference':[v['final_nmse_step_difference'] for v in base],
                                      'events':pair.get('events')}),flush=True)
        allruns=[r for s in report['seeds'] for r in s['runs']]
        primary=[s for s in report['seeds'] if s['primary']]
        primary_steps=all(r['step_check_pass'] for s in primary for r in s['runs'] if r['rule'] in ['stop','cached'])
        secondary_steps=all(r['step_check_pass'] for s in primary for r in s['runs'] if r['rule']=='input')
        implementation=all(r.get('fixed_target_pass',True) and r['halved_step'].get('fixed_target_pass',True)
                           and r['fixed_coordinate_error']==0 and r['halved_step']['fixed_coordinate_error']==0 for r in allruns)
        implementation=implementation and all(pair[step]['checks']['pass'] for s in report['seeds'] for pair in s['pair_records'] for step in ['main','fine'])
        outcomes=[];success=[]
        for s in report['seeds']:
            original=next(r for r in s['runs'] if r['rule']=='stop' and r['tau']==1)
            cached=next(r for r in s['runs'] if r['rule']=='cached' and r['tau']==1)
            values=[];r0=s['no_adaptation_nmse']
            for a,b in [(original,cached),(original['halved_step'],cached['halved_step'])]:
                re,rc=a['final']['nmse'],b['final']['nmse']
                values.append({'eta':a['eta'],'R0':r0,'Re':re,'Rc':rc,
                               'nonnegative_excess_ratio':max(rc-r0,0)/(re-r0) if re>r0 else None,
                               'material_original_deterioration':bool(re>=2*r0 and re-r0>=.01),
                               'effect_inequalities_pass':bool(re>=2*r0 and re-r0>=.01 and max(rc-r0,0)<=.25*(re-r0))})
            passed=all(v['effect_inequalities_pass'] for v in values)
            outcomes.append({'seed':s['seed'],'primary':s['primary'],'outcomes':values,'both_steps_effect_inequalities_pass':passed})
            if s['primary'] and passed:success.append(s['seed'])
        numeric=implementation and primary_steps
        report['gates']={'implementation_checks_pass':implementation,'primary_step_checks_pass':primary_steps,
                         'secondary_primary_seed_step_checks_pass':secondary_steps,'primary_numerical_pass':numeric,
                         'individual_effect_outcomes':outcomes,'primary_effect_inequality_seeds':success,
                         'effect_count_pass_ignoring_numerical_gate':len(success)>=3,
                         'primary_gate_pass':numeric and len(success)>=3,
                         'decision':'numerically_unresolved' if not numeric else
                         ('large_absolute_loss_input_effect_supported' if len(success)>=3 else 'large_effect_did_not_replicate')}
        assert report['completed_updates']==90000 and report['completed_trajectories']==60
        report['status']='complete'
    except Exception as exc:
        report['status']='incomplete';report['failure']=f'{type(exc).__name__}: {exc}';save();raise
    save();print(json.dumps({'gates':report['gates'],'cpu_seconds':report['cpu_seconds'],'wall_seconds':report['wall_seconds']}),flush=True)
if __name__=='__main__':main()
