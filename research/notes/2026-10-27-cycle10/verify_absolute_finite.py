"""Saved-data verification only: no adaptation rerun or step selection."""
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SRC=(Path(__file__).resolve().parents[3] / 'research/notes/2026-10-22-cycle3')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forward(p,x,z):
    u=np.concatenate((x,z),axis=1);h=np.tanh(u@p[:64].reshape(4,16)+p[64:80])
    return h@p[80:112].reshape(16,2)+p[112:],(u,h)
def explicit_field(s,t,x,c,rule):
    q,(u,h)=forward(s,x,np.zeros_like(x));target,_=forward(t,x,c if rule=='cached' else q)
    res=target-q;sig=np.sign(res);f=np.zeros(114);w2=s[80:112].reshape(16,2)
    for i in range(len(x)):
        for o in range(2):
            dh=(1-h[i]**2)*w2[:,o];dw2=np.zeros((16,2));dw2[:,o]=h[i]
            jac=np.r_[np.outer(u[i],dh).ravel(),dh,dw2.ravel(),np.eye(2)[o]]
            f+=sig[i,o]*jac/(len(x)*2)
    return f,res,sig
def main():
    start=time.process_time();r=json.loads((ROOT/'absolute_finite_results.json').read_text())
    assert r['status']=='complete' and r['completed_updates']==90000 and r['completed_trajectories']==60
    for path,key in [(ROOT/'absolute_finite_response.py','script_sha256'),(ROOT/'PROTOCOL.md','protocol_sha256'),(SRC/'shared_data.npz','source_data_sha256'),(SRC/'neural_geometry_gate.py','source_model_sha256')]:assert sha(path)==r[key]
    d=np.load(SRC/'shared_data.npz');x=d['x_eval'][:64];xe=d['x_eval'][64:];ye=d['y_eval'][64:]
    var=float(np.mean(np.sum((ye-ye.mean(0))**2,axis=1)));assert var==r['evaluation_label_variance']
    errors={k:0. for k in ['metrics','targets','cache','inactive','event_field','event_residual','first_kick','reference_update']}
    def check(key,a,b):
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));errors[key]=max(errors[key],error);assert error<=1e-12,(key,error)
    metrics=events=zeros=0;files={};onsets=[];failed_primary=[];failed_secondary=[];effects=[];all_cached_above=True
    for seed in r['seeds']:
        sn=seed['seed'];cp=SRC/f'seed_{sn}.npz';assert sha(cp)==seed['checkpoint_sha256']
        p=np.load(cp)['parameters'];c=forward(p,x,np.zeros_like(x))[0];target0=forward(p,x,c)[0]
        q0=forward(p,xe,np.zeros_like(xe))[0];check('metrics',np.mean(np.sum((q0-ye)**2,axis=1))/var,seed['no_adaptation_nmse'])
        for base in seed['runs']:
            name=base['archive'];assert sha(ROOT/name)==base['archive_sha256'];files[name]=sha(ROOT/name)
            a=np.load(ROOT/name);rule=base['rule']
            for prefix,run in [('',base),('fine_',base['halved_step'])]:
                ss=a[prefix+rule+'_state'];tt=a[prefix+rule+'_target'];cc=a[prefix+rule+'_cache']
                assert ss.shape==(51,2,114) and tt.shape==(51,64,2) and cc.nbytes==1024
                check('cache',cc,c);check('inactive',ss[:,:,32:64],p[None,None,32:64])
                assert run['history'][0]==run['initial'] and run['history'][-1]==run['final']
                for k,hist in enumerate(run['history']):
                    s,t=ss[k];q=forward(s,x,np.zeros_like(x))[0];target=forward(t,x,cc if rule=='cached' else q)[0]
                    qe=forward(s,xe,np.zeros_like(xe))[0];check('targets',tt[k],target)
                    expected=[a[prefix+'time'][k],np.mean(np.sum((qe-ye)**2,axis=1))/var,np.mean(np.abs(target-q)),np.linalg.norm(s-t),np.sqrt(np.mean(np.sum((target-target0)**2,axis=1))),np.max(np.abs(target-target0)),np.count_nonzero(target-q==0)]
                    keys=['time','nmse','discrepancy','reference_lag','reference_target_change_rms','reference_target_change_max_abs','zero_residual_count']
                    check('metrics',expected,[hist[key] for key in keys]);metrics+=1
                if rule=='cached' and base['tau'] is None:check('targets',tt,target0);assert run['fixed_target_pass'] and run['fixed_target_max_abs']==0
                if rule=='cached' and base['tau']==1:all_cached_above=all_cached_above and run['final']['nmse']>seed['no_adaptation_nmse']
                zeros+=run['zero_residual_observations']
            diff=abs(base['final']['nmse']-base['halved_step']['final']['nmse'])/max(abs(base['halved_step']['final']['nmse']),.01)
            check('metrics',diff,base['final_nmse_step_difference']);assert base['step_check_pass']==(diff<=.05)
            if seed['primary'] and not base['step_check_pass']:(failed_secondary if rule=='input' else failed_primary).append({'seed':sn,'rule':rule,'tau':base['tau'],'relative_step_difference':diff})
        for pair in seed['pair_records']:
            a=np.load(ROOT/pair['archive'])
            for prefix,key in [('', 'main'),('fine_','fine')]:
                pr=pair[key];ps=a[prefix+'per_step'];eta=pr['eta'];tau=pr['tau'];assert len(ps)==round(20/eta)+1
                np.testing.assert_array_equal(ps[:,0],np.arange(len(ps))*eta)
                selected={'first_sign_disagreement':ps[:,3]>0,'first_velocity_disagreement':ps[:,7]>1e-12,'first_state_disagreement':ps[:,6]>1e-12}
                for event,mask in selected.items():
                    indices=np.flatnonzero(mask);ev=pr['events'][event];assert len(indices)>0 and ev['step']==indices[0] and ev['time']==indices[0]*eta
                    state=a[prefix+event+'_state'];field=a[prefix+event+'_field'];residual=a[prefix+event+'_residual'];sign=a[prefix+event+'_sign']
                    for arm,rule in enumerate(['stop','cached']):
                        f,res,sig=explicit_field(*state[arm],x,c,rule);check('event_field',f,field[arm]);check('event_residual',res,residual[arm]);np.testing.assert_array_equal(sig,sign[arm])
                    check('metrics',np.max(np.abs(state[0]-state[1])),ev['state_max_abs_difference']);check('metrics',np.max(np.abs(field[0]-field[1])),ev['velocity_max_abs_difference'])
                    assert np.count_nonzero(sign[0]!=sign[1])==ev['sign_disagreement_count'];events+=1
                se=pr['events']['first_sign_disagreement'];ve=pr['events']['first_velocity_disagreement'];ste=pr['events']['first_state_disagreement']
                assert se['step']==ve['step'] and ste['step']==ve['step']+1
                ss=a[prefix+'first_velocity_disagreement_state'];ff=a[prefix+'first_velocity_disagreement_field'];after=a[prefix+'first_state_disagreement_state'];alpha=1. if tau is None else tau/(tau+eta)
                np.testing.assert_array_equal(ss[0],ss[1]);check('first_kick',after[:,0],ss[:,0]+eta*ff);check('first_kick',after[0,0]-after[1,0],eta*(ff[0]-ff[1]))
                expected=alpha*ss[:,1]+(1-alpha)*after[:,0];expected[:,32:64]=p[32:64];check('reference_update',after[:,1],expected)
                assert not np.any(ps[:se['step'],3]) and not np.any(ps[:ste['step'],6])
                assert pr['checks']['pass'] and pr['checks']['max_velocity_difference_under_shared_state_and_sign']==0 and pr['checks']['max_next_state_difference_under_shared_state_and_sign']==0
                onsets.append({'seed':sn,'primary':seed['primary'],'tau':tau,'eta':eta,'first_sign_time':se['time'],'first_sign_step':se['step'],'first_state_time':ste['time'],'sign_disagreements_at_onset':se['sign_disagreement_count'],'total_sign_mismatches':int(np.sum(ps[:,3])),'sign_changes_stop':int(np.sum(ps[:,1])),'sign_changes_cached':int(np.sum(ps[:,2]))})
        orig=next(v for v in seed['runs'] if v['rule']=='stop' and v['tau']==1);cached=next(v for v in seed['runs'] if v['rule']=='cached' and v['tau']==1);r0=seed['no_adaptation_nmse'];flags=[]
        for a,b in [(orig,cached),(orig['halved_step'],cached['halved_step'])]:
            re,rc=a['final']['nmse'],b['final']['nmse'];flags.append(re>=2*r0 and re-r0>=.01 and max(rc-r0,0)<=.25*(re-r0))
        effects.append({'seed':sn,'primary':seed['primary'],'eta_02_pass':bool(flags[0]),'eta_01_pass':bool(flags[1])})
    assert metrics==3060 and events==60 and len(files)==20
    supporting=[e['seed'] for e in effects if e['primary'] and e['eta_02_pass'] and e['eta_01_pass']]
    assert supporting==r['gates']['primary_effect_inequality_seeds'] and r['gates']['implementation_checks_pass']
    assert failed_primary and not r['gates']['primary_numerical_pass'] and r['gates']['decision']=='numerically_unresolved'
    out={'status':'passed','results_sha256':sha(ROOT/'absolute_finite_results.json'),'verifier_sha256':sha(Path(__file__)),'metric_records_checked':metrics,'event_snapshots_checked':events,'archives_checked':len(files),'archive_sha256':files,'max_errors':errors,'failed_primary_step_checks':failed_primary,'failed_secondary_step_checks':failed_secondary,'effect_inequalities':effects,'primary_effect_inequality_seeds':supporting,'decision':'numerically_unresolved','all_cached_ema_final_nmse_above_initial':bool(all_cached_above),'exact_zero_residual_observations':zeros,'onset_records':onsets,'cpu_seconds':time.process_time()-start}
    (ROOT/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['archive_sha256','onset_records']},indent=2));print(json.dumps({'onsets':onsets},indent=2))
if __name__=='__main__':main()
