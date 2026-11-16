"""One fixed derivative envelope; exact rational budget comparisons."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,math
import numpy as np
HERE=Path(__file__).resolve().parent
NOTES=(Path(__file__).resolve().parents[3] / 'research/notes')
C3=NOTES/'2026-10-22-cycle3'; C8=NOTES/'2026-10-25-cycle8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sq(a):return sum((F(float(v))**2 for v in a.flat),F(0))
def up_sqrt(x):
    # Exact integer arithmetic, no reliance on floating-point sqrt rounding.
    v=x*1024**2; n=math.isqrt(v.numerator//v.denominator)
    if F(n*n)<v:n+=1
    assert F(n,1024)**2>=x
    return F(n,1024)
def rat(x):return [str(x.numerator),str(x.denominator)]
def main():
    assert not (HERE/'enclosure_results.json').exists()
    start=time.process_time(); wall=time.perf_counter()
    prior=json.loads((HERE/'obstruction_results.json').read_text())
    data=np.load(C3/'shared_data.npz'); xx=data['x_eval']; assert sha(C3/'shared_data.npz')==prior['source_data_sha256']
    l=up_sqrt(max(F(1)+sq(x) for x in xx[:64])); le=up_sqrt(max(F(1)+sq(x) for x in xx[64:]))
    rows=[]
    for old in prior['seeds']:
        seed=old['seed']; cp=C3/f'seed_{seed}.npz'; ap=C8/f'seed{seed}_affine_prediction.npz'
        assert sha(cp)==old['checkpoint_sha256']; assert sha(ap)==old['forecast_sha256']
        p=np.load(cp)['parameters']; a=np.load(ap)
        v0=up_sqrt(sq(p[80:112])); b0=up_sqrt(sq(p[112:])); k=up_sqrt(sq(p[32:64]))
        for tag,eta in [('main',F(1,50)),('fine',F(1,100))]:
            steps=int(2/eta); end={}; prev={}; allnorms=[]; max_endpoint_error=0.
            for arm in ['stop','cached']:
                mat=a[f'{tag}_{arm}_matrix']; c=a[f'{tag}_{arm}_forcing']; u=np.zeros(164)
                for n in range(steps+1):
                    allnorms.append(sq(u))
                    if n==steps-1:prev[arm]=u.copy()
                    if n<steps:u=mat@u+c
                end[arm]=u
                max_endpoint_error=max(max_endpoint_error,float(np.max(np.abs(u-a[f'{tag}_{arm}_state'][5]))))
            assert max_endpoint_error<1e-12
            # Radius fixed by the actual declared floating-point affine recurrence.
            r2=4*max(allnorms); rb=up_sqrt(r2); v=v0+rb; b=b0+rb
            q0=4*v+b; q1=v*l+5; q2=v*l*l+2*l; q3=2*v*l**3+3*l*l
            aa=up_sqrt(l*l+k*k); t1=v*aa+5; t2=v*aa*aa+2*aa; g1=max(F(1),q1)
            beta=eta/(1+eta); constants={}; state_floor_sum=F(0)
            for arm in ['stop','cached']:
                r0=2*q0
                r1=t1*g1+q1 if arm=='stop' else 2*q1
                rr2=t2*g1*g1+(t1+1)*q2 if arm=='stop' else 2*q2
                f2=q3*r0+2*q2*r1+q1*rr2
                L=eta*(1+beta)*f2
                e_floor=L*sq(prev[arm])/2
                state_floor_sum+=e_floor
                constants[arm]={'F_second_derivative_upper':float(f2),'L_map_upper':float(L),
                    'L_exact':rat(L),'last_step_state_enclosure_floor':float(e_floor)}
            h2=(v*le*le+2*le)**2/2
            total=sq(end['stop'])+sq(end['cached'])
            outfloor2=h2*total**2/4; statefloor2=state_floor_sum**2/2
            pred=a[tag+'_predicted_output_difference'][5]; sep2=sq(pred)/384
            allowed2=sep2/25
            rows.append({'seed':seed,'primary':old['primary'],'eta':float(eta),'radius':math.sqrt(float(r2)),
                'outward_radius':float(rb),'radius_squared_exact':rat(r2),'outward_radius_exact':rat(rb),
                'H_output_upper':math.sqrt(float(h2)), 'H_squared_exact':rat(h2),
                'constants':constants,'max_replay_endpoint_difference':max_endpoint_error,
                'output_only_B_over_prediction':math.sqrt(float(outfloor2/sep2)),
                'state_term_B_over_prediction_at_least':math.sqrt(float(statefloor2/sep2)),
                'output_budget_failed':outfloor2>allowed2,'state_budget_failed':statefloor2>allowed2,
                'output_exact_squared_margin':rat(outfloor2-allowed2),
                'state_exact_squared_margin':rat(statefloor2-allowed2)})
            assert time.process_time()-start+prior['cpu_seconds']<60
    report={'recipe_sha256':sha(HERE/'ENCLOSURE_RECIPE.md'),'script_sha256':sha(Path(__file__)),
        'lower_test_results_sha256':sha(HERE/'obstruction_results.json'),'rows':rows,
        'labels_loaded':False,'new_nonlinear_updates':0,'training':False,
        'primary_recipe_success_seeds':[s for s in [0,2,3,4] if not any(r['output_budget_failed'] or r['state_budget_failed'] for r in rows if r['seed']==s)],
        'cpu_seconds':time.process_time()-start,'wall_seconds':time.perf_counter()-wall,
        'cpu_seconds_including_first_test':time.process_time()-start+prior['cpu_seconds']}
    (HERE/'enclosure_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'rows':[{k:r[k] for k in ['seed','eta','radius','output_only_B_over_prediction','state_term_B_over_prediction_at_least','output_budget_failed','state_budget_failed']} for r in rows], 'cpu_seconds_total':report['cpu_seconds_including_first_test']},indent=2))
if __name__=='__main__':main()
