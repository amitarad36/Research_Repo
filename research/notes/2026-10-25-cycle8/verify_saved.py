"""Independent artifact audit, no new neural training or adaptation updates."""
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
NOTES=ROOT.parent
if not (NOTES/'2026-10-25-cycle7').exists():
    NOTES=(Path(__file__).resolve().parents[3] / 'research/notes')
SRC=NOTES/'2026-10-22-cycle3'; PREV=NOTES/'2026-10-25-cycle7'
spec=importlib.util.spec_from_file_location('model',SRC/'neural_geometry_gate.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pred(p,x):return model.forward(p,x,np.zeros_like(x))[0]
def rms(v):return float(np.linalg.norm(v)/np.sqrt(len(v)))
report=json.loads((ROOT/'affine_prediction_results.json').read_text())
assert report['status']=='complete' and report['new_nonlinear_updates']==0 and not report['labels_used']
assert report['script_sha256']==sha(ROOT/'affine_prediction.py')
assert report['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
xe=np.load(SRC/'shared_data.npz')['x_eval'][64:]
max_actual_error=max_prediction_error=max_metric_error=max_map_error=0.
rows=[]; count=0; independently_passing=[]
for s in report['seeds']:
    seed=s['seed'];a=np.load(ROOT/s['archive'])
    assert sha(ROOT/s['archive'])==s['archive_sha256']
    passing=[]
    for tag,prefix,run in [('main','',s['runs'][0]),('fine','fine_',s['runs'][1])]:
        arrays={arm:np.load(PREV/f'seed{seed}_{arm}_tau1.npz') for arm in ['stop','cached']}
        times=arrays['stop'][prefix+'time']
        assert np.array_equal(times,a[tag+'_time'])
        uo=a[tag+'_stop_state'][:,:82]; uc=a[tag+'_cached_state'][:,:82]
        # Explicit per-input multiply, independent of the analysis einsum.
        predicted=np.array([[ji@(u-v) for ji in a['evaluator_jacobian']] for u,v in zip(uo,uc)])
        real=np.array([pred(p[0],xe)-pred(q[0],xe) for p,q in
                       zip(arrays['stop'][prefix+'state'],arrays['cached'][prefix+'state'])])
        max_actual_error=max(max_actual_error,float(np.max(np.abs(real-a[tag+'_actual_output_difference']))))
        max_prediction_error=max(max_prediction_error,float(np.max(np.abs(predicted-a[tag+'_predicted_output_difference']))))
        for arm in ['stop','cached']:
            m=a[tag+'_'+arm+'_matrix'];b=a[tag+'_'+arm+'_forcing']
            eta=run['eta'];alpha=1/(1+eta);s0=a[arm+'_student'];t0=a[arm+'_teacher'];f=a['initial_velocity']
            # Read block coefficients back from the assembled map.
            errs=[np.max(np.abs((m[:82,:82]-np.eye(82))/eta-s0)),
                  np.max(np.abs(m[:82,82:]/eta-t0)),
                  np.max(np.abs(m[82:,:82]-(1-alpha)*m[:82,:82])),
                  np.max(np.abs(m[82:,82:]-alpha*np.eye(82)-(1-alpha)*m[:82,82:])),
                  np.max(np.abs(b[:82]-eta*f)),np.max(np.abs(b[82:]-(1-alpha)*eta*f))]
            max_map_error=max(max_map_error,float(max(errs)))
        for index,(tm,rd,pd,h) in enumerate(zip(times,real,predicted,run['history'])):
            assert tm==h['time']
            magnitude=rms(rd); floor=1e-8*max(rms(a['q0']),1.)
            rel=rms(pd-rd)/max(magnitude,floor)
            max_metric_error=max(max_metric_error,abs(rel-h['relative_prediction_error']))
            count+=1
            if tm==2:
                passing.append(bool(magnitude>floor and rel<=.25))
        rows.append({'seed':seed,'primary':s['primary'],'eta':run['eta'],
                     'T2_error_percent':100*run['primary_T2']['relative_prediction_error'],
                     'T2_actual_rms':run['primary_T2']['actual_separation_rms'],
                     'T2_predicted_rms':run['primary_T2']['predicted_separation_rms'],
                     'T2_cosine':run['primary_T2']['cosine'],
                     'first_saved_error_above_25_percent':run['first_saved_time_above_25_percent_error'],
                     'T20_error_percent':100*run['history'][-1]['relative_prediction_error']})
    assert all(passing)==s['both_steps_effect_pass']
    if s['primary'] and all(passing):independently_passing.append(seed)
assert count==510 and max_actual_error==0 and max_map_error<1e-12
assert max_prediction_error<1e-10 and max_metric_error<1e-10
assert independently_passing==report['gates']['primary_success_seeds']
allruns=[r for s in report['seeds'] for r in s['runs']]
summary={'verifier_sha256':sha(Path(__file__)),'results_sha256':sha(ROOT/'affine_prediction_results.json'),
         'records_checked':count,'max_actual_array_difference':max_actual_error,
         'max_predicted_array_difference':max_prediction_error,'max_relative_metric_difference':max_metric_error,
         'max_map_coefficient_error':max_map_error,
         'max_evaluator_jacobian_error':max(s['initial_checks']['evaluator_jacobian_fd_max_abs'] for s in report['seeds']),
         'max_independent_propagation_relative_error':max(c['max_relative_state_difference'] for r in allruns for c in r['checks']['propagation'].values()),
         'max_second_step_identity_error':max(r['checks']['second_step_identity_max_abs'] for r in allruns),
         'primary_success_seeds':independently_passing,'table':rows,
         'cpu_seconds':report['cpu_seconds'],'wall_seconds':report['wall_seconds'],
         'initial_attempt_failure':json.loads((ROOT/'initial_attempt_failure.json').read_text())['failure']}
(ROOT/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
