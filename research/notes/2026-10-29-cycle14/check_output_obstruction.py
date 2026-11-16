"""Exact rational necessary-condition test; no neural trajectories or labels."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json, hashlib, time, math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
HERE=Path(__file__).resolve().parent
NOTES=(Path(__file__).resolve().parents[3] / 'research/notes')
C3=NOTES/'2026-10-22-cycle3'; C7=NOTES/'2026-10-25-cycle7'; C8=NOTES/'2026-10-25-cycle8'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def sq(a): return sum((F(float(v))**2 for v in a.flat),F(0))
def rational(x): return {'numerator':str(x.numerator),'denominator':str(x.denominator)}
def main():
    start=time.process_time(); wall=time.perf_counter()
    assert not (HERE/'obstruction_results.json').exists()
    def cap(): assert time.process_time()-start<60, '60 CPU second cap'
    r7=json.loads((C7/'frozen_input_results.json').read_text())
    r8=json.loads((C8/'affine_prediction_results.json').read_text())
    assert sha(C7/'frozen_input_results.json')=='bec2eb7dab4713c4ba7910d5ec5bf40642e4346af7e816af01b2c584faf496ac'
    assert sha(C8/'affine_prediction_results.json')=='6e1b291b6c1550a1378d02382407cfaeaad51121dd7ff931e5a801959d50f563'
    for folder,r,script in [(C7,r7,'frozen_input_control.py'),(C8,r8,'affine_prediction.py')]:
        assert sha(folder/script)==r['script_sha256']
        assert sha(folder/'PROTOCOL.md')==r['protocol_sha256']
        assert sha(C3/'neural_geometry_gate.py')==r['source_model_sha256']
        assert sha(C3/'shared_data.npz')==r['source_data_sha256']
    verified={}
    for name,digest in r8['input_archive_hashes'].items():
        assert sha(C7/name)==digest; verified[name]=digest
    for s in r7['seeds']:
        for item in [s['initial_checks']]+s['runs']:
            assert verified[item['archive']]==item['archive_sha256']
    data=np.load(C3/'shared_data.npz'); xe=data['x_eval'][64:]
    assert xe.shape==(192,2)
    xf=[[F(float(v)) for v in row] for row in xe]
    # cosh(1) <= 3/2+(1/24)/(1-1/30) < 8/5, hence sech^2 >=25/64.
    assert F(3,2)+F(1,24)/(1-F(1,30))<F(8,5)
    report={'protocol_sha256':sha(HERE/'OUTPUT_OBSTRUCTION_PROTOCOL.md'),
        'parent_protocol_sha256':sha(NOTES/'2026-10-29-cycle13/NEXT_CERTIFICATE_PROTOCOL.md'),
        'script_sha256':sha(Path(__file__)), 'verified_cycle7_archives':verified,
        'source_data_sha256':sha(C3/'shared_data.npz'),'source_model_sha256':sha(C3/'neural_geometry_gate.py'),
        'labels_loaded':False,'new_nonlinear_updates':0,'seeds':[]}
    for s in r8['seeds']:
        cap(); seed=s['seed']; cp=C3/f'seed_{seed}.npz'; ap=C8/s['archive']
        assert sha(cp)==s['checkpoint_sha256']; assert sha(ap)==s['archive_sha256']
        p=np.load(cp)['parameters']; a=np.load(ap)
        assert np.array_equal(a['active'],np.r_[0:32,64:114])
        pf=[F(float(v)) for v in p]
        counts=[]
        for j in range(16):
            counts.append(sum(abs(xx[0]*pf[j]+xx[1]*pf[16+j]+pf[64+j])<=1 for xx in xf))
        m=max(counts); hsq=F(25,64)**2*F(m,384)
        entry={'seed':seed,'primary':s['primary'],'checkpoint_sha256':sha(cp),'forecast_sha256':sha(ap),
               'counts_abs_preactivation_at_most_one':counts,'witness_unit':counts.index(m),
               'hessian_bound_required_at_least':math.sqrt(float(hsq)), 'hessian_lower_squared':rational(hsq),'runs':[]}
        for tag,eta in [('main',.02),('fine',.01)]:
            tt=a[tag+'_time']; indices=np.flatnonzero(np.abs(tt-2)<1e-12); assert len(indices)==1
            ix=int(indices[0]); us=a[tag+'_stop_state'][ix]; uc=a[tag+'_cached_state'][ix]
            assert us.shape==uc.shape==(164,)
            pred=a[tag+'_predicted_output_difference'][ix]; assert pred.shape==(192,2)
            total=sq(us)+sq(uc); sep2=sq(pred)/384
            floor2=hsq*total**2/4; allowed2=sep2/25
            assert sep2>0
            entry['runs'].append({'eta':eta,'T':2,'sum_displacement_squared':float(total),
                'prediction_rms_per_scalar':math.sqrt(float(sep2)),
                'mandatory_B_lower':math.sqrt(float(floor2)),
                'mandatory_B_over_prediction':math.sqrt(float(floor2/sep2)),
                'required_ratio_at_most':.2,'obstructed':floor2>allowed2,
                'exact_squared_margin':rational(floor2-allowed2),
                'sum_displacement_squared_exact':rational(total), 'prediction_squared_exact':rational(sep2)})
        report['seeds'].append(entry)
    cap()
    report['primary_unobstructed_seeds']=[s['seed'] for s in report['seeds'] if s['primary'] and not any(r['obstructed'] for r in s['runs'])]
    report['primary_gate_necessarily_rejected']=len(report['primary_unobstructed_seeds'])<3
    report['cpu_seconds']=time.process_time()-start; report['wall_seconds']=time.perf_counter()-wall
    (HERE/'obstruction_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'seeds':[{k:v for k,v in s.items() if k in ['seed','primary','hessian_bound_required_at_least']}|{'runs':[{k:r[k] for k in ['eta','mandatory_B_over_prediction','obstructed']} for r in s['runs']]} for s in report['seeds']], 'gate_rejected':report['primary_gate_necessarily_rejected'],'cpu_seconds':report['cpu_seconds']},indent=2))
if __name__=='__main__': main()
