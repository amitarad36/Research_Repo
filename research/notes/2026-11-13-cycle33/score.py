"""Cycle33: score locked predictions against already observed targets."""
from pathlib import Path
import json,hashlib,math,time
import numpy as np
D=Path(__file__).resolve().parent;N=D.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for f,h in read(D/'prediction_lock.json').items():assert sha(D/f)==h
old0=read(N/'2026-11-01-cycle20/run/risk_results.json')
old123=read(N/'2026-11-07-cycle31/risk_results.json')
a=read(D/'locked_predictions.json');start=time.process_time()
result={'status':'complete','prediction_sha256':sha(D/'locked_predictions.json'),'cohort':[],'checks':{},'source_scores':{}}
for p in [N/'2026-11-01-cycle20/run/risk_results.json',N/'2026-11-07-cycle31/risk_results.json']:result['source_scores'][str(p)]=sha(p)
def sign(v):return 1 if v>1e-12 else (-1 if v<-1e-12 else 0)
for cell in a['seeds']:
 s=cell['seed'];prior=old0 if s==0 else old123['cohort'][s-1]
 y=np.array(prior['normalized_targets'] if s==0 else prior['targets']).reshape(40)
 base=np.array(cell['baseline']);q1=np.array(cell['q1']);q2=np.array(cell['q2'])
 mse=lambda q:float(math.fsum(float(v-yv)**2 for v,yv in zip(q,y))/40)
 b=mse(base);m1=[mse(q) for q in q1];m2=[mse(q) for q in q2]
 old=prior['mse_normalized'] if s==0 else prior['mse']
 assert abs(b-old['no_adaptation'])<1e-12 and abs(m2[0]-old['current-connected'])<1e-12
 err=max(abs(mse(q)-float(np.mean((q-y)**2))) for q in [base,*q1,*q2])
 result['checks']['independent_mse']=max(err,result['checks'].get('independent_mse',0))
 diffs=[v-b for v in m2]
 result['cohort'].append({'seed':s,'baseline_mse':b,'first_mse':dict(zip(a['arm_order'],m1)),'final_mse':dict(zip(a['arm_order'],m2)),
  'final_minus_baseline':dict(zip(a['arm_order'],diffs)),'percent_vs_baseline':dict(zip(a['arm_order'],[100*v/b for v in diffs])),
  'final_minus_carried':dict(zip(a['arm_order'],[v-m2[0] for v in m2])),'baseline_signs':[sign(v) for v in diffs],
  'targets':y.tolist(),'per_example_final_sqerrors':((q2-y)**2).tolist(),'per_example_baseline_sqerrors':((base-y)**2).tolist()})
result['all_state_arms_preserve_seed_sign']=all(len(set(c['baseline_signs']))==1 for c in result['cohort'])
result['uniform_reset_advantage']={k:all(c['final_minus_carried'][k]<-1e-12 for c in result['cohort']) for k in a['arm_order'][1:]}
result['cpu_seconds']=time.process_time()-start
assert result['checks']['independent_mse']<1e-12 and result['cpu_seconds']<10
(D/'risk_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({**result,'cohort':[{k:v for k,v in c.items() if k not in ('targets','per_example_final_sqerrors','per_example_baseline_sqerrors')} for c in result['cohort']]}))
