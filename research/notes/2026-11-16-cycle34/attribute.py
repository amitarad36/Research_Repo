"""Cycle34. Read-only first-step Adam attribution; no model or target access."""
import os
for k in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,time,math
import numpy as np
D=Path(__file__).resolve().parent;N=D.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def norm(x):return float(np.linalg.norm(x))
def fit(h,f):
 den=float(f@f)
 scale=max(0.,float(h@f)/den) if den>1e-28 else 0.
 r=h-scale*f
 return dict(scale=scale,residual_norm=norm(r),relative=norm(r)/max(norm(h),1e-14),reference_norm=norm(f),
             direction_event=bool(norm(r)>1e-6*max(norm(h),1e-12)),reference_degenerate=den<=1e-28)
def main():
 for p,h in read(D/'registered.json').items():assert sha(Path(p))==h,p
 start=time.process_time();report={'status':'complete','new_updates':0,'new_model_calls':0,'target_access':False,'checks':{},'records':[]}
 def ck(key,a,b):
  a=np.asarray(a);b=np.asarray(b);assert np.isfinite(a).all() and np.isfinite(b).all()
  e=float(np.max(np.abs(a-b)));assert e<=1e-10*max(1,float(np.max(abs(a))),float(np.max(abs(b)))),(key,e)
  report['checks'][key]=max(e,report['checks'].get(key,0))
 a=np.load(N/'2026-11-13-cycle33/intervention_trace.npz',allow_pickle=False)
 for seed in range(4):
  p=N/('2026-11-01-cycle19/run/trace.npz' if seed==0 else '2026-11-07-cycle31/seed_'+str(seed)+'/run/trace.npz')
  old=np.load(p,allow_pickle=False);prefix='seed'+str(seed)+'_'
  data={k:a[prefix+k] for k in ['before_theta','after_theta','before_m','before_v','before_step','g','baseline','q1']}
  for b in range(20):
   assert time.process_time()-start<10
   ids=[6*b,6*b+2,6*b+4];g=data['g'][ids[0]];m=data['before_m'][ids[0]];v=data['before_v'][ids[0]]
   n=2*b+1
   for arm,idx in enumerate(ids):
    ck('common_gradient',g,data['g'][idx]);ck('common_weights',data['before_theta'][ids[0]],data['before_theta'][idx])
    ck('counter',data['before_step'][idx],np.full(8,n-1 if arm<2 else 0))
   ck('original_gradient',g,old['g'][8*b])
   J=old['J'][8*b];assert J.shape==(2,1291)
   V=.999*v+.001*g*g;Vz=.001*g*g
   DH=-1e-4/(1-.9**n)/(np.sqrt(V/(1-.999**n))+1e-8)
   DZ=-1e-4/(1-.9**n)/(np.sqrt(Vz/(1-.999**n))+1e-8)
   uH=DH*(.9*m+.1*g);uZ=DZ*(.1*g);uF=-1e-4*g/(np.abs(g)+1e-8)
   u=[uH,uZ,uF]
   for arm,idx in enumerate(ids):ck('formula_step',u[arm],data['after_theta'][idx]-data['before_theta'][idx])
   # Explicit ordering: add old m using historical denominator, then change denominator for current g.
   term_m=DH*(.9*m);term_v=(DH-DZ)*(.1*g)
   ck('decomposition',uH-uZ,term_m+term_v)
   ck('projected_decomposition',J@(uH-uZ),J@term_m+J@term_v)
   # Scalar math.fsum checks formula accumulation and the least-squares solution.
   hf=math.fsum(float(x)*float(y) for x,y in zip(uH,uF))
   ck('dot_product',hf,float(uH@uF))
   an=.1/(1-.9**n);bn=np.sqrt(.001/(1-.999**n));scale0=an/bn
   # Epsilon is retained; this expression is the exact deviation from the epsilon-zero scalar relation.
   correction=-1e-4*an*g*1e-8*(bn-1)/(bn*(bn*np.abs(g)+1e-8)*(np.abs(g)+1e-8))
   ck('counter_epsilon_correction',uZ-scale0*uF,correction)
   q0=data['baseline'][b].reshape(2);outs=data['q1'][3*b:3*b+3].reshape(3,2)-q0
   ck('baseline_archive',q0,old['noadapt'][b].reshape(2))
   raw=data['after_theta'][ids]-data['before_theta'][ids]
   rec={'seed':seed,'batch':b,'counter_after':n,'parameter_H_vs_F':fit(uH,uF),'parameter_H_vs_Z':fit(uH,uZ),
        'parameter_Z_vs_F':fit(uZ,uF),'linear_output_H_vs_F':fit(J@uH,J@uF),'actual_output_H_vs_F':fit(outs[0],outs[2]),
        'counter_scale_epsilon_zero':float(scale0),'counter_scaling_relative_error':norm(correction)/max(norm(uZ),1e-14),
        'history_delta_norm':norm(uH-uZ),'term_m_norm':norm(term_m),'term_v_norm':norm(term_v),
        'term_dot':float(term_m@term_v),'J_term_m_norm':norm(J@term_m),'J_term_v_norm':norm(J@term_v),
        'inactive_gradient_history_coordinates':int(np.count_nonzero((g==0)&(m!=0))),
        'nonlinear_remainder_norms':[norm(outs[k]-J@u[k]) for k in range(3)]}
   for key,ha,fa in [('parameter_H_vs_F',raw[0],raw[2]),('parameter_H_vs_Z',raw[0],raw[1]),('parameter_Z_vs_F',raw[1],raw[2])]:
    ck('saved_fit_scale',rec[key]['scale'],fit(ha,fa)['scale'])
    ck('saved_fit_residual',rec[key]['residual_norm'],fit(ha,fa)['residual_norm'])
   if b==0:
    ck('first_batch_equal',uH,uF);ck('first_batch_zero_components',term_m+term_v,np.zeros(1291))
   report['records'].append(rec)
 report['summary']=[]
 for s in range(4):
  rows=[r for r in report['records'] if r['seed']==s]
  sums={'seed':s}
  for k in ['parameter_H_vs_F','parameter_H_vs_Z','parameter_Z_vs_F','linear_output_H_vs_F','actual_output_H_vs_F']:
   sums[k]={'events':sum(r[k]['direction_event'] for r in rows),
    'median_relative_nonfirst':float(np.median([r[k]['relative'] for r in rows[1:]])),
    'min_relative_nonfirst':min(r[k]['relative'] for r in rows[1:]),
    'max_relative':max(r[k]['relative'] for r in rows)}
  sums['max_counter_scaling_relative_error']=max(r['counter_scaling_relative_error'] for r in rows)
  report['summary'].append(sums)
 assert len(report['records'])==80
 report['cpu_seconds_after_import']=time.process_time()-start;assert report['cpu_seconds_after_import']<10
 (D/'attribution_results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 print(json.dumps({k:v for k,v in report.items() if k!='records'}))
if __name__=='__main__':main()
