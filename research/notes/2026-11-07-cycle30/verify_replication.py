"""Independent NumPy forward/scaling/qualification audit, training rows only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,time
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--output",required=True)
HERE=Path(parser.parse_args().output).resolve()
SOURCE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.process_time();assert not (HERE/'verification_results.json').exists()
 report=json.loads((HERE/'training_results.json').read_text())
 assert report['status']=='complete' and report['training_steps']==450 and report['cpu_seconds']<=20
 assert report['held_out_values_parsed'] is False and report['adaptation_steps']==0
 for name,key in [('checkpoint.npz','checkpoint_sha256'),('preprocessing.npz','preprocessing_sha256')]:assert sha(HERE/name)==report[key]
 assert sha((Path(__file__).resolve().parents[3] / 'research/notes/2026-10-31-cycle17/bostonHousing.txt'))==report['data_sha256']
 assert sha(SOURCE/'train_replication.py')==report['script_sha256']
 assert sha(SOURCE/'REPLICATION_PROTOCOL.md')==report['protocol_sha256']
 weights=np.load(HERE/'checkpoint.npz',allow_pickle=False); prep=np.load(HERE/'preprocessing.npz',allow_pickle=False)
 shapes={'input.weight':(30,14),'input.bias':(30,),'hidden1.weight':(20,30),'hidden1.bias':(20,),
         'hidden2.weight':(10,20),'hidden2.bias':(10,),'output.weight':(1,10),'output.bias':(1,)}
 assert set(weights.files)==set(shapes)
 for key,shape in shapes.items():assert weights[key].shape==shape and weights[key].dtype==np.float64 and np.isfinite(weights[key]).all()
 assert sum(weights[k].size for k in weights.files)==1291
 assert np.array_equal(prep['train_rows'],np.arange(455)) and np.array_equal(prep['held_out_rows'],np.arange(455,506))
 lines=[l for l in ((Path(__file__).resolve().parents[3] / 'research/notes/2026-10-31-cycle17/bostonHousing.txt')).read_bytes().splitlines() if l.strip()]
 data=np.array([[float(v) for v in line.split()] for line in lines[:455]],dtype=np.float64)
 rawx,rawy=data[:,:13],data[:,13:]
 independent={'x_mean':rawx.mean(0),'x_std':rawx.std(0,ddof=1),'y_mean':rawy.mean(0),'y_std':rawy.std(0,ddof=1)}
 scaling_error={k:float(np.max(np.abs(prep[k]-v))) for k,v in independent.items()}
 assert all(np.allclose(prep[k],v,rtol=1e-12,atol=1e-12) for k,v in independent.items())
 x=(rawx-prep['x_mean'])/prep['x_std'];y=(rawy-prep['y_mean'])/prep['y_std']
 def forward(xx,z):
  a=np.concatenate([xx,z],axis=1)
  for layer in ['input','hidden1','hidden2']:
   a=np.maximum(a@weights[layer+'.weight'].T+weights[layer+'.bias'],0)
  return a@weights['output.weight'].T+weights['output.bias']
 q=forward(x,np.full_like(y,50));qy=forward(x,y);qq=forward(x,q)
 baseline=np.mean((y-y.mean(0))**2)
 metrics={'default_nmse':float(np.mean((q-y)**2)/baseline),'label_input_nmse':float(np.mean((qy-y)**2)/baseline),
          'prediction_std_ratio':float(q.std(ddof=0)/y.std(ddof=0)),
          'training_self_consistency_mse':float(np.mean((qq-q)**2)),'training_baseline_mse':float(baseline)}
 errors={k:abs(v-report['metrics'][k]) for k,v in metrics.items()}
 assert max(errors.values())<=1e-10
 gates={'complete':len(report['epochs'])==30 and all(e['minibatches']==15 for e in report['epochs']),
        'default_fit':metrics['default_nmse']<=.75,'label_input_fit':metrics['label_input_nmse']<=.25,
        'nonconstant':metrics['prediction_std_ratio']>=.10}
 assert gates==report['qualification'] and all(gates.values())==report['qualified']
 # Reconstruct the declared last-minibatch scheduler without Torch.
 best=float('inf');bad=0;lr=.001
 for epoch,e in enumerate(report['epochs'],1):
  assert e['epoch']==epoch and e['lr']==lr
  value=e['scheduler_last_minibatch_loss']
  if value<best*(1-1e-4):best=value;bad=0
  else:bad+=1
  if bad>10:
   new=lr*.5
   if lr-new>1e-8:lr=new
   bad=0
  assert e['lr_next']==lr
 result={'status':'passed','verifier_sha256':sha(Path(__file__)),'training_results_sha256':sha(HERE/'training_results.json'),
    'checkpoint_sha256':sha(HERE/'checkpoint.npz'),'preprocessing_sha256':sha(HERE/'preprocessing.npz'),
    'independent_metrics':metrics,'metric_absolute_errors':errors,'scaling_max_absolute_errors':scaling_error,
    'qualification':gates,'scheduler_trace_matches':True,'held_out_numeric_values_parsed':False,
    'extra_training_steps':0,'adaptation_steps':0,'cpu_seconds':time.process_time()-start}
 (HERE/'verification_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
