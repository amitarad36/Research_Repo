"""Prepared cycle-17 reproduction. Run only under TRAINING_PROTOCOL.md."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import time
CPU0=time.process_time()
import argparse,json,hashlib,sys,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
 parser.add_argument("--seed",type=int,required=True,choices=(1,2,3))
 args=parser.parse_args();seed=args.seed
 out=Path(args.output).resolve();assert not out.exists();out.mkdir(parents=True)
 report={'status':'running','training_steps':0,'held_out_values_parsed':False,
         'data_sha256':sha((Path(__file__).resolve().parents[3] / 'research/notes/2026-10-31-cycle17/bostonHousing.txt')),'protocol_sha256':sha(HERE/'REPLICATION_PROTOCOL.md'),
         'script_sha256':sha(Path(__file__)),'python':sys.executable,'epochs':[]}
 model=None;scales=None
 def cap():
  if time.process_time()-CPU0>20:raise TimeoutError('Prespecified 20 CPU second cap')
 def save():
  report['cpu_seconds']=time.process_time()-CPU0
  (out/'training_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
 try:
  import numpy as np
  import torch
  from torch import nn
  torch.set_num_threads(1)
  assert torch.__version__=='2.9.0+cpu', 'Runtime changed; re-audit rather than substitute'
  assert report['data_sha256']=='baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a'
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  lines=[l for l in ((Path(__file__).resolve().parents[3] / 'research/notes/2026-10-31-cycle17/bostonHousing.txt')).read_bytes().splitlines() if l.strip()]
  assert len(lines)==506
  assert all(len(l.split())==14 for l in lines) # token counts only, no held-out numeric parsing
  arr=np.array([[float(v) for v in l.split()] for l in lines[:455]],dtype=np.float64)
  assert np.isfinite(arr).all()
  x=torch.from_numpy(arr[:,:13].copy());y=torch.from_numpy(arr[:,13:].copy())
  xm=x.mean(0);xs=x.std(0,correction=1);ym=y.mean(0);ys=y.std(0,correction=1)
  assert torch.isfinite(xs).all() and (xs>0).all() and torch.isfinite(ys).all() and (ys>0).all()
  x=(x-xm)/xs;y=(y-ym)/ys
  scales={'x_mean':xm.numpy(),'x_std':xs.numpy(),'y_mean':ym.numpy(),'y_std':ys.numpy(),
          'train_rows':np.arange(455),'held_out_rows':np.arange(455,506)}
  class Model(nn.Module):
   def __init__(self):
    super().__init__()
    self.input=nn.Linear(14,30);self.hidden1=nn.Linear(30,20)
    self.hidden2=nn.Linear(20,10);self.output=nn.Linear(10,1)
    for layer in self.modules():
     if isinstance(layer,nn.Linear):nn.init.zeros_(layer.bias)
   def forward(self,xx,z=None):
    if z is None:z=xx.new_full((len(xx),1),50.)
    a=torch.cat([xx,z],1)
    return self.output(torch.relu(self.hidden2(torch.relu(self.hidden1(torch.relu(self.input(a)))))))
  model=Model().double()
  opt=torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
  sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=10,
      threshold=1e-4,threshold_mode='rel',cooldown=0,min_lr=0,eps=1e-8)
  loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=32,shuffle=True,
      generator=torch.Generator().manual_seed(0),num_workers=0,drop_last=False)
  assert len(loader)==15
  report.update({'torch_version':torch.__version__,'numpy_version':np.__version__,'device':'cpu','dtype':'float64',
      'architecture':{'inputs':13,'outputs':1,'hidden':[30,20,10],'activation':'ReLU','auxiliary_default':50},
      'parameter_count':sum(p.numel() for p in model.parameters()),'seed':seed,'loader_seed':0,'train_rows':455,'held_out_rows':51,
      'normalization':'training-only sample standard deviation; correction=1','adaptation_steps':0})
  assert report['parameter_count']==1291
  for epoch in range(30):
   cap();total=0.;batches=0;lr=opt.param_groups[0]['lr']
   for xb,yb in loader:
    cap();opt.zero_grad(set_to_none=True)
    l0=((model(xb)-yb)**2).mean();l1=((model(xb,yb)-yb)**2).mean();loss=l0+l1
    assert torch.isfinite(loss)
    loss.backward();opt.step();last=float(loss.detach());total+=last;batches+=1
    report['training_steps']+=1
   sched.step(last)
   report['epochs'].append({'epoch':epoch+1,'minibatches':batches,'lr':lr,'mean_minibatch_loss':total/batches,
       'scheduler_last_minibatch_loss':last,'lr_next':opt.param_groups[0]['lr']})
  cap();model.eval()
  with torch.no_grad():
   q=model(x);labelq=model(x,y);selfq=model(x,q)
   baseline=((y-y.mean(0))**2).mean()
   metrics={'default_nmse':float(((q-y)**2).mean()/baseline),
      'label_input_nmse':float(((labelq-y)**2).mean()/baseline),
      'prediction_std_ratio':float(q.std(correction=0)/y.std(correction=0)),
      'training_self_consistency_mse':float(((selfq-q)**2).mean()),'training_baseline_mse':float(baseline)}
  assert all(np.isfinite(v) for v in metrics.values())
  assert all(torch.isfinite(p).all() for p in model.parameters())
  gates={'complete':report['training_steps']==450,'default_fit':metrics['default_nmse']<=.75,
         'label_input_fit':metrics['label_input_nmse']<=.25,'nonconstant':metrics['prediction_std_ratio']>=.10}
  report.update({'metrics':metrics,'qualification':gates,'qualified':all(gates.values()),'status':'complete'})
  cap()
 except Exception as exc:
  report.update({'status':'failed','error':f'{type(exc).__name__}: {exc}','qualified':False})
 finally:
  if model is not None:
   import numpy as np
   np.savez_compressed(out/'checkpoint.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
   report['checkpoint_sha256']=sha(out/'checkpoint.npz')
  if scales is not None:
   import numpy as np
   np.savez_compressed(out/'preprocessing.npz',**scales);report['preprocessing_sha256']=sha(out/'preprocessing.npz')
  if time.process_time()-CPU0>20:
   report.update({'status':'failed','qualified':False,'budget_failure':'Total CPU including saving exceeded20s'})
  save()
 print(json.dumps({k:report[k] for k in ['status','qualified','training_steps','cpu_seconds']},indent=2))
 if report['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
