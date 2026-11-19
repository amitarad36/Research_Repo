"""One registered checkpoint, source input features only; no optimization/labels."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json, hashlib, time, argparse
import numpy as np
import torch as t
parser=argparse.ArgumentParser(); parser.add_argument('--repo',type=Path,required=True)
args=parser.parse_args(); root=args.repo; here=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reg=json.loads((here/'registered.json').read_text())
assert reg['script_sha256']==sha(Path(__file__))
assert reg['protocol_sha256']==sha(here/'PROTOCOL.md')
for p,h in reg['inputs'].items():assert sha(root/p)==h,p
assert t.__version__=='2.9.0+cpu';t.set_num_threads(1)
start=time.process_time()
def cap():assert time.process_time()-start<30,'CPU budget exceeded'
old=root/'research/notes/2026-10-31-cycle18'
training=json.loads((old/'training_results.json').read_text())
for file,key in [('checkpoint.npz','checkpoint_sha256'),('preprocessing.npz','preprocessing_sha256')]:assert sha(old/file)==training[key]
layers=['input','hidden1','hidden2','output']
keys=[l+s for l in layers for s in ['.weight','.bias']]
with np.load(old/'checkpoint.npz',allow_pickle=False) as data:w={k:data[k].copy() for k in keys}
with np.load(old/'preprocessing.npz',allow_pickle=False) as prep:xmean=prep['x_mean'].copy();xstd=prep['x_std'].copy()
# Only the 13 feature tokens in the first 455 nonempty rows are parsed.
rows=[]
with (root/'research/notes/2026-10-31-cycle17/bostonHousing.txt').open('rb') as f:
 for line in f:
  if not line.strip():continue
  rows.append([float(v) for v in line.split()[:13]])
  if len(rows)==455:break
x=(np.array(rows)-xmean)/xstd;assert x.shape==(455,13)
def forward(ww,z,derivatives=False):
 a=np.concatenate([x,np.asarray(z).reshape(-1,1)],axis=1);ins=[];pre=[]
 for layer in layers:
  ins.append(a);a=a@ww[layer+'.weight'].T+ww[layer+'.bias']
  if layer!='output':pre.append(a);a=np.maximum(a,0)
 if not derivatives:return a[:,0]
 # Reverse differentiation of a single scalar output for every input row.
 delta=np.ones((len(x),1));parts={}
 for i in range(3,-1,-1):
  l=layers[i]
  parts[l+'.weight']=np.einsum('ni,nj->nij',delta,ins[i]).reshape(len(x),-1)
  parts[l+'.bias']=delta.copy()
  delta=delta@ww[l+'.weight']
  if i:delta=delta*(pre[i-1]>0)
 return a[:,0],np.concatenate([parts[k] for k in keys],axis=1),delta[:,-1],np.concatenate(pre,axis=1)
q,J,_,preq=forward(w,np.full(len(x),50.),True)
ref,_,A,prer=forward(w,q,True);r=ref-q
assert J.shape==(455,1291)
per={'squared_input':J*((A-1)*r)[:,None], 'squared_stop':-J*r[:,None],
     'absolute_input':J*((A-1)*np.sign(r))[:,None], 'absolute_stop':-J*np.sign(r)[:,None]}
fields={k:v.mean(0) for k,v in per.items()}
# Separate PyTorch implementation with frozen reference parameters, live input gradient.
tx=t.from_numpy(x);tw={k:t.from_numpy(v.copy()).requires_grad_(True) for k,v in w.items()}
fw={k:v.detach().clone() for k,v in tw.items()}
def tf(ww,z):
 a=t.cat([tx,z.reshape(-1,1)],dim=1)
 for l in layers:
  a=a@ww[l+'.weight'].T+ww[l+'.bias']
  if l!='output':a=a.relu()
 return a[:,0]
errors={}
for kind in per:
 tq=tf(tw,t.full((len(x),),50.,dtype=t.float64));tr=tf(fw,tq)
 err=max(np.max(abs(tq.detach().numpy()-q)),np.max(abs(tr.detach().numpy()-ref)))
 errors['forward']=max(errors.get('forward',0),float(err))
 rr=(tr.detach() if kind.endswith('stop') else tr)-tq
 loss=(rr.square().mean()/2) if kind.startswith('squared') else rr.abs().mean()
 grads=t.autograd.grad(loss,[tw[k] for k in keys]);g=np.concatenate([a.detach().numpy().ravel() for a in grads])
 errors[kind]=float(np.max(abs(g-fields[kind])))
 assert errors[kind]<=1e-10*max(1,np.linalg.norm(fields[kind]))
assert errors['forward']<1e-10
cap()
singular=np.linalg.svd(J,compute_uv=False);ranktol=1e-10*max(1,float(singular[0]));rank=int(sum(singular>ranktol))
# Known exact positive-homogeneity symmetry, and its derivative at scale 1.
scaled={k:v.copy() for k,v in w.items()};c=1.1
scaled['input.weight'][0]*=c;scaled['input.bias'][0]*=c;scaled['hidden1.weight'][:,0]/=c
symerror=max(float(np.max(abs(forward(scaled,np.full(len(x),50.))-q))),float(np.max(abs(forward(scaled,q)-ref))))
tangent={k:np.zeros_like(v) for k,v in w.items()}
tangent['input.weight'][0]=w['input.weight'][0];tangent['input.bias'][0]=w['input.bias'][0]
tangent['hidden1.weight'][:,0]=-w['hidden1.weight'][:,0]
tv=np.concatenate([tangent[k].ravel() for k in keys]);tverr=float(np.max(abs(J@tv)))
assert np.linalg.norm(tv)>0 and symerror<1e-10 and tverr<1e-10
field_results={}
for k,g in fields.items():
 scale=max(1,float(np.mean(np.linalg.norm(per[k],axis=1))));norm=float(np.linalg.norm(g))
 field_results[k]={'norm':norm,'scale':scale,'threshold':1e-8*scale,'stationarity_gate':norm<=1e-8*scale}
rs=1e-8*max(1,float(max(abs(q))),float(max(abs(ref))))
result={'status':'pass','checkpoint_seed':0,'inputs':455,'parameters':1291,'auxiliary_default':50,
 'residual':{'rms':float(np.sqrt(np.mean(r*r))),'max_abs':float(max(abs(r))),'threshold':rs,'zero_residual_gate':bool(max(abs(r))<=rs)},
 'fields':field_results,'activation_margins':{k:{'minimum_abs':float(np.min(abs(v))),'near_zero_count':int(np.sum(abs(v)<=1e-10))} for k,v in [('student',preq),('reference',prer)]},
 'empirical_J_rank':rank,'rank_threshold':ranktol,'parameter_nullity_lower_bound':1291-rank,
 'rescaling_symmetry':{'factor':c,'output_error':symerror,'tangent_error':tverr,'tangent_norm':float(np.linalg.norm(tv))},
 'independent_errors':errors,'hessian_spectrum_computed':False,'training_steps':0,'adaptation_steps':0,'label_values_parsed':0,
 'torch':t.__version__,'numpy':np.__version__,'script_sha256':sha(Path(__file__)),'registered_sha256':sha(here/'registered.json')}
result['equilibrium_candidate']=all(field_results[k]['stationarity_gate'] for k in ['squared_input','squared_stop'])
result['decision']='abstain_at_checkpoint' if not result['equilibrium_candidate'] else 'needs_exact_equilibrium_and_curvature_audit'
np.savez_compressed(here/'diagnostics.npz',q=q,reference=ref,residual=r,A=A,J_singular_values=singular,**fields)
result['cpu_seconds_excluding_imports']=time.process_time()-start;cap()
(here/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
