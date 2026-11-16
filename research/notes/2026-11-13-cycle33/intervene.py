"""Cycle33: registered per-batch Adam state intervention, no target access."""
import os
for k in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time,copy,importlib.util,traceback
from pathlib import Path
import numpy as np
import torch as t
D=Path(__file__).resolve().parent;N=D.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def paths(seed):
 c=N/('2026-10-31-cycle18' if seed==0 else '2026-11-07-cycle30/seed_'+str(seed))
 tr=N/('2026-11-01-cycle19/run/trace.npz' if seed==0 else '2026-11-07-cycle31/seed_'+str(seed)+'/run/trace.npz')
 return c,tr
def main():
 reg=read(D/'registered.json')
 for p,h in reg.items():assert sha(Path(p))==h,p
 spec=importlib.util.spec_from_file_location('diag',N/'2026-11-01-cycle19/diagnostic.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 assert t.__version__=='2.9.0+cpu';t.set_num_threads(1)
 report={'status':'running','updates':0,'seeds':[],'checks':{},'torch':str(t.__version__),'registration_sha256':sha(D/'registered.json')}
 trace={};start=time.process_time()
 def ck(key,a,b):
  e=mod.check(np.asarray(a),np.asarray(b));report['checks'][key]=max(e,report['checks'].get(key,0))
 def cap():assert time.process_time()-start<60,'60-second CPU cap'
 def independent(m,x):
  w={k:mod.arr(v) for k,v in m.state_dict().items()};h=np.concatenate([mod.arr(x),np.full((2,1),50.)],axis=1)
  for layer in ('input','hidden1','hidden2','output'):
   h=h@w[layer+'.weight'].T+w[layer+'.bias']
   if layer!='output':h=np.maximum(h,0)
  return h
 try:
  raw=[r for r in (N/'2026-10-31-cycle17/bostonHousing.txt').read_bytes().splitlines() if r.strip()]
  Xraw=np.array([[float(v) for v in r.split()[:13]] for r in raw[455:495]])
  assert Xraw.shape==(40,13)
  for seed in range(4):
   cap();c,tr=paths(seed);old=np.load(tr,allow_pickle=False)
   ref=mod.Net().double().eval()
   with np.load(c/'checkpoint.npz',allow_pickle=False) as w:ref.load_state_dict({k:t.from_numpy(w[k]) for k in w.files})
   ref.requires_grad_(False)
   with np.load(c/'preprocessing.npz',allow_pickle=False) as sc:X=t.from_numpy((Xraw-sc['x_mean'])/sc['x_std'])
   buf={k:[] for k in ('baseline','q1','q2','before_theta','before_m','before_v','before_step','after_theta','after_m','after_v','after_step','g')}
   for b,x in enumerate(X.split(2)):
    idx=b*8;q0=mod.arr(ref(x));ck('baseline',q0,old['noadapt'][b]);buf['baseline'].append(q0)
    initial={k:old['before_'+k][idx] for k in ('theta','m','v','step')}
    assert np.all(initial['step']==2*b) and np.all(initial['v']>=0)
    ck('reset_theta',initial['theta'],mod.flat(ref.parameters()))
    firstg=None;firstfinal=None
    for arm in range(3):
     cap();m=copy.deepcopy(ref).requires_grad_(True);o=mod.adam(m);offset=0
     for pi,p in enumerate(m.parameters()):
      n=p.numel()
      o.state[p]['step']=t.tensor(float(initial['step'][pi] if arm<2 else 0),dtype=t.float32)
      for key,field in [('exp_avg','m'),('exp_avg_sq','v')]:
       v=initial[field][offset:offset+n] if arm==0 else np.zeros(n)
       o.state[p][key]=t.from_numpy(v.copy()).reshape(p.shape)
      offset+=n
     assert offset==1291
     for step in range(2):
      cap();st=mod.state(m,o)
      for k,v in st.items():buf['before_'+k].append(v)
      if step==0:
       ck('common_start_theta',st['theta'],initial['theta'])
       ck('set_start_counter',st['step'],initial['step'] if arm<2 else np.zeros(8))
       for k in ('m','v'):ck('set_start_moments',st[k],initial[k] if arm==0 else np.zeros(1291))
      q=m(x);r=ref(x,q)-q;o.zero_grad(set_to_none=True);r.abs().mean().backward();g=mod.flat(p.grad for p in m.parameters())
      if step==0:
       if firstg is None:firstg=g.copy()
       else:ck('shared_initial_gradient',g,firstg)
      if arm==0:
       for k in st:ck('replay_before_'+k,st[k],old['before_'+k][idx+step])
       ck('replay_gradient',g,old['g'][idx+step])
      o.step();report['updates']+=1;post=mod.state(m,o)
      # Independent NumPy Adam recurrence from the pre-update state.
      mm=.9*st['m']+.1*g;vv=.999*st['v']+.001*g*g
      assert np.all(st['step']==st['step'][0]);n=int(st['step'][0])+1
      theta=st['theta']-.0001*(mm/(1-.9**n))/(np.sqrt(vv/(1-.999**n))+1e-8)
      for k,v in [('theta',theta),('m',mm),('v',vv),('step',st['step']+1)]:ck('numpy_adam_'+k,post[k],v)
      if arm==0:
       for k in post:ck('replay_after_'+k,post[k],old['after_'+k][idx+step])
      for k,v in post.items():buf['after_'+k].append(v)
      buf['g'].append(g)
      out=mod.arr(m(x));ck('numpy_forward',out,independent(m,x));buf['q'+str(step+1)].append(out)
     if arm==0:
      ck('replay_final_output',out,old['final_output'][b*4]);firstfinal=out.copy()
     if b==0:ck('zero_history_equal',out,firstfinal)
   for k,v in buf.items():trace['seed'+str(seed)+'_'+k]=np.asarray(v)
   report['seeds'].append(seed)
  assert report['updates']==480 and report['seeds']==list(range(4))
  cap();report['status']='complete'
 except BaseException:report.update(status='invalid',failure=traceback.format_exc())
 np.savez_compressed(D/'intervention_trace.npz',**trace)
 report['cpu_seconds']=time.process_time()-start
 if report['cpu_seconds']>=60:report.update(status='invalid',failure='CPU cap including trace save')
 (D/'run_results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 if report['status']=='complete':
  predictions={'arm_order':['carried','zero_moments_matched_counter','fresh'],'seeds':[]}
  for s in range(4):
   predictions['seeds'].append({'seed':s,'baseline':trace['seed'+str(s)+'_baseline'].reshape(40).tolist(),
    'q1':trace['seed'+str(s)+'_q1'].reshape(20,3,2).transpose(1,0,2).reshape(3,40).tolist(),
    'q2':trace['seed'+str(s)+'_q2'].reshape(20,3,2).transpose(1,0,2).reshape(3,40).tolist()})
  (D/'locked_predictions.json').write_text(json.dumps(predictions,indent=2,allow_nan=False))
  (D/'prediction_lock.json').write_text(json.dumps({p:sha(D/p) for p in ['locked_predictions.json','intervention_trace.npz','run_results.json']},indent=2))
 print(json.dumps(report));assert report['status']=='complete'
if __name__=='__main__':main()
