"""Data-free CPU graph/reset fixture. Not trained-model adaptation evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys,time,json,hashlib,copy
from pathlib import Path
start=time.process_time()
HERE=Path(__file__).resolve().parent
assert not (HERE/'conformance_results.json').exists()
report={'python':sys.executable,'device':'cpu','labels_used':False,'trained_checkpoint_loaded':False,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
try:
 import torch
 from torch import nn
 torch.set_num_threads(1)
 report['torch_version']=torch.__version__
 class Model(nn.Module):
  def __init__(self):
   super().__init__()
   self.input=nn.Linear(14,30);self.hidden1=nn.Linear(30,20)
   self.hidden2=nn.Linear(20,10);self.output=nn.Linear(10,1)
  def forward(self,x,z=None):
   if z is None:z=x.new_full((len(x),1),50.)
   a=torch.cat([x,z],1)
   return self.output(torch.relu(self.hidden2(torch.relu(self.hidden1(torch.relu(self.input(a)))))))
 model=Model().double()
 with torch.no_grad():
  for i,p in enumerate(model.parameters()):
   p.copy_(((torch.arange(p.numel(),dtype=torch.float64).reshape(p.shape)%7)-3)/100+(.1 if p.ndim==1 else .02))
 initial=copy.deepcopy(model.state_dict());reference=copy.deepcopy(model)
 for p in reference.parameters():p.requires_grad_(False)
 x=torch.arange(26,dtype=torch.float64).reshape(2,13)/25-0.5
 opt=torch.optim.Adam(model.parameters(),lr=1e-4,foreach=False)
 # Construct nonzero history without training or consuming data.
 for p in model.parameters():
  opt.state[p]={'step':torch.tensor(7.),'exp_avg':torch.full_like(p,.001),
                'exp_avg_sq':torch.full_like(p,.002)}
 snapshot=copy.deepcopy(opt.state_dict())
 def flat(items):return torch.cat([p.reshape(-1) for p in items])
 def clone():
  m=copy.deepcopy(model);m.load_state_dict(initial)
  o=torch.optim.Adam(m.parameters(),lr=1e-4,foreach=False);o.load_state_dict(copy.deepcopy(snapshot))
  return m,o
 def input_value(q,c,v,b):
  if v=='current':return q if b else q.detach()
  return c+(q-q.detach()) if b else c
 def gradients(m,c,v,b):
  q=m(x);z=input_value(q,c,v,b);target=reference(x,z)
  loss=(target-q).abs().mean()
  gg=torch.autograd.grad(loss,tuple(m.parameters()))
  return flat(gg),q,z,target
 results={};max_formula=0.;max_cache=0.
 for v in ['current','cached']:
  for b in [0,1]:
   m,o=clone();cache=m(x).detach().clone();cache0=cache.clone()
   step_records=[]
   for step in range(2):
    g,q,z,t=gradients(m,cache,v,b)
    zv=z.detach().requires_grad_(True);tt=reference(x,zv)
    A=torch.autograd.grad(tt.sum(),zv)[0].detach()
    weight=((1-b*A)*torch.sign(t.detach()-q.detach()))/q.numel()
    qcheck=m(x)
    U=flat(torch.autograd.grad(qcheck,tuple(m.parameters()),grad_outputs=weight))
    max_formula=max(max_formula,float((g+U).abs().max()))
    if v=='cached':max_cache=max(max_cache,float((z.detach()-cache).abs().max()))
    o.zero_grad(set_to_none=True)
    cursor=0
    for p in m.parameters():
     p.grad=g[cursor:cursor+p.numel()].reshape_as(p).clone();cursor+=p.numel()
    o.step()
    assert torch.equal(cache,cache0)
    step_records.append({'g':g.clone(),'p':flat(m.parameters()).detach().clone(),
                         'state':copy.deepcopy(o.state_dict())})
   # Source-style reset changes weights but must retain moment/counter state.
   state_before=copy.deepcopy(o.state_dict());m.load_state_dict(initial)
   for k,s in o.state_dict()['state'].items():
    for field in ['step','exp_avg','exp_avg_sq']:assert torch.equal(s[field],state_before['state'][k][field])
   assert all(torch.equal(m.state_dict()[k],v0) for k,v0 in initial.items())
   results[(v,b)]=step_records
 first_equal={}
 for b in [0,1]:
  l=results[('current',b)][0];r=results[('cached',b)][0]
  first_equal[str(b)]={'gradient_max_abs':float((l['g']-r['g']).abs().max()),
                      'parameter_max_abs':float((l['p']-r['p']).abs().max())}
  assert first_equal[str(b)]['gradient_max_abs']<=1e-12
  assert first_equal[str(b)]['parameter_max_abs']<=1e-12
 # Negative control: same weights/gradient, different initial Adam history.
 m,o=clone();cache=m(x).detach();g,*_=gradients(m,cache,'current',1)
 for p in m.parameters():o.state[p]['exp_avg'].zero_()
 cursor=0
 for p in m.parameters():p.grad=g[cursor:cursor+p.numel()].reshape_as(p).clone();cursor+=p.numel()
 o.step();negative=float((flat(m.parameters()).detach()-results[('current',1)][0]['p']).abs().max())
 assert negative>1e-12
 assert max_formula<=1e-12 and max_cache==0
 report.update({'status':'passed','dtype':'float64','batch_size':2,'parameter_count':sum(p.numel() for p in model.parameters()),
   'fixture_optimizer_steps':9,'gradient_formula_max_abs':max_formula,'cached_forward_max_abs':max_cache,
   'first_update_pair_checks':first_equal,'different_history_update_difference':negative,
   'reset_preserves_moments_and_step':True,'cache_unchanged':True,'cpu_seconds':time.process_time()-start})
 assert report['cpu_seconds']<=10,'10 CPU second cap exceeded; do not classify as protocol pass'
except Exception as exc:
 report.update({'status':'failed','error':f'{type(exc).__name__}: {exc}','cpu_seconds':time.process_time()-start})
 (HERE/'conformance_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 raise
(HERE/'conformance_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
