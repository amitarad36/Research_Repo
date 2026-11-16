"""Registered current/cached forward x connected/stopped input experiment, cycle 19."""
import os
for k in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, json, hashlib, time, copy, traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch as t
from torch import nn
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--seed",type=int,choices=(1,2,3),required=True)
parser.add_argument("--budget",type=float,required=True)
args=parser.parse_args()
assert 0 < args.budget <= 60
ROOT=Path(__file__).resolve().parent
D=ROOT/("seed_"+str(args.seed))
COHORT=Path('C:\\Users\\amita\\source\\repos\\Detach Proposal Presentation\\research\\notes\\2026-11-07-cycle30')
OLD=COHORT/("seed_"+str(args.seed))
DATA=Path('C:\\Users\\amita\\source\\repos\\Detach Proposal Presentation\\research\\notes\\2026-10-31-cycle17\\bostonHousing.txt')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def arr(x): return x.detach().numpy().copy()
def flat(ps): return np.concatenate([arr(p).reshape(-1) for p in ps])
def check(a,b):
    assert np.isfinite(a).all() and np.isfinite(b).all()
    err=float(np.max(np.abs(a-b)))
    assert err<=1e-10*max(1,np.max(np.abs(a)),np.max(np.abs(b))),err
    return err
def norm(a): return float(np.linalg.norm(a))
def rms(a): return float(np.sqrt(np.mean(a*a)))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(14,30);self.hidden1=nn.Linear(30,20)
        self.hidden2=nn.Linear(20,10);self.output=nn.Linear(10,1)
    def forward(self,x,z=None,details=False):
        if z is None: z=x.new_full((len(x),1),50.)
        a=self.input(t.cat([x,z],1));b=self.hidden1(a.relu());c=self.hidden2(b.relu())
        y=self.output(c.relu())
        return (y,t.cat([a,b,c],1)) if details else y
def adam(m): return t.optim.Adam(m.parameters(),lr=.0001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
def state(m,o):
    pp=list(m.parameters())
    def field(k,p): return o.state.get(p,{}).get(k,t.zeros_like(p))
    return {'theta':flat(pp),'m':flat(field('exp_avg',p) for p in pp),'v':flat(field('exp_avg_sq',p) for p in pp),
            'step':np.array([int(o.state.get(p,{}).get('step',0)) for p in pp])}
def main():
    assert sha(Path(__file__))==(ROOT/'registered.sha256').read_text().strip().lower()
    assert sha(COHORT/'evidence_manifest.json')=='56ae87921643246515d56fa370d5ada649ae42a0eb28396453cffadd98cc8f34'
    for k,v in read(COHORT/'evidence_manifest.json').items(): assert sha(COHORT/k)==v
    assert read(OLD/'verification_results.json')['status']=='passed' and read(OLD/'training_results.json')['qualified']
    assert sha(DATA)=='baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a'
    D.mkdir()
    out=D/'run';out.mkdir()
    report={'status':'running','updates':0,'batches':[],'torch':str(t.__version__),'script_sha256':sha(Path(__file__)),'seed':args.seed,'cpu_budget':args.budget,'arm_order':['current-connected','cached-connected','current-stopped','cached-stopped']}
    trace=defaultdict(list); start=None
    try:
        assert t.__version__=='2.9.0+cpu';t.set_num_threads(1)
        ref=Net().double().eval()
        with np.load(OLD/'checkpoint.npz',allow_pickle=False) as w: ref.load_state_dict({k:t.from_numpy(w[k]) for k in w.files})
        raw=[r for r in DATA.read_bytes().splitlines() if r.strip()]
        X=np.array([[float(v) for v in r.split()[:13]] for r in raw[455:495]])
        assert X.shape==(40,13)
        with np.load(OLD/'preprocessing.npz',allow_pickle=False) as sc: X=t.from_numpy((X-sc['x_mean'])/sc['x_std'])
        ref.requires_grad_(False)
        start=time.process_time()
        def cap(): assert time.process_time()-start<args.budget,'CPU cap'
        history=None
        for batch,x in enumerate(X.split(2)):
            q0=arr(ref(x));trace['noadapt'].append(q0); results=[]
            for arm in range(4):
                cap();m=copy.deepcopy(ref).requires_grad_(True);o=adam(m)
                if history is not None:o.load_state_dict(copy.deepcopy(history))
                c=m(x).detach().clone();assert np.array_equal(arr(c),q0)
                first=None;second=None
                for step in range(2):
                    cap();st=state(m,o);assert np.all(st['step']==2*batch+step)
                    for k,v in st.items():trace['before_'+k].append(v)
                    q=m(x);z=[q,c+(q-q.detach()),q.detach(),c][arm]
                    if arm%2:assert np.array_equal(arr(z),q0)
                    y,pre=ref(x,z,True);r=y-q;s=r.sign()
                    probe=z.detach().requires_grad_(True);A=t.autograd.grad(ref(x,probe).sum(),probe)[0]
                    J=np.stack([flat(t.autograd.grad(q[i].sum(),tuple(m.parameters()),retain_graph=True)) for i in range(2)])
                    o.zero_grad(set_to_none=True);r.abs().mean().backward()
                    g=flat(p.grad for p in m.parameters());aa=arr(A);ss=arr(s)
                    analytic=(J*((1-(arm<2)*aa)*ss)).mean(0)
                    err=check(-g,analytic)
                    for k,v in dict(q=arr(q),z=arr(z),r=arr(r),sign=ss,A=aa,J=J,pre=arr(pre),mask=arr(pre>0),g=g,formula_err=err).items():trace[k].append(v)
                    o.step();report['updates']+=1
                    post=state(m,o)
                    for k,v in post.items():trace['after_'+k].append(v)
                    assert np.array_equal(arr(c),q0)
                    if step==0:first=(g,post)
                    else: second=(J,aa,ss,g,arr(pre>0),arr(m(x)),post)
                if arm==0:future=copy.deepcopy(o.state_dict())
                results.append((first,second))
            history=future
            for a,b in ((0,1),(2,3)):
                ga,pa=results[a][0];gb,pb=results[b][0]
                check(ga,gb)
                for k in pa:check(pa[k],pb[k])
            J,A,s,g,mask,q,post=results[0][1];K,B,u,h,mask2,qc,post2=results[1][1]
            check(J,K)
            signpart=(J*(1-A)*(s-u)).mean(0);jacpart=(J*(B-A)*u).mean(0)
            delta=h-g
            err=check(signpart+jacpart,delta)
            for k,v in dict(signpart=signpart,jacpart=jacpart,sum_parts=signpart+jacpart,delta_direction=delta).items():trace[k].append(v)
            for a in range(4):trace['final_output'].append(results[a][1][5])
            gt=1e-4*max(norm(g),norm(h),1e-8);ot=1e-6*max(rms(q0),1.)
            report['batches'].append(dict(index=batch,rows=[455+2*batch,456+2*batch],gradient_norm=norm(delta),gradient_gate=gt,output_rms=rms(q-qc),output_gate=ot,
              passes=bool(norm(delta)>gt and rms(q-qc)>ot),sign_changes=int((s!=u).sum()),jac_changes=int((A!=B).sum()),mask_bits_changed=int((mask!=mask2).sum()),
              sign_term=norm(signpart),jac_term=norm(jacpart),cancellation=norm(signpart)+norm(jacpart)-norm(signpart+jacpart),
              decomposition_error=err,parameter_difference=norm(post['theta']-post2['theta'])))
        assert report['updates']==160 and len(report['batches'])==20
        for k in ('r','pre'):
            v=np.asarray(trace[k]);trace[k+'_zero']=(v==0);trace[k+'_near']=(abs(v)<=1e-10)
        n=sum(b['passes'] for b in report['batches']);report.update(count=n,outcome='measurable' if n>=5 else 'reject_this_setting')
        cap();report['status']='complete'
    except BaseException:report.update(status='invalid',failure=traceback.format_exc())
    finally:
        np.savez_compressed(out/'trace.npz',**trace)
        if start is not None:
            report['cpu_seconds_after_load_including_trace_save']=time.process_time()-start
            if time.process_time()-start>=args.budget:report.update(status='invalid',failure='CPU cap')
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='batches'}))
    assert report['status']=='complete'
if __name__=='__main__':main()
