"""Run beside the cycle-23 evidence directory. No fitted data or targets."""
import contextlib
import hashlib
import io
import json
import runpy
import sys
from decimal import Decimal as D
from pathlib import Path
from time import process_time

prior = Path(sys.argv[1])
assert hashlib.sha256(prior.read_bytes()).hexdigest() == "4cc98c9e512793cd14b7f009bd491dfca8588df9ed1973184136b87b6e3dc899"
start = process_time()
with contextlib.redirect_stdout(io.StringIO()):
    adam_step = runpy.run_path(str(prior))["step"]

A = [(2, 2), (2, -2)]
C = [(-2, 2), (-2, -2)]
alias = [(2, 1), (2, -1)]
B = [(0, 2), (0, 2)]
neutral = [(0, 2), (0, -2)]
origin = (D(0), D(0))

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))
def state0(count):
    return origin,origin,count
def batch(batch_x,state,steps):
    theta = origin
    mean_x = tuple(sum(D(row[j]) for row in batch_x)/len(batch_x) for j in range(2))
    for _ in range(steps):
        for row in batch_x:
            q = dot(theta,row)
            assert (q-50)/2-q < 0
        theta,m,v,count = adam_step(mean_x,theta,*state)
        state = m,v,count
    return theta,state

def prefix(first):
    _,state = batch(first,state0(0),2)
    return batch(B,state,2)[1]

states = {key:prefix(x) for key,x in [("A_B",A),("C_B",C),("alias_B",alias),("neutral_B",neutral)]}
assert states["A_B"]==states["alias_B"]
assert states["A_B"][1:]==states["C_B"][1:]
states.update(fresh=state0(0),cleared=state0(4))
outputs = {}
weights = {}
for key,state in states.items():
    theta,_ = batch(A,state,1)
    weights[key] = theta
    outputs[key] = tuple(dot(theta,row) for row in A)
fresh = outputs["fresh"]
def projection(y):
    scale = max(D(0),dot(y,fresh)/dot(fresh,fresh))
    residual = tuple(a-scale*b for a,b in zip(y,fresh))
    return scale,dot(residual,residual).sqrt()
def off_axis(y):
    return (y[0]-y[1])/2

assert fresh[0]==fresh[1] and fresh[0]<0
assert outputs["A_B"]==outputs["alias_B"]
assert weights["A_B"][1]==weights["C_B"][1]==weights["neutral_B"][1]
assert weights["A_B"][0]!=weights["C_B"][0]
tol=D("1e-48")
contrast = tuple(a-c for a,c in zip(outputs["A_B"],outputs["C_B"]))
assert abs(off_axis(contrast))<tol and contrast[0]<0
assert projection(contrast)[1]<tol
assert off_axis(outputs["A_B"])!=0 and projection(outputs["A_B"])[1]>0
for k in ["A_B","C_B","neutral_B","alias_B"]:
    assert abs(off_axis(outputs[k])-2*weights[k][1])<tol
assert abs(off_axis(outputs["neutral_B"])-off_axis(outputs["A_B"]))<tol
elapsed = process_time()-start
assert elapsed<10
print(json.dumps({"pre_return":[0,0],"return_gradient":[1,0],"precision":50,
  "arms":{k:{"post":list(map(str,y)), "alpha_and_l2_residual":list(map(str,projection(y))),"off_axis":str(off_axis(y))} for k,y in outputs.items()},
  "A_minus_C":list(map(str,contrast)), "cpu_seconds":elapsed},indent=2))

