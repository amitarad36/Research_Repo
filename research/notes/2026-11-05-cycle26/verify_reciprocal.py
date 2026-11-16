"""Evaluate the symbolic four-coordinate Adam witness; no sampled data or fitting."""
from decimal import Decimal as D, getcontext
import json
getcontext().prec=50
b,v,eta,eps=D(".9"),D(".999"),D(".0001"),D("1e-8")
zero=(D(0),)*4
def g(axis,sign=1):
    return tuple(D(sign if j==axis else 0) for j in range(4))
def step(history):
    m=q=zero
    for gradient in history:
        m=tuple(b*a+(1-b)*c for a,c in zip(m,gradient))
        q=tuple(v*a+(1-v)*c*c for a,c in zip(q,gradient))
    t=len(history)
    return tuple(-eta*(a/(1-b**t))/((c/(1-v**t)).sqrt()+eps) for a,c in zip(m,q))
def risk(weights,axis):
    return sum((weights[axis]+s*weights[2]-1)**2 for s in (-1,1))/2
results={}
for name,bridge in [('active',g(3)),('sham',zero)]:
    cells=[]
    for prefix in (0,1):
        w=step([g(prefix,-1)]*2+[bridge]*2+[g(2)])
        cells.append([risk(w,axis) for axis in (0,1)])
    results[name]=cells
assert results['active']==results['sham']
a,bad=results['active'][0]
assert results['active'][1]==[bad,a] and a<1<bad
u,c=step([g(0,-1)]*2+[g(3)]*2+[g(2)])[::2]
assert abs((bad-a)-(2*u-u*u))<D("1e-46")
print(json.dumps({'matching_risk':str(a),'mismatching_risk':str(bad),'prefix_advantage_each_domain':str(bad-a),'active_minus_sham_advantage':'0','reset_baseline':'1','best_possible_scaled_fresh_risk':'1','prefix_axis_weight':str(u),'return_axis_weight':str(c)},indent=2))

