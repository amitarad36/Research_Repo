"""Two-dimensional affine IT3 surrogate. Decimal arithmetic; no task labels."""
from decimal import Decimal as D, getcontext
from time import process_time
import json
getcontext().prec = 50
zero, one = D(0), D(1)
beta, gamma = D(".9"), D(".999")
eta, eps = D(".0001"), D("1e-8")
A, C, B = (2,0), (-2,0), (0,2)

def step(x, theta, m, v, count):
    q = sum(t*i for t,i in zip(theta,x))
    residual = (q-50)/2-q
    assert residual < 0
    gradient = [D(i)/2 for i in x]
    m = [beta*a + (one-beta)*b for a,b in zip(m,gradient)]
    v = [gamma*a + (one-gamma)*b*b for a,b in zip(v,gradient)]
    count += 1
    theta = [t-eta*(s/(1-beta**count))/((z/(1-gamma**count)).sqrt()+eps)
             for t,s,z in zip(theta,m,v)]
    return theta,m,v,count

def history(first):
    m, v, t = [zero]*2, [zero]*2, 0
    for x in [first,B]:
        theta = [zero]*2
        for _ in range(2):
            theta,m,v,t = step(x,theta,m,v,t)
    return m,v,t

def probe(state):
    theta,m,v,t = step(A,[zero]*2,*state)
    return {"pre":"0","post":str(2*theta[0]),"return_step":t,
            "m":list(map(str,m)),"v":list(map(str,v))}

started=process_time()
plus,minus=history(A),history(C)
assert plus[1:]==minus[1:] and plus[2]==4
report={k:probe(h) for k,h in {"A_B":plus,"C_B":minus,
        "zero_at_four":([zero]*2,[zero]*2,4),"fresh":([zero]*2,[zero]*2,0)}.items()}
for sign,key in [(1,"A_B"),(-1,"C_B")]:
    assert D(report[key]["m"][0])==one-beta+sign*beta**3*(one-beta**2)
assert report["A_B"]["v"]==report["C_B"]["v"]
assert D(report["A_B"]["post"])<0<D(report["C_B"]["post"])
assert D(report["zero_at_four"]["post"])<0 and D(report["fresh"]["post"])<0
assert beta**6*(1+beta)>1>beta**7*(1+beta)
elapsed=process_time()-started
assert elapsed<10
print(json.dumps({"predictions":report,"decimal_precision":50,"cpu_seconds":elapsed},indent=2))

