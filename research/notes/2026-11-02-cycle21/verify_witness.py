"""Exact arithmetic for the compatible-target witness. No research data needed."""
from fractions import Fraction as F
import json
def dot(a, b):
    return sum(x*y for x, y in zip(a, b))
def project(design, w):
    return [dot(x, w) for x in design]
def mse(a, b):
    return sum((x-y)**2 for x, y in zip(a, b)) / len(a)
old = [(1, 1)]
current = [(2, 1), (1, -1)]
w = [F(1,2), F(1,2)]
h = [F(1,4), -F(1,4)]
p = project(current, w)
updated = [v+u for v,u in zip(w,h)]
post = project(current, updated)
d = [a-b for a,b in zip(post,p)]
energy = dot(d,d) / len(d)
assert energy == F(5,32)
report = {}
for direction in (1,-1):
    world = [v+direction*u for v,u in zip(w,h)]
    assert project(old,world) == project(old,w) == project(old,updated)
    assert dot(world,world) == F(5,8)
    y = project(current,world)
    change = mse(post,y)-mse(p,y)
    assert mse(p,y)==energy
    assert change==(1-2*direction)*energy
    assert change==(dot(d,d)-2*dot([a-b for a,b in zip(y,p)],d))/len(d)
    report[str(direction)] = {"initial_mse":str(energy),"post_mse":str(mse(post,y)),"change":str(change)}
print(json.dumps(report,indent=2))
