"""Independent central-difference check of the loss built from full forward calls."""
from pathlib import Path
import json
import numpy as np
import verify_vector_stability as m

rng=np.random.default_rng(42)
maximum=0.
for _ in range(5):
    W=rng.normal(size=(2,2))*.03
    V=rng.normal(size=(2,2))*.03
    def objective(Warg):
        value=0.
        for i,(x,y) in enumerate(zip(m.xs,m.ys)):
            B=.5*np.eye(2)+m.rot+(.1*np.diag([1.,-1.]) if i else 0.)
            A=np.eye(2)-B
            q=m.base_model(Warg,x,y,A,np.zeros(2))
            t=m.base_model(V,x,y,A,q)
            value+=np.linalg.norm(t-q)**2/4
        return value
    state=np.r_[W.flatten(order='F'),V.flatten(order='F')]
    analytical=-m.velocity(state,.1,1.,'input')[:4]
    numeric=[]; eps=1e-6
    for j in range(4):
        perturb=np.zeros(4); perturb[j]=eps
        P=perturb.reshape((2,2),order='F')
        numeric.append((objective(W+P)-objective(W-P))/(2*eps))
    maximum=max(maximum,float(np.max(np.abs(analytical-numeric))))
assert maximum<1e-9
out={'full_architecture_gradient_check_cases':5,'max_absolute_error':maximum,
     'seed':42,'finite_difference_epsilon':1e-6}
(Path(__file__).resolve().parent/'gradient_verification.json').write_text(
    json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))

