"""Cycle 36: local robustness obstruction, no training or evaluator data."""
from pathlib import Path
import sympy as S
import numpy as np
import json,time
start=time.process_time()
th,ph,ep,tau,eta,alpha,x=S.symbols('theta phi epsilon tau eta alpha x',real=True)
lam=S.Symbol('lambda')
r=x*(1+ph)+ep*ph*x*(1+th)-x*(1+th)
D=S.expand(r*r/2).subs(x*x,1)
gi=S.simplify(S.diff(D,th))
gs=S.expand(-x*r).subs(x*x,1)
M_expected=S.Matrix([[-1,1+ep],[1/tau,-1/tau]])
for name,g in [('input',gi),('stop',gs)]:
    assert S.simplify(g.subs(ph,0)-th)==0
    M=S.Matrix([-g,(th-ph)/tau]).jacobian([th,ph]).subs({th:0,ph:0})
    assert S.simplify(M-M_expected)==S.zeros(2)
    assert S.simplify(M.charpoly(lam).as_expr()-(lam**2+(1+1/tau)*lam-ep/tau))==0
    student=th-eta*g
    N=S.Matrix([student,alpha*ph+(1-alpha)*student]).jacobian([th,ph]).subs({th:0,ph:0})
    assert S.simplify((S.eye(2)-N).det()+eta*(1-alpha)*ep)==0
    assert S.simplify(N.det()-alpha*(1-eta))==0

# Exact projection-defect identity for an arbitrary 2x2 input Jacobian.
a,b,c,d=S.symbols('a b c d')
A=S.Matrix([[a,b],[c,d]]); B=S.eye(2)-A
assert S.simplify(B.T*B-B-((A.T-A)*(A-S.eye(2))+(A*A-A)))==S.zeros(2)

cases=[]
for eps in [S.Rational(s,n) for s in [-1,1] for n in [10,100,1000000]]:
    for tv in [S.Rational(1,4),S.Integer(1),S.Integer(4)]:
        for ev in [S.Rational(1,100),S.Rational(1,10)]:
            av=tv/(tv+ev)
            mm=M_expected.subs({ep:eps,tau:tv})
            nn=N.subs({ep:eps,tau:tv,eta:ev,alpha:av})
            chi1=S.simplify((S.eye(2)-nn).det())
            assert bool(mm.det()<0)==bool(eps>0)
            assert bool(chi1<0)==bool(eps>0)
            eig=np.linalg.eigvals(np.array(mm,dtype=float))
            multipliers=np.linalg.eigvals(np.array(nn,dtype=float))
            if eps>0:
                assert eig.real.max()>0 and max(abs(multipliers))>1
            else:
                # Strict Schur tests, verified in exact rational arithmetic.
                tr,det=nn.trace(),nn.det()
                assert 1-det>0 and 1-tr+det>0 and 1+tr+det>0
                assert eig.real.max()<0 and max(abs(multipliers))<1
            cases.append(dict(epsilon=str(eps),tau=str(tv),eta=str(ev),
                max_real_eigenvalue=float(eig.real.max()),spectral_radius=float(max(abs(multipliers))),
                discrete_characteristic_at_one=str(chi1)))

# Independent finite differences through the explicit scalar loss / stopped target.
rng=np.random.default_rng(36)
maxerr=0.
for theta,phi,eps in rng.uniform(-.1,.1,(40,3)):
    h=1e-6
    target=1+phi+eps*phi*(1+theta)
    def loss(v):
        rr=1+phi+eps*phi*(1+v)-(1+v)
        return .5*rr**2
    def stopped(v):
        return .5*(target-(1+v))**2
    rb=(1+eps)*phi-theta+eps*phi*theta
    exact=[rb*(eps*phi-1),-rb]
    actual=[(loss(theta+h)-loss(theta-h))/(2*h),
            (stopped(theta+h)-stopped(theta-h))/(2*h)]
    maxerr=max(maxerr,max(abs(a-b) for a,b in zip(exact,actual)))
assert maxerr<1e-7

# Check the stated zero-residual matrix identities and upper bounds on 50 fixtures.
n2=lambda a:np.linalg.norm(a,2)
max_identity_error=0.
for _ in range(50):
    A=rng.normal(size=(3,3)); J=rng.normal(size=(3,2)); C=rng.normal(size=(3,2))
    B=np.eye(3)-A; defect=C-B@J
    hi=J.T@B.T@B@J; hs=J.T@B@J
    ki=J.T@B.T@C; ks=J.T@C
    max_identity_error=max(max_identity_error,n2(ki-hi-J.T@B.T@defect),
                           n2(ks-hs-J.T@defect))
    assert n2(hi-hs)<=n2(J)**2*(n2(A.T-A)*n2(B)+n2(A@A-A))+1e-10
    assert n2(ki-hi)<=n2(J)*n2(B)*n2(defect)+1e-10
    assert n2(ks-hs)<=n2(J)*n2(defect)+1e-10
assert max_identity_error<1e-10
result=dict(status='passed',cases=cases,case_count=len(cases),
    gradient_probes=40,max_gradient_error=maxerr,
    matrix_fixtures=50,max_matrix_identity_error=max_identity_error,
    symbolic_checks='both fields, both joint Jacobians, characteristic polynomial, discrete chi(1), projection-defect identity',
    cpu_seconds=time.process_time()-start,
    limitations=['Constructed local smooth squared-loss model.',
                 'Idempotence is exact only at baseline when epsilon is nonzero.',
                 'No theorem-level novelty claim; no trained data or accuracy experiment.',
                 'Fails full-state robustness from a neutral baseline; conditional disagreement bound remains valid.'])
Path(__file__).with_name('verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','case_count','max_gradient_error','max_matrix_identity_error','cpu_seconds']}))
