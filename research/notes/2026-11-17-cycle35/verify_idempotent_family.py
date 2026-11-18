"""Cycle 35: exact squared-loss model; no training/data access or original IT3 claim."""
from pathlib import Path
import json, time
import numpy as np
import sympy as S

start=time.process_time()
t,p,w,z1,z2=S.symbols('theta phi w z1 z2', real=True)
tau,eta,alpha=S.symbols('tau eta alpha', positive=True)
z=S.Matrix([z1,z2])
def A(v): return S.Matrix([[1,1-3*v],[0,0]])
def q(v,sign=1): return sign*S.Matrix([-(1-3*v)*(1+v),1+v])
def f(v,zz,sign=1): return q(v,sign)+A(v)*zz
def zero(expr):
    entries=list(expr) if isinstance(expr,S.MatrixBase) else [expr]
    assert all(S.simplify(e)==0 for e in entries), expr

zero(A(w)*A(w)-A(w))
zero(A(w)*q(w))
for sign in [-1,1]:
    zero(f(w,f(w,z,sign),sign)-f(w,z,sign))
    zero(f(w,S.zeros(2,1),sign)-q(w,sign))
    zero(f(w,q(w,sign),sign)-q(w,sign))
r=f(p,q(t))-q(t)
zero(r-(t-p)*S.Matrix([1-3*p,-1]))
D=(r.dot(r))/2
J=q(t).diff(t)
Ginput=S.diff(D,t)
Gstop=-(J.T*r)[0]
zero(Ginput-(t-p)*((1-3*p)**2+1))
zero(Gstop+(t-p)*((2+6*t)*(1-3*p)-1))
zero(Ginput.subs(p,0)-2*t)
zero(Gstop.subs(p,0)+t*(1+6*t))
assert S.diff(-Ginput.subs(p,0),t).subs(t,0)==-2
assert S.diff(-Gstop.subs(p,0),t).subs(t,0)==1
checks=[]
for name,g,h in [('input',Ginput,2),('stop',Gstop,-1)]:
    field=S.Matrix([-g,(t-p)/tau])
    M=field.jacobian([t,p]).subs({t:0,p:0})
    zero(M-S.Matrix([[-h,h],[1/tau,-1/tau]]))
    theta_next=t-eta*g
    phi_next=alpha*p+(1-alpha)*theta_next
    N=S.Matrix([theta_next,phi_next]).jacobian([t,p]).subs({t:0,p:0})
    lam=S.symbols('lam')
    zero(M.charpoly(lam).as_expr()-lam*(lam+h+1/tau))
    zero(N.charpoly(lam).as_expr()-(lam-1)*(lam-alpha*(1-eta*h)))
    for tv in [S.Rational(1,4),S.Rational(1,2),S.Integer(2),S.Integer(5)]:
        for ev in [S.Rational(1,100),S.Rational(1,10)]:
            av=tv/(tv+ev)
            continuous=-h-1/tv
            discrete=av*(1-ev*h)
            assert bool(continuous<0)==bool(abs(discrete)<1)
            checks.append(dict(rule=name,tau=str(tv),eta=str(ev),
                transverse_eigenvalue=str(continuous),transverse_multiplier=str(discrete),
                transverse_stable=bool(continuous<0),neutral_eigenvalue=0,neutral_multiplier=1))

# Independent finite differences of scalar losses, including a held-constant stopped target.
def qn(v): return np.array([-(1-3*v)*(1+v),1+v],dtype=float)
def fn(v,zz): return qn(v)+np.array([[1,1-3*v],[0,0]])@zz
rng=np.random.default_rng(35)
max_gradient_error=0.
for th,ph in rng.uniform(-.1,.1,size=(40,2)):
    e=1e-6
    target=fn(ph,qn(th))
    def loss(v):
        res=fn(ph,qn(v))-qn(v)
        return .5*res@res
    def stopped(v):
        res=target-qn(v)
        return .5*res@res
    gi=(loss(th+e)-loss(th-e))/(2*e)
    gs=(stopped(th+e)-stopped(th-e))/(2*e)
    expected_i=(th-ph)*((1-3*ph)**2+1)
    expected_s=-(th-ph)*((2+6*th)*(1-3*ph)-1)
    max_gradient_error=max(max_gradient_error,abs(gi-expected_i),abs(gs-expected_s))
assert max_gradient_error<1e-7

# Finite-step diagnostic only, with predeclared small unmatched initial states.
trajectories=[]
for name in ['input','stop']:
    for tv in [.5,2.]:
        for ev in [.1,.05]:
            th,ph=1e-6,0.
            av=tv/(tv+ev)
            for _ in range(round(2/ev)):
                if name=='input': g=(th-ph)*((1-3*ph)**2+1)
                else: g=-(th-ph)*((2+6*th)*(1-3*ph)-1)
                th-=ev*g
                ph=av*ph+(1-av)*th
            ratio=abs(th-ph)/1e-6
            expected_growth=(name=='stop' and tv==2.)
            assert (ratio>1)==expected_growth
            trajectories.append(dict(rule=name,tau=tv,eta=ev,horizon=2,
                gap_ratio=ratio,theta_final=th,phi_final=ph))
result=dict(status='passed',symbolic_checks='idempotence, residual, both fields, both Jacobians and characteristic polynomials',
    frozen_reference_linear_eigenvalues={'input':-2,'stop':1},
    exact_parameter_cases=checks,finite_difference_probes=40,max_gradient_error=max_gradient_error,
    small_nonlinear_diagnostics=trajectories, cpu_seconds=time.process_time()-start,
    limitations=['Constructed smooth squared-loss model, not trained IT3.',
                 'Neutral direction changes predictions: no full-state or accuracy recovery guarantee.',
                 'Matched initialization is stationary for both rules.',
                 'Local transverse classification only; no boundary classification or global stopped result.',
                 'No novelty claim; residual versus semi-gradient separation has extensive prior art.'])
Path(__file__).with_name('verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','finite_difference_probes','max_gradient_error','cpu_seconds']}))
