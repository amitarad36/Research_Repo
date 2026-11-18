"""Cycle 37: restoring geometry and exact student-first EMA step boundary.
No training, data downloads, or evaluator labels. Run with Python/SymPy/NumPy.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np
import sympy as s
start=time.process_time()
tau,eta,alpha=s.symbols('tau eta alpha', positive=True)
lam=s.Symbol('lambda')
I=s.eye(2); Z=s.zeros(2)
H=s.diag(1,4); L=s.Matrix([[2,1],[1,2]]); K=H-L
M=(-H).row_join(K).col_join((I/tau).row_join(-I/tau))
N=(I-eta*H).row_join(eta*K).col_join(((1-alpha)*(I-eta*H)).row_join(alpha*I+(1-alpha)*eta*K))
# Exact second-order coordinate change [u,v] -> [v,v_dot].
T=Z.row_join(I).col_join((I/tau).row_join(-I/tau))
F=Z.row_join(I).col_join((-L/tau).row_join(-H-I/tau))
assert s.simplify(T*M*T.inv()-F)==s.zeros(4)
P=L.row_join(Z).col_join(Z.row_join(tau*I))
Q=Z.row_join(Z).col_join(Z.row_join(2*(I+tau*H)))
assert s.simplify(F.T*P+P*F+Q)==s.zeros(4)
poly=lam**2*I-lam*((1+alpha)*I-alpha*eta*H-(1-alpha)*eta*L)+alpha*(I-eta*H)
assert s.simplify(N.charpoly(lam).as_expr()-poly.det())==0
assert s.simplify(poly.subs(lam,1)-(1-alpha)*eta*L)==Z
assert s.simplify(poly.subs(lam,-1)-(2*(1+alpha)*I-eta*(2*alpha*H+(1-alpha)*L)))==Z
assert H*L!=L*H
bound=6/(7+s.sqrt(10))
assert s.simplify((N.subs({alpha:s.Rational(1,2),eta:bound})+s.eye(4)).det())==0
# Realization by a single shared dual-input architecture, with planted baseline.
t1,t2,p1,p2,x,z1,z2=s.symbols('t1 t2 p1 p2 x z1 z2',real=True)
th=s.Matrix([t1,t2]); ph=s.Matrix([p1,p2]); b=s.Matrix([1,0]); R=s.diag(1,2)
U=-R.T.inv()*L
q=x*(b+R*th)
f=x*(b+R*ph)+(U*ph)*z1
r=f.subs({z1:q[0],z2:q[1]})-q
loss=s.expand((r.T*r)[0]/2).subs(x**2,1)
gi=s.Matrix([s.diff(loss,t1),s.diff(loss,t2)])
gs=s.simplify(-x*R.T*r).subs(x**2,1)
zero={t1:0,t2:0,p1:0,p2:0}
for g in (gi,gs):
    assert s.simplify(g.jacobian(th).subs(zero)-H)==Z
    assert s.simplify(-g.jacobian(ph).subs(zero)-K)==Z
assert s.simplify(f.jacobian([z1,z2]).subs({p1:0,p2:0}))==Z
# Independent central differences of realized fields at the matched equilibrium.
fd_error=0.
for g in (gi,gs):
    fn=s.lambdify((t1,t2,p1,p2),g,'numpy')
    for j in range(4):
        e=np.eye(4)[j]*1e-6
        col=(np.asarray(fn(*e),float).ravel()-np.asarray(fn(*(-e)),float).ravel())/(2e-6)
        fd_error=max(fd_error,float(np.max(abs(col-np.array(H.row_join(-K),float)[:,j]))))
assert fd_error<1e-8
# Fixed numerical matrix fixtures. Boundary status tested separately, not rounded to stable.
rng=np.random.default_rng(3718)
ct=dt=0; boundary_error=0.; commutators=[]
for dim in range(1,5):
    eye=np.eye(dim)
    for j in range(20):
        a=rng.normal(size=(dim,dim)); c=rng.normal(size=(dim,dim))
        h=a.T@a if j else np.zeros((dim,dim))
        l=c.T@c+0.3*eye; k=h-l
        commutators.append(float(np.linalg.norm(h@l-l@h)))
        for tv in (.1,1.,10.):
            m=np.block([[-h,k],[eye/tv,-eye/tv]])
            assert max(np.linalg.eigvals(m).real)<-1e-9
            ct+=1
        for av in (.1,.5,.9):
            limit=2*(1+av)/np.linalg.eigvalsh(2*av*h+(1-av)*l)[-1]
            for factor in (.5,.99,1.,1.01,2.):
                ev=factor*limit
                nn=np.block([[eye-ev*h,ev*k],[(1-av)*(eye-ev*h),av*eye+(1-av)*ev*k]])
                eig=np.linalg.eigvals(nn); rho=float(max(abs(eig)))
                if factor<1: assert rho<1-1e-9
                elif factor>1: assert rho>1+1e-9
                else:
                    err=float(min(abs(eig+1))); boundary_error=max(boundary_error,err)
                    assert err<1e-8 and abs(rho-1)<1e-8
                dt+=1
# Cycle-36 positive, zero and negative diagonal defects: no false positive certificate.
rejections=[]
for eps in (-.1,0.,.1):
    h=np.eye(1); l=np.array([[-eps]]); k=h-l
    m=np.block([[-h,k],[h,-h]])
    eigmax=float(max(np.linalg.eigvals(m).real))
    assert (eigmax<0)==(eps<0)
    rejections.append({'epsilon':eps,'largest_real_eigenvalue':eigmax,'L_positive':eps<0})
fixture=[]
for factor in (.99,1.,1.01):
    nn=np.array(N.subs({alpha:s.Rational(1,2),eta:float(bound)*factor}),float)
    fixture.append({'factor':factor,'eta':float(bound)*factor,'spectral_radius':float(max(abs(np.linalg.eigvals(nn))))})
result={'status':'pass','numpy':np.__version__,'sympy':s.__version__,
 'continuous_matrix_cases':ct,'discrete_matrix_cases':dt,'finite_difference_columns':8,
 'max_finite_difference_error':fd_error,'max_minus_one_boundary_error':boundary_error,
 'noncommuting_matrix_pairs':sum(v>1e-10 for v in commutators),
 'noncommuting_example':{'H':[[1,0],[0,4]],'L':[[2,1],[1,2]],'alpha':.5,'eta_boundary_exact':'6/(7+sqrt(10))','cases':fixture},
 'cycle36_rejection_controls':rejections,'cpu_seconds_excluding_imports':time.process_time()-start,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(__file__).with_name('verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
