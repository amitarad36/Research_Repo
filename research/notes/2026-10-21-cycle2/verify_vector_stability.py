"""Exact linear certificates and nonlinear checks for the constructed two-view model.

Python stdlib + numpy only. No network, training data, or external compute.
The model embeds a prescribed correct pretrained predictor; it is not a trained IT3 net.
"""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parent

def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]

def mm(A, B):
    return [[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]

def charpoly(A):
    n=len(A); Q=eye(n); cs=[F(1)]
    for k in range(1,n+1):
        Q=mm(A,Q); c=-sum(Q[i][i] for i in range(n))/k
        cs.append(c)
        for i in range(n): Q[i][i]+=c
    assert all(v==0 for row in Q for v in row), 'Cayley-Hamilton check failed'
    return cs

def routh(cs):
    if cs[0]<0: cs=[-c for c in cs]
    n=len(cs)-1; width=(n+2)//2
    a=cs[::2]+[F(0)]*width; b=cs[1::2]+[F(0)]*width
    rows=[a[:width],b[:width]]
    for _ in range(2,n+1):
        a,b=rows[-2:]
        assert b[0]!=0, 'Degenerate Routh pivot needs separate treatment'
        row=[(b[0]*a[j+1]-a[0]*b[j+1])/b[0] for j in range(width-1)]+[F(0)]
        assert any(row), 'Zero Routh row needs separate treatment'
        rows.append(row)
    col=[r[0] for r in rows]
    assert all(col)
    signs=[1 if a>0 else -1 for a in col]
    return col, sum(a!=b for a,b in zip(signs,signs[1:]))

def bilinear(cs):
    # q(s)=(1-s)^n p((1+s)/(1-s)); maps unit disk to left half-plane.
    n=len(cs)-1; out=[F(0)]*(n+1)
    for j,c in enumerate(cs):
        for a in range(n-j+1):
            for b in range(j+1):
                out[a+b]+=c*comb(n-j,a)*comb(j,b)*(-1)**b
    assert out[-1]!=0, 'Eigenvalue -1 requires separate treatment'
    return out[::-1]

def exact_blocks(delta, tau, rule):
    xs=[[F(1),F(0)],[F(3,5),F(4,5)]]
    Bs=[[[F(1,2),F(-1)],[F(1),F(1,2)]],
        [[F(1,2)+delta,F(-1)],[F(1),F(1,2)-delta]]]
    H=[[F(0) for _ in range(4)] for _ in range(4)]
    K=[[F(0) for _ in range(4)] for _ in range(4)]
    for x,B in zip(xs,Bs):
        BT=list(map(list,zip(*B)))
        hh=mm(BT,B) if rule=='input' else B
        kk=BT if rule=='input' else eye(2)
        for i in range(2):
            for j in range(2):
                for a in range(2):
                    for b in range(2):
                        H[2*i+a][2*j+b]+=x[i]*x[j]*hh[a][b]/2
                        K[2*i+a][2*j+b]+=x[i]*x[j]*kk[a][b]/2
    I=eye(4)
    M=[[-v for v in H[i]]+K[i] for i in range(4)]
    M += [[v/tau for v in I[i]]+[-v/tau for v in I[i]] for i in range(4)]
    eta=F(1,10); alpha=tau/(tau+eta)
    U=[[I[i][j]-eta*H[i][j] for j in range(4)] for i in range(4)]
    V=[[eta*K[i][j] for j in range(4)] for i in range(4)]
    D=[U[i]+V[i] for i in range(4)]
    D += [[(1-alpha)*v for v in U[i]]+
          [alpha*I[i][j]+(1-alpha)*V[i][j] for j in range(4)] for i in range(4)]
    return M,D,H,K

def npmat(A): return np.asarray(A,dtype=float)

xs=np.array([[1.,0.],[.6,.8]])
ys=np.eye(2) # Different nonzero predictions at the two input locations.
rot=np.array([[0.,-1.],[1.,0.]])

def base_model(W, x, y, A, z):
    return y+W@x + A@(z-y)*float(y@z/(y@y))

def velocity(state, delta, tau, rule):
    W=state[:4].reshape((2,2),order='F'); V=state[4:].reshape((2,2),order='F')
    grad=np.zeros((2,2))
    for i,(x,y) in enumerate(zip(xs,ys)):
        B=.5*np.eye(2)+rot+(delta*np.diag([1.,-1.]) if i else 0.)
        A=np.eye(2)-B; u=W@x; v=V@x; ell=y/(y@y)
        r=v-u+A@u*(1+ell@u)
        if rule=='input':
            dr=-np.eye(2)+A@((1+ell@u)*np.eye(2)+np.outer(u,ell))
            grad-=.5*np.outer(dr.T@r,x)
        else: grad+=.5*np.outer(r,x)
    return np.r_[grad.flatten(order='F'),((W-V)/tau).flatten(order='F')]

def risk(state):
    W=state[:4].reshape((2,2),order='F')
    return float(sum(np.linalg.norm(W@x)**2 for x in xs)/4)

def rk4(state,delta,tau,rule,dt,steps):
    s=state.copy(); initial=risk(s); maximum=initial
    for _ in range(steps):
        f=lambda z: velocity(z,delta,tau,rule)
        a=f(s); b=f(s+dt*a/2); c=f(s+dt*b/2); d=f(s+dt*c)
        s+=dt*(a+2*b+2*c+d)/6
        assert np.isfinite(s).all()
        maximum=max(maximum,risk(s))
    return {'initial_risk':initial,'final_risk':risk(s),'max_risk':maximum,
            'final_state_norm':float(np.linalg.norm(s)), 'final_state':s.tolist()}

def main():
    records=[]; maxjac=0.; maxpoly=0.
    for delta in [F(0),F(1,10)]:
        for tau in [F(1),F(20)]:
            for rule in ['input','stop']:
                M,D,H,K=exact_blocks(delta,tau,rule)
                cs=charpoly(M); col,count=routh(cs)
                ds=charpoly(D); dcol,dcount=routh(bilinear(ds))
                ev=np.linalg.eigvals(npmat(M)); dev=np.linalg.eigvals(npmat(D))
                expected=4 if rule=='stop' and tau==1 else 0
                assert count==expected and dcount==expected
                assert sum(ev.real>1e-8)==count
                assert sum(abs(dev)>1+1e-8)==dcount
                maxpoly=max(maxpoly,float(np.max(np.abs(np.array([float(c) for c in cs])-np.poly(npmat(M))))))
                # Independent finite differences of the actual smooth shared architecture's field.
                eps=1e-6; fd=np.column_stack([(velocity(eps*e,float(delta),float(tau),rule)-velocity(-eps*e,float(delta),float(tau),rule))/(2*eps) for e in np.eye(8)])
                err=float(np.max(np.abs(fd-npmat(M)))); maxjac=max(maxjac,err)
                assert err<1e-9
                records.append({'delta':str(delta),'tau':str(tau),'rule':rule,
                    'characteristic_coefficients':[str(c) for c in cs],
                    'routh_first_column':[str(c) for c in col], 'right_half_plane_roots':count,
                    'max_real_eigenvalue':float(max(ev.real)),
                    'eta':'1/10','alpha':str(tau/(tau+F(1,10))),
                    'discrete_characteristic_coefficients':[str(c) for c in ds],
                    'bilinear_routh_first_column':[str(c) for c in dcol],
                    'outside_unit_disk_roots':dcount,'spectral_radius':float(max(abs(dev))),
                    'nonlinear_field_jacobian_error':err})
    # Verify the nonconstant correct base prediction and its fixed point using the full architecture.
    for i,(x,y) in enumerate(zip(xs,ys)):
        B=.5*np.eye(2)+rot+(.1*np.diag([1.,-1.]) if i else 0.)
        A=np.eye(2)-B; zero=np.zeros((2,2))
        assert np.array_equal(base_model(zero,x,y,A,np.zeros(2)),y)
        assert np.array_equal(base_model(zero,x,y,A,y),y)
    # Closed-form normal-mode threshold, checked on a deterministic grid away from equality.
    grid=0
    for mu in [.2,.8,1.]:
        for tau in [.1,1.,3.,7.,11.,20.,100.]:
            B=.5*np.eye(2)+rot; I=np.eye(2)
            for rule in ['input','stop']:
                H=mu*(B.T@B if rule=='input' else B); K=mu*(B.T if rule=='input' else I)
                M=np.block([[-H,K],[I/tau,-I/tau]])
                stable=max(np.linalg.eigvals(M).real)<-1e-9
                assert stable==(rule=='input' or tau>2/mu)
                grid+=1
    # Predeclared nonlinear check after exploratory eigenvalue search: matched student/teacher
    # perturbations recover for input/tau=1 and stop/tau=20, grow for stop/tau=1.
    perturb=1e-6*np.array([1.,-.4,.2,.7]); init=np.r_[perturb,perturb]
    sims=[]
    for rule,tau,T in [('input',1.,50.),('stop',1.,20.),('stop',20.,100.)]:
        coarse=rk4(init,.1,tau,rule,.02,round(T/.02))
        fine=rk4(init,.1,tau,rule,.01,round(T/.01))
        rel=abs(coarse['final_risk']-fine['final_risk'])/max(fine['final_risk'],1e-30)
        assert rel<1e-6
        assert (fine['final_risk']>fine['initial_risk'])==(rule=='stop' and tau==1.)
        sims.append({'rule':rule,'tau':tau,'duration':T,'dt':.01,
                     'half_step_relative_risk_difference':rel,**fine})
    out={'status':'passed','scope':'Constructed correct nonlinear equilibrium; exact rational local ODE/discrete certificates, numerical nonlinear checks. No IT3 empirical or novelty claim.',
         'exact_cases':len(records),'normal_mode_grid_cases':grid,
         'max_nonlinear_jacobian_error':maxjac,'max_charpoly_float_comparison_error':maxpoly,
         'certificates':records,'nonlinear_checks':sims}
    (OUT/'vector_stability_verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['certificates','nonlinear_checks']},indent=2))
    print(json.dumps(sims,indent=2))

if __name__=='__main__': main()
