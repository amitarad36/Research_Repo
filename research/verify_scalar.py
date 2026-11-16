"""Reproduce algebraic identities and numerical checks for the research note.

Run with Python containing sympy, numpy, scipy. Numerical checks are evidence,
not substitutes for the inequalities proved in Scalar_Certificate_Result.md.
"""
import json
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def symbolic_checks():
    u, v, p, c, lam, gamma = sp.symbols('u v p c lam gamma', real=True)
    x = sp.Matrix([u, v])
    n = u*u + v*v
    m = n + lam
    sig = 2*c*u*v
    h = sig/m
    field = sp.Matrix([-(p*p+lam)*u+p*c*v,
                       -(p*p+lam)*v+p*c*u,
                       gamma*(sig-p*m)])
    V = m**3/3-sig**2/2
    grad = sp.Matrix([sp.diff(V,u),sp.diff(V,v)])
    K = (m*sp.eye(2)-x*x.T)/(2*m*m)
    assert all(sp.factor(z)==0 for z in field[:2,0].subs(p,h)+K*grad)
    D = n-p*p/gamma
    Ddot = sum(sp.diff(D,z)*field[i] for i,z in enumerate((u,v,p)))
    assert sp.factor(Ddot+2*lam*n-2*lam*p*p)==0
    e = sp.symbols('e', real=True)
    delta = -(2*h*e+e*e)*x+c*e*sp.Matrix([v,u])
    assert all(sp.factor(z)==0 for z in
               field[:2,0].subs(p,h+e)-field[:2,0].subs(p,h)-delta)
    q = sp.symbols('q', positive=True)
    J = field.jacobian((u,v,p)).subs({u:q/sp.sqrt(2),v:q/sp.sqrt(2),
                                    p:q,lam:q*(c-q)})
    T = sp.Matrix([[1/sp.sqrt(2),1/sp.sqrt(2),0],
                   [1/sp.sqrt(2),-1/sp.sqrt(2),0],[0,0,1]])
    expected = sp.Matrix([[0,0,q*(c-2*q)], [0,-2*c*q,0],
                          [2*gamma*q*(c-q),0,-gamma*c*q]])
    assert all(sp.simplify(z)==0 for z in T*J*T.T-expected)
    return {'metric_identity': True, 'balance_identity': True,
            'tracking_perturbation_identity': True, 'modal_jacobian': True}


def quantities(x, c, lam):
    n = float(x@x)
    m = n+lam
    Sx = c*x[::-1]
    sig = float(x@Sx)
    h = sig/m
    grad = 2*(m*m*x-sig*Sx)
    dh = 2*(Sx-h*x)/m
    V = m**3/3-sig*sig/2
    f0 = -(h*h+lam)*x+h*Sx
    return n,m,h,grad,dh,V,f0


def numerical_checks():
    rng = np.random.default_rng(4217)
    c,lam,a = .8,.1,1.
    initial = np.array([.3,.2,.5])
    n0 = float(initial[:2]@initial[:2])
    Pmax = max(abs(initial[2]),c)
    R = n0+Pmax**2
    E = Pmax+c
    b = np.sqrt(R)*(3*c+E)
    H = 2*c/np.sqrt(lam)
    L = 2*(R+lam)**2/lam
    threshold = H*b/lam+(L*b+a*H)**2/(4*a*lam**2)
    gamma = max(1.,2*threshold)
    dissipation = a*(gamma*lam-H*b)-(L*b+a*H)**2/(4*lam)
    max_violation = -float('inf')
    for _ in range(10000):
        angle = rng.uniform(0,2*np.pi)
        radius = np.sqrt(R*rng.uniform())
        x = radius*np.array([np.cos(angle),np.sin(angle)])
        err = rng.uniform(-E,E)
        n,m,h,g,dh,V,f0 = quantities(x,c,lam)
        p = h+err
        f = -(p*p+lam)*x+p*c*x[::-1]
        edot = -gamma*m*err-dh@f
        Wdot = g@f+a*err*edot
        violation = Wdot+lam*(f0@f0)+dissipation*err*err
        max_violation = max(max_violation,float(violation))
        assert violation <= 1e-7
        assert 2*lam*(f0@f0) <= -(g@f0)+1e-9
        assert np.linalg.norm(g) <= L*np.linalg.norm(f0)+1e-9
        assert np.linalg.norm(dh) <= H+1e-9

    qminus = (c-np.sqrt(c*c-4*lam))/2
    qplus = (c+np.sqrt(c*c-4*lam))/2
    spectra = []
    for speed in [.01,1.,100.]:
        for branch,q in [('lower',qminus),('upper',qplus)]:
            J = np.array([[0,0,q*(c-2*q)],[0,-2*c*q,0],
                          [2*speed*q*(c-q),0,-speed*c*q]])
            real = np.linalg.eigvals(J).real
            assert (np.max(real)>0) if branch=='lower' else (np.max(real)<0)
            spectra.append({'gamma':speed,'branch':branch,
                            'real_parts':real.tolist()})

    simulations=[]
    for speed in [.1,1.,10.,gamma]:
        def rhs(t,z):
            u,v,p=z
            return [-(p*p+lam)*u+p*c*v,
                    -(p*p+lam)*v+p*c*u,
                    speed*(2*c*u*v-p*(u*u+v*v+lam))]
        sol=solve_ivp(rhs,[0,500],initial,method='Radau',
                      rtol=1e-9,atol=1e-11,dense_output=True)
        assert sol.success
        samples=sol.sol(np.linspace(0,500,2001)).T
        ns=np.sum(samples[:,:2]**2,axis=1)
        D0=n0-initial[2]**2/speed
        bound=Pmax**2/speed+max(D0,max(1-1/speed,0)*Pmax**2)
        assert np.max(ns)<=bound+1e-7
        assert np.max(np.abs(samples[:,2]))<=Pmax+1e-7
        residual=float(np.linalg.norm(rhs(500,sol.y[:,-1])))
        assert residual<1e-7
        simulations.append({'gamma':speed,'terminal_state':sol.y[:,-1].tolist(),
                            'terminal_velocity_norm':residual,
                            'max_sampled_encoder_norm_squared':float(np.max(ns)),
                            'proved_encoder_bound':float(bound)})
    return {'parameters':{'c':c,'lambda':lam,'a':a,'initial':initial.tolist()},
            'certificate_constants':{'R':R,'E':E,'b':float(b),'H':H,'L':L},
            'sufficient_speed_threshold':float(threshold),
            'checked_speed':float(gamma),'random_points':10000,
            'max_dissipation_inequality_violation':max_violation,
            'local_stability_checks':spectra,'simulations':simulations}


def matrix_identity_checks():
    rng=np.random.default_rng(773)
    max_balance_error=0.
    max_tracking_error=0.
    for _ in range(1000):
        A,B,P=[rng.normal(size=(2,2)) for _ in range(3)]
        C=np.diag([.8,.4]); lam=.2; gamma=3.
        N=A@A.T+B@B.T
        Sigma=B@C@A.T+A@C@B.T
        M=N+lam*np.eye(2)
        inv=np.linalg.inv(M)
        h=Sigma@inv
        FA=-(P.T@P+lam*np.eye(2))@A+P.T@B@C
        FB=-(P.T@P+lam*np.eye(2))@B+P.T@A@C
        FP=gamma*(Sigma-P@M)
        lhs=2*np.sum(A*FA)+2*np.sum(B*FB)-2*np.sum(P*FP)/gamma
        rhs=-2*lam*(np.sum(A*A)+np.sum(B*B))+2*lam*np.sum(P*P)
        max_balance_error=max(max_balance_error,abs(float(lhs-rhs)))
        assert abs(lhs-rhs)<1e-9
        DN=FA@A.T+A@FA.T+FB@B.T+B@FB.T
        DS=FB@C@A.T+B@C@FA.T+FA@C@B.T+A@C@FB.T
        Dh=DS@inv-h@DN@inv
        error=FP-Dh-(-gamma*(P-h)@M-Dh)
        max_tracking_error=max(max_tracking_error,float(np.linalg.norm(error)))
        assert np.linalg.norm(error)<1e-9
    return {'random_nonsymmetric_states':1000,
            'max_balance_identity_error':max_balance_error,
            'max_tracking_identity_error':max_tracking_error}


if __name__=='__main__':
    results={'symbolic':symbolic_checks(),'numerical':numerical_checks(),
             'matrix_identity_checks':matrix_identity_checks()}
    output=Path(__file__).with_name('scalar_verification.json')
    output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'all_checks_passed':True,'results':str(output),
                      'threshold':results['numerical']['sufficient_speed_threshold']},indent=2))
