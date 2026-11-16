"""Bounded synthetic IT3-style geometry test. See immutable pre-run PROTOCOL.md.
Run with Python + NumPy only; output goes next to this script.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import json
import time
import hashlib
import platform
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
HIDDEN = 16
P = 114


def pack(w1, b1, w2, b2):
    return np.concatenate([w1.ravel(), b1.ravel(), w2.ravel(), b2.ravel()])


def unpack(p):
    return p[:64].reshape(4, 16), p[64:80], p[80:112].reshape(16, 2), p[112:]


def forward(p, x, z):
    w1, b1, w2, b2 = unpack(p)
    u = np.concatenate([x, z], axis=1)
    h = np.tanh(u @ w1 + b1)
    return h @ w2 + b2, (u, h, w2)


def vjp(cache, v):
    """Average J_parameter^T v; v has one row per observation."""
    u, h, w2 = cache
    dh = (v @ w2.T) * (1 - h*h)
    n = len(u)
    return pack(u.T @ dh/n, dh.mean(axis=0), h.T @ v/n, v.mean(axis=0))


def jac_z(p, cache):
    w1, _, w2, _ = unpack(p)
    _, h, _ = cache
    return np.einsum('ho,nh,kh->nok', w2, 1-h*h, w1[2:, :])


def parameter_jacobian(p, x, z):
    _, (u, h, w2) = forward(p, x, z)
    j = np.zeros((len(x), 2, P))
    for o in range(2):
        dh = (1-h*h)*w2[:, o]
        j[:, o, :64] = np.einsum('ni,nh->nih', u, dh).reshape(len(x),64)
        j[:, o, 64:80] = dh
        j[:, o, 80+o:112:2] = h
        j[:, o, 112+o] = 1
    return j


def labels(x):
    a,b=x.T
    return np.column_stack([1+.3*a+.15*a*b, .8+.3*b+.15*np.sin(np.pi*a)])


def init(seed):
    rng=np.random.default_rng(seed)
    p=pack(rng.normal(size=(4,16))*np.sqrt(2/20), np.zeros(16),
           rng.normal(size=(16,2))*np.sqrt(2/18), np.zeros(2))
    return p,rng


def loss(p,x,z,y):
    q,c=forward(p,x,z)
    e=q-y
    return .5*np.mean(np.sum(e*e,axis=1)),vjp(c,e)


def velocity(theta,phi,x,rule):
    q,cq=forward(theta,x,np.zeros_like(x))
    t,ct=forward(phi,x,q)
    r=t-q
    if rule=='input':
        v=r-np.einsum('nok,no->nk',jac_z(phi,ct),r)
    else:
        assert rule=='stop'
        v=r
    return vjp(cq,v)


def directional_fd(fun,p,d,eps=1e-6):
    return (fun(p+eps*d)-fun(p-eps*d))/(2*eps)


def check_derivatives(p,x,y,seed):
    rng=np.random.default_rng(10000+seed)
    d=rng.normal(size=P); d/=np.linalg.norm(d)
    phi=p+.01*rng.normal(size=P)
    z=y*.9
    l,g=loss(p,x,z,y)
    err_train=abs(directional_fd(lambda t:loss(t,x,z,y)[0],p,d)-g@d)
    q,cq=forward(p,x,np.zeros_like(x))
    t,ct=forward(phi,x,q)
    az=jac_z(phi,ct)
    dz=rng.normal(size=q.shape)
    fz=(forward(phi,x,q+1e-6*dz)[0]-forward(phi,x,q-1e-6*dz)[0])/2e-6
    err_z=np.max(np.abs(fz-np.einsum('nok,nk->no',az,dz)))
    jq=parameter_jacobian(p,x,np.zeros_like(x))
    fq=directional_fd(lambda th:forward(th,x,np.zeros_like(x))[0],p,d)
    err_j=np.max(np.abs(fq-np.einsum('nop,p->no',jq,d)))
    errors={'supervised_directional_abs':float(err_train), 'input_jacobian_abs':float(err_z),
            'parameter_jacobian_abs':float(err_j)}
    for rule in ['input','stop']:
        def objective(th):
            q2=forward(th,x,np.zeros_like(x))[0]
            target=forward(phi,x,q2)[0] if rule=='input' else t
            return .5*np.mean(np.sum((target-q2)**2,axis=1))
        fd=directional_fd(objective,p,d)
        errors[rule+'_loss_directional_abs']=float(abs(fd+velocity(p,phi,x,rule)@d))
    assert max(errors.values()) < 1e-7,errors
    return errors


def train(seed,x,y):
    p,rng=init(seed)
    checks=check_derivatives(p,x[:8],y[:8],seed)
    m=np.zeros(P);v=np.zeros(P);hist=[]
    for step in range(1,6001):
        ids=rng.integers(0,len(x),size=128)
        xb,yb=x[ids],y[ids]
        xx=np.concatenate([xb,xb]); zz=np.concatenate([np.zeros_like(yb),yb]); yy=np.concatenate([yb,yb])
        l,g=loss(p,xx,zz,yy)
        m=.9*m+.1*g;v=.999*v+.001*g*g
        lr=.001 if step<=4500 else .0003
        p-=lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step%1000==0:hist.append({'step':step,'minibatch_loss':float(l)})
    return p,hist,checks


def summary(a):
    a=np.asarray(a)
    return {k:float(v) for k,v in zip(['min','median','mean','max'],[a.min(),np.median(a),a.mean(),a.max()])}


def geometry(p,x,z):
    _,cache=forward(p,x,z)
    a=jac_z(p,cache)
    eig=np.linalg.eigvals(a)
    real=eig[:,0].real;imag=np.abs(eig[:,0].imag)
    rotational=(imag>1e-6)&(real>0)&(real<1)&(imag**2-real*(1-real)>1e-6)
    norm_f=np.linalg.norm(a,axis=(1,2))
    defect=np.linalg.norm(a@a-a,axis=(1,2))
    metrics={'operator_norm':summary(np.linalg.norm(a,ord=2,axis=(1,2))),
        'skew_frobenius':summary(np.linalg.norm((a-a.transpose(0,2,1))/2,axis=(1,2))),
        'idempotence_defect':summary(defect), 'relative_idempotence_defect':summary(defect/(norm_f+1e-12)),
        'complex_fraction':float(np.mean(imag>1e-6)), 'rotation_gate_fraction':float(np.mean(rotational)),
        'eigen_real':summary(eig.real),'eigen_imag_abs':summary(np.abs(eig.imag))}
    return metrics,a


def field_derivatives(p,x,rule,eps):
    ds=np.empty((P,P));dt=np.empty((P,P))
    for i in range(P):
        e=np.zeros(P);e[i]=eps
        ds[:,i]=(velocity(p+e,p,x,rule)-velocity(p-e,p,x,rule))/(2*eps)
        dt[:,i]=(velocity(p,p+e,x,rule)-velocity(p,p-e,x,rule))/(2*eps)
    return ds,dt


def full_check(p,x,seed):
    rng=np.random.default_rng(20000+seed)
    dq=rng.normal(size=P);dq/=np.linalg.norm(dq)
    dr=rng.normal(size=P);dr/=np.linalg.norm(dr)
    q=forward(p,x,np.zeros_like(x))[0]
    r=forward(p,x,q)[0]-q
    jq=parameter_jacobian(p,x,np.zeros_like(x)).reshape(-1,P)
    sj=np.linalg.svd(jq,compute_uv=False)
    out={'residual_rms':float(np.sqrt(np.mean(np.sum(r*r,axis=1)))),
         'parameter_jacobian_singular_values':sj.tolist(),
         'parameter_jacobian_rank_tol_1e-8_relative':int(np.sum(sj>sj[0]*1e-8)),
         'rules':{}}
    arrays={}
    for rule in ['input','stop']:
        ds,dt=field_derivatives(p,x,rule,1e-5)
        ds2,dt2=field_derivatives(p,x,rule,5e-6)
        err=float(max(np.max(np.abs(ds-ds2)),np.max(np.abs(dt-dt2))))
        fd=(velocity(p+1e-5*dq,p+1e-5*dr,x,rule)-velocity(p-1e-5*dq,p-1e-5*dr,x,rule))/2e-5
        directional_error=float(np.max(np.abs(fd-ds@dq-dt@dr)))
        assert err<1e-7 and directional_error<1e-7,(err,directional_error)
        vel=velocity(p,p,x,rule)
        predict_velocity=jq@vel
        diag={'velocity_norm':float(np.linalg.norm(vel)),
            'instantaneous_prediction_velocity_rms':float(np.linalg.norm(predict_velocity)/np.sqrt(len(x))),
            'fd_step_max_abs_difference':err,'field_directional_abs_error':directional_error,
            'spectra':[]}
        for tau in [1,20]:
            eye=np.eye(P)
            m=np.block([[ds,dt],[eye/tau,-eye/tau]])
            eigen=np.linalg.eigvals(m)
            eta=.1;alpha=tau/(tau+eta)
            jd=np.block([[eye+eta*ds,eta*dt],[(1-alpha)*(eye+eta*ds),alpha*eye+(1-alpha)*eta*dt]])
            evd=np.linalg.eigvals(jd)
            diag['spectra'].append({'tau':tau,'max_real':float(eigen.real.max()),
              'rhp_count_above_1e-6':int(np.sum(eigen.real>1e-6)),
              'near_zero_count_abs_below_1e-6':int(np.sum(np.abs(eigen)<1e-6)),
              'discrete_spectral_radius':float(np.max(np.abs(evd)))})
            arrays[f'{rule}_tau{tau}_ode_eigen']=eigen
            arrays[f'{rule}_tau{tau}_discrete_eigen']=evd
        arrays[rule+'_dstudent']=ds;arrays[rule+'_dteacher']=dt
        out['rules'][rule]=diag
    vi=velocity(p,p,x,'input');vs=velocity(p,p,x,'stop')
    out['relative_velocity_difference']=float(np.linalg.norm(vi-vs)/(np.linalg.norm(vs)+1e-15))
    out['velocity_cosine']=float(vi@vs/(np.linalg.norm(vi)*np.linalg.norm(vs)+1e-30))
    return out,arrays


def main():
    start=time.perf_counter()
    x=np.random.default_rng(314159).uniform(-1,1,size=(256,2)); y=labels(x)
    xt=np.random.default_rng(271828).uniform(-1,1,size=(256,2));yt=labels(xt)
    np.savez_compressed(ROOT/'shared_data.npz',x_train=x,y_train=y,x_eval=xt,y_eval=yt)
    variance=float(np.mean(np.sum((yt-yt.mean(axis=0))**2,axis=1)))
    report={'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
                       'dtype':'float64','blas_thread_environment':{k:os.environ[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}},
        'label_total_variance':variance,'seeds':[]}
    for seed in range(5):
        t0=time.perf_counter()
        p,hist,checks=train(seed,x,y)
        q=forward(p,xt,np.zeros_like(xt))[0]
        qy=forward(p,xt,yt)[0]
        nmse_q=float(np.mean(np.sum((q-yt)**2,axis=1))/variance)
        nmse_y=float(np.mean(np.sum((qy-yt)**2,axis=1))/variance)
        gq,aq=geometry(p,xt,q);gy,ay=geometry(p,xt,yt)
        full,arrays=full_check(p,xt[:64],seed)
        checks_final=check_derivatives(p,xt[:8],yt[:8],seed)
        result={'seed':seed,'zero_input_nmse':nmse_q,'label_input_nmse':nmse_y,
            'fit_pass':bool(max(nmse_q,nmse_y)<=.05),'geometry_at_prediction':gq,'geometry_at_label':gy,
            'full_parameter_check':full,'derivative_checks_initial':checks,'derivative_checks_final':checks_final,
            'training_history':hist,'wall_seconds':time.perf_counter()-t0}
        result['rotation_pass']=bool(result['fit_pass'] and min(gq['rotation_gate_fraction'],gy['rotation_gate_fraction'])>=.1)
        np.savez_compressed(ROOT/f'seed_{seed}.npz',parameters=p,A_at_prediction=aq,A_at_label=ay,**arrays)
        report['seeds'].append(result)
        (ROOT/'neural_gate_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'seed':seed,'zero_nmse':nmse_q,'label_nmse':nmse_y,
              'rotation_fraction':gq['rotation_gate_fraction'],'A_operator_norm':gq['operator_norm'],
              'seconds':result['wall_seconds']}),flush=True)
    report['fit_gate_pass']=sum(s['fit_pass'] for s in report['seeds'])>=3
    report['rotation_gate_pass']=sum(s['rotation_pass'] for s in report['seeds'])>=3
    report['wall_seconds']=time.perf_counter()-start
    (ROOT/'neural_gate_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['fit_gate_pass','rotation_gate_pass','wall_seconds']}),flush=True)


if __name__=='__main__':
    main()
