"""Exact algebra for cycle 12; no integration, neural data, or parameter search."""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import hashlib,json,time
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mv(A,v):return tuple(sum(x*y for x,y in zip(row,v)) for row in A)
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def scale(a,k):return tuple(k*x for x in a)
def transpose(A):return tuple(zip(*A))
def mm(A,B):return tuple(tuple(sum(x*y for x,y in zip(row,col)) for col in transpose(B)) for row in A)
def sign(v):return Q(int(v>0)-int(v<0))
def rank(A):
    A=[list(map(Q,row)) for row in A];i=0
    for j in range(len(A[0])):
        pivot=next((k for k in range(i,len(A)) if A[k][j]),None)
        if pivot is None:continue
        A[i],A[pivot]=A[pivot],A[i];p=A[i][j];A[i]=[x/p for x in A[i]]
        for k in range(len(A)):
            if k!=i:
                p=A[k][j];A[k]=[x-p*y for x,y in zip(A[k],A[i])]
        i+=1
        if i==len(A):break
    return i
def main():
    start=time.process_time();expected='991ac18a54beb2d9c2ed97d347e6db55bd813d892ecf56aab2c8261c6fd58f01'
    assert sha(ROOT/'PROTOCOL.md')==expected
    A=((Q(0),Q(0)),(Q(1),Q(1)));I=((Q(1),Q(0)),(Q(0),Q(1)))
    h=(Q(1),Q(-1));zero=(Q(0),Q(0));R=tuple(sub(x,y) for x,y in zip(A,I))
    Mstop=tuple(scale(row,Q(1,2)) for row in I)
    Mfull=tuple(scale(row,Q(-1,2)) for row in transpose(R))
    assert mm(A,A)==A and mv(A,h)==zero and rank(R)==1
    def forward(w,z):return add(add(h,w),mv(A,z))
    def residual(theta,phi=zero,cached=False):
        q=forward(theta,zero);z=h if cached else q
        return sub(forward(phi,z),q)
    records=[]
    # Finite exact substitutions verify formulas; the proof covers all states.
    for u,v in product([Q(-2),Q(-1,3),Q(0),Q(1,3),Q(2)],repeat=2):
        theta=(u,v);r=residual(theta);rc=residual(theta,cached=True)
        assert r==(-u,u) and rc==(-u,-v)
        assert forward(zero,forward(zero,theta))==forward(zero,theta)
        sig=tuple(map(sign,r));fstop=mv(Mstop,sig);ffull=mv(Mfull,sig)
        fcached=mv(Mstop,tuple(map(sign,rc)))
        assert fstop==(-sign(u)/2,sign(u)/2) and ffull==(-sign(u),Q(0))
        assert fcached==(-sign(u)/2,-sign(v)/2)
        assert sum(fstop)==0
        if u:
            eps=Q(1,10000)
            for j in range(2):
                plus=list(theta);minus=list(theta);plus[j]+=eps;minus[j]-=eps
                loss=lambda p:sum(abs(x) for x in residual(p))/2
                assert -(loss(plus)-loss(minus))/(2*eps)==ffull[j]
        records.append({'theta':list(map(str,theta)),'residual':list(map(str,r)),'stop':list(map(str,fstop)),'full':list(map(str,ffull)),'cached':list(map(str,fcached))})
    patterns=[(Q(-1),Q(1)),(Q(1),Q(-1))]
    corners=list(product([Q(-1),Q(1)],repeat=2))
    true_stop=[mv(Mstop,s) for s in patterns];box_stop=[mv(Mstop,s) for s in corners]
    true_full=set(mv(Mfull,s) for s in patterns);box_full=set(mv(Mfull,s) for s in corners)
    assert all(sum(v)==0 for v in true_stop)
    spurious=(Q(0),Q(1,2));assert all(abs(x)<=Q(1,2) for x in spurious) and sum(spurious)!=0
    assert true_full=={(Q(-1),Q(0)),(Q(1),Q(0))}
    assert box_full=={(Q(-1),Q(0)),(Q(0),Q(0)),(Q(1),Q(0))}
    path_checks=[]
    for t in [Q(0),Q(1,3),Q(1),Q(2)]:
        theta=(Q(0),t/2);assert residual(theta)==zero
        assert mv(Mstop,(Q(0),Q(1)))==spurious
        risk=sum(x*x for x in theta)/2;assert risk==t*t/8
        path_checks.append({'t':str(t),'theta':list(map(str,theta)),'risk':str(risk)})
    # With a moving reference, active residuals depend independently on its coordinates.
    Rema=((Q(-1),Q(0),Q(1),Q(0)),(Q(1),Q(0),Q(0),Q(1)))
    assert rank(Rema)==2
    for s in corners:
        eps=Q(1,100);phi=scale(s,eps)
        assert tuple(map(sign,residual(zero,phi)))==s
    # Cached residual map is -I: all signs are genuinely reachable in student state.
    for s in corners:assert tuple(map(sign,residual(scale(s,Q(-1,100)),cached=True)))==s
    report={'status':'passed','protocol_sha256':expected,'script_sha256':sha(Path(__file__)),
            'checks':'Exact identities and chamber images only; analytic proofs establish closure and solution claims.',
            'state_substitutions':len(records),'records':records,'idempotent_reference':True,
            'fixed_residual_jacobian_rank':rank(R),'ema_residual_jacobian_rank':rank(Rema),
            'reachable_fixed_signs':[list(map(str,s)) for s in patterns],
            'true_stop_vertices':[list(map(str,v)) for v in true_stop],
            'box_stop_vertices':[list(map(str,v)) for v in box_stop],
            'true_full_vertices':[list(map(str,v)) for v in sorted(true_full)],
            'box_full_images':[list(map(str,v)) for v in sorted(box_full)],
            'spurious_box_path_derivative':list(map(str,spurious)),'path_checks':path_checks,
            'neural_updates':0,'integrated_trajectories':0,'cpu_seconds':time.process_time()-start}
    assert report['cpu_seconds']<30
    (ROOT/'closure_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
if __name__=='__main__':main()
