"""Cycle 15 exact local graph algebra, not a neural adaptation experiment."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
HERE=Path(__file__).resolve().parent
ROOT=Path(r'C:\Users\amita\source\repos\Detach Proposal Presentation')
C9=ROOT/'research/notes/2026-10-26-cycle9'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sign(x):return F((x>0)-(x<0))
def target(z):
    C=[[F(2),F(1,3)],[F(-1,2),F(3,2)]]; h=[F(3),F(4)]
    t=[h[i]+sum(C[i][j]*max(z[j],0) for j in range(2)) for i in range(2)]
    A=[[C[i][j]*(z[j]>0) for j in range(2)] for i in range(2)]
    return t,A
def velocity(q,z,b,kind):
    t,A=target(z); r=[t[i]-q[i] for i in range(2)]
    w=r if kind=='squared' else list(map(sign,r))
    scale=F(1,2) if kind=='squared' else F(1,4) # N=2,d=2
    return [scale*(w[j]-b*sum(A[i][j]*w[i] for i in range(2))) for j in range(2)]
def local_loss(qbase,zbase,qpert,b,kind):
    zz=[zbase[i]+b*(qpert[i]-qbase[i]) for i in range(2)]
    t,_=target(zz); r=[t[i]-qpert[i] for i in range(2)]
    return sum(v*v for v in r)/4 if kind=='squared' else sum(abs(v) for v in r)/4
def main():
    start=time.process_time(); assert not (HERE/'graph_verification.json').exists()
    man=json.loads((C9/'source_manifest.json').read_text())
    assert sha(C9/man['file'])==man['sha256']; assert sha(C9/man['text_file'])==man['text_sha256']
    nb=json.loads((C9/man['file']).read_text(encoding='utf-8'))
    cell=''.join(nb['cells'][13]['source'])
    for token in ['model_original.forward(x,y1)','(y1 - y2).abs().mean()',
                  'f_ttt.load_state_dict(f.state_dict())','optim.Adam(f_ttt.parameters(), lr=lr)',
                  'n_steps=2, lr=1e-4']:
        assert token in cell
    checks=0; matched=0; decomposition=0
    fixture_q=[[F(1,2),F(2,3)],[F(-1,2),F(1,3)],[F(3,4),F(-2,3)]]
    fixture_c=[[F(1),F(1)],[F(-1),F(1)]]
    eps=F(1,100000)
    for q in fixture_q:
        for c in fixture_c:
            for kind in ['squared','absolute']:
                for z in [q,c]:
                    for b in [0,1]:
                        u=velocity(q,z,b,kind)
                        for j in range(2):
                            plus=q.copy();minus=q.copy();plus[j]+=eps;minus[j]-=eps
                            numerical=(local_loss(q,z,plus,b,kind)-local_loss(q,z,minus,b,kind))/(2*eps)
                            assert numerical==-u[j];checks+=1
                # At c=q both pairs have exactly equal fields and Adam moments.
                for b in [0,1]:
                    g=[-v for v in velocity(q,q,b,kind)]
                    gcache=[-v for v in velocity(q,list(q),b,kind)]
                    assert g==gcache
                    m=[F(2,7),F(-3,5)];v=[F(5,9),F(7,11)]
                    def moments(gg):return ([F(9,10)*m[i]+F(1,10)*gg[i] for i in range(2)],
                       [F(999,1000)*v[i]+F(1,1000)*gg[i]**2 for i in range(2)])
                    assert moments(g)==moments(gcache);matched+=1
                if kind=='absolute':
                    tq,Aq=target(q);tc,Ac=target(c)
                    sq=[sign(tq[i]-q[i]) for i in range(2)];sc=[sign(tc[i]-q[i]) for i in range(2)]
                    diff=[velocity(q,q,1,kind)[j]-velocity(q,c,1,kind)[j] for j in range(2)]
                    dec=[F(1,4)*(sq[j]-sc[j]-sum(Aq[i][j]*(sq[i]-sc[i]) for i in range(2))+
                                   sum((Ac[i][j]-Aq[i][j])*sc[i] for i in range(2))) for j in range(2)]
                    assert diff==dec;decomposition+=1
    # Scalar exact witnesses, N=d=1. Connected velocity (1-A)*sign(r).
    q,c=F(1,2),F(1); tq,tc=2*q+1,2*c+1
    assert sign(tq-q)==sign(tc-q)==1
    assert (1-2)*sign(tq-q)==(1-2)*sign(tc-q)==-1
    assert (1-2)*(tq-q)!=(1-2)*(tc-q) # squared differs within same region/sign
    q=F(-1,2);tq=F(1);tc=F(3)
    assert sign(tq-q)==sign(tc-q)==1 and (1-0)*sign(tq-q)!=(1-2)*sign(tc-q)
    q,c=F(3,2),F(0);tq=q/2+1;tc=c/2+1
    assert sign(tq-q)==1 and sign(tc-q)==-1
    assert (1-F(1,2))*sign(tq-q)!=(1-F(1,2))*sign(tc-q)
    report={'script_sha256':sha(Path(__file__)),'source_manifest':man,
        'parent_protocol_sha256':sha(ROOT/'research/notes/2026-10-29-cycle14/NEXT_GRAPH_PROTOCOL.md'),
        'exact_local_derivative_checks':checks,'matched_gradient_and_moment_checks':matched,
        'connected_absolute_decomposition_checks':decomposition,
        'scalar_witnesses':['same ReLU region and signs: absolute equal, squared unequal',
           'activation-only mismatch changes connected absolute field',
           'residual-sign-only mismatch changes connected absolute field'],
        'labels_used':False,'neural_trajectories':0,'downloaded_code_executed':False,
        'cpu_seconds':time.process_time()-start,'status':'all checks passed'}
    assert report['cpu_seconds']<10
    (HERE/'graph_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
