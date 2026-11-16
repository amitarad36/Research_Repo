"""Cycle 11: exact scalar algebra checks; one eta, 12 steps, no neural data."""
from fractions import Fraction as Q
from decimal import Decimal as D, getcontext
from pathlib import Path
import hashlib,json,time,sys
getcontext().prec=60
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dec(q):return D(q.numerator)/D(q.denominator)
def sign(v):return Q(int(v>0)-int(v<0))
def main():
    begin=time.process_time()
    assert sha(ROOT/'PROTOCOL.md')=='0bc6760d30e7787459e794847cc8ace0e6b5fbeb129ae061838169b40f29ec9a'
    h=Q(1);a=Q(1,4);b=1-a;c=a*h;tau=Q(1);eta=Q(1,10);alpha=tau/(tau+eta)
    def graph(theta,phi,rule):
        q=h+theta;z=h if rule=='cached' else q
        target=h+phi+a*z;r=target-q
        return (b if rule=='input' else Q(1))*sign(r),r,target
    records=[];checks=0
    for rule in ['stop','input','cached']:
        for moving in [False,True]:
            theta=phi=Q(0);rows=[];teacher_sum=Q(0)
            for n in range(13):
                f,r,target=graph(theta,phi,rule);e=theta-phi
                rr=c-e if rule=='cached' else phi-b*theta+c
                ff=(b if rule=='input' else Q(1))*sign(rr)
                assert (f,r)==(ff,rr)
                if r:
                    eps=Q(1,10**6)
                    def loss(p):
                        z=h if rule=='cached' else (h+p if rule=='input' else h+theta)
                        targ=h+phi+a*z
                        return abs(targ-(h+p))
                    assert -(loss(theta+eps)-loss(theta-eps))/(2*eps)==f
                rows.append({'n':n,'time':str(n*eta),'theta':str(theta),'phi':str(phi),'lag':str(e),'residual':str(r),'field':str(f)})
                checks+=1
                if n==12:break
                new_theta=theta+eta*f
                new_phi=alpha*phi+(1-alpha)*new_theta if moving else phi
                if moving:
                    assert new_theta-new_phi==alpha*(e+eta*f)
                    teacher_sum+=eta*(new_theta-new_phi)/tau
                    assert new_phi==teacher_sum
                theta,phi=new_theta,new_phi
            records.append({'rule':rule,'schedule':'ema' if moving else 'fixed','rows':rows})
    fixed=next(r['rows'] for r in records if r['rule']=='cached' and r['schedule']=='fixed')
    assert [Q(row['theta']) for row in fixed[2:]]==[Q(1,5) if k%2==0 else Q(3,10) for k in range(11)]
    for initial in [c,c-eta,c+eta]:
        next_theta=initial+eta*sign(c-initial)
        assert next_theta==c and next_theta+eta*sign(c-next_theta)==c
    # Exact invariant interval: verify each affine branch at its endpoints/one-sided limits.
    L=alpha*(c-eta);U=alpha*(c+eta)
    candidates=[alpha*(L+eta),alpha*(c+eta),alpha*c,alpha*(c-eta),alpha*(U-eta)]
    assert L<c<U<tau and all(L<=x<=U for x in candidates)
    C=1+c/tau;assert C>Q(1,2) and c-L==alpha*eta*C and U-c==alpha*eta*(1-c/tau)
    ema=next(r['rows'] for r in records if r['rule']=='cached' and r['schedule']=='ema')
    first=next(row['n'] for row in ema if Q(row['lag'])>=c)
    for row in ema[:first+1]:assert Q(row['lag'])==tau*(1-alpha**row['n'])
    assert all(L<=Q(row['lag'])<=U for row in ema[first:])
    tstar=-dec(tau)*(1-dec(c/tau)).ln();bounds=[];max_ratios=[D(0),D(0),D(0)];max_inclusion_error=D(0)
    for row in ema:
        t=dec(Q(row['time']));ee=min(dec(tau)*(1-(-t/dec(tau)).exp()),dec(c))
        th=t if t<=tstar else tstar+dec(c/tau)*(t-tstar)
        ph=th-ee
        chosen=D(1) if t<tstar else dec(c/tau)
        eprime=(-t/dec(tau)).exp() if t<tstar else D(0)
        max_inclusion_error=max(max_inclusion_error,abs(eprime-(chosen-ee/dec(tau))))
        if t>=tstar:assert chosen!=0 and abs(chosen)<=1 and ee==dec(c)
        errs=[abs(dec(Q(row['lag']))-ee),abs(dec(Q(row['phi']))-ph),abs(dec(Q(row['theta']))-th)]
        bd=[dec(C*eta),dec(eta)*(dec(C)*t/dec(tau)+dec(c/tau)),dec(eta)*(dec(C)*(1+t/dec(tau))+dec(c/tau))]
        assert all(err<=bound for err,bound in zip(errs,bd))
        max_ratios=[max(v,err/bound) for v,err,bound in zip(max_ratios,errs,bd)]
        bounds.append({'n':row['n'],'errors':[str(e) for e in errs],'bounds':[str(v) for v in bd]})
    assert max_inclusion_error<D('1e-55')
    # Nonexistence and uniqueness rely on written all-state inequalities, not sampling.
    assert 1-c/tau>0
    output={'status':'passed','protocol_sha256':sha(ROOT/'PROTOCOL.md'),'script_sha256':sha(Path(__file__)),
            'python':sys.version,'parameters':{k:str(v) for k,v in {'h':h,'a':a,'b':b,'c':c,'tau':tau,'eta':eta,'alpha':alpha}.items()},
            'field_records_checked':checks,'scalar_updates':72,'no_neural_data_loaded':True,'no_step_sweep':True,
            'records':records,'cached_fixed_cycle':['1/5','3/10'],'cached_ema_first_crossing_step':first,
            'invariant_interval':[str(L),str(U)],'tstar':str(tstar),'bound_checks':bounds,
            'max_error_to_bound_ratios':[str(v) for v in max_ratios],'max_inclusion_equation_error':str(max_inclusion_error),
            'analytic_proof_required':'Nonexistence, uniqueness and bounds for all steps are proved in Nonsmooth_Baseline_Result.md; finite checks are illustrations only.',
            'cpu_seconds':time.process_time()-begin}
    assert output['cpu_seconds']<30
    (ROOT/'nonsmooth_verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ['records','bound_checks']},indent=2))
if __name__=='__main__':main()
