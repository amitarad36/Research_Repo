"""Post-result audit of the invariant parameter subspace; no new training."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'neural_gate_results.json').read_text())
active=np.r_[0:32,64:114]
inactive=np.arange(32,64)
report=[]
for s in r['seeds']:
    n=np.load(ROOT/f"seed_{s['seed']}.npz")
    out={'seed':s['seed'],'probes_at_label':s['geometry_at_label'],'rules':{}}
    for rule in ['input','stop']:
        ds=n[rule+'_dstudent'];dt=n[rule+'_dteacher']
        rows=float(max(abs(ds[inactive]).max(),abs(dt[inactive]).max()))
        entries=[]
        for tau in [1,20]:
            ss=ds[np.ix_(active,active)];tt=dt[np.ix_(active,active)]
            m=np.block([[ss,tt],[np.eye(82)/tau,-np.eye(82)/tau]])
            ev=np.linalg.eigvals(m)
            full=n[f'{rule}_tau{tau}_ode_eigen']
            entries.append({'tau':tau,'restricted_max_real':float(ev.real.max()),
                'max_real_difference_from_full':float(abs(ev.real.max()-full.real.max()))})
        assert rows==0
        out['rules'][rule]={'inactive_derivative_rows_max_abs':rows,
                           'active_parameters':82,'spectra':entries}
    report.append(out)
(ROOT/'invariant_subspace_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('All 32 inactive student rows are exactly zero.')
print('Maximum spectral-abscissa difference:',max(c['max_real_difference_from_full']
      for s in report for v in s['rules'].values() for c in v['spectra']))
