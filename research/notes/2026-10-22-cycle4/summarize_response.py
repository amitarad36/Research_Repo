"""Summarize all prespecified runs without changing protocol or selecting seeds."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
r=json.loads((root/'finite_response_results.json').read_text())
primary=[s for s in r['seeds'] if s['primary']]
out={'gates':r['gates'],'wall_seconds':r['wall_seconds'],'primary_seed_table':[],
     'all_run_count':sum(len(s['runs']) for s in r['seeds'])}
for s in r['seeds']:
    row={'seed':s['seed'],'primary':s['primary'],'no_adaptation_nmse':s['no_adaptation_nmse'],'runs':[]}
    for v in s['runs']:
        row['runs'].append({'rule':v['rule'],'tau':v['tau'],'modal_gain':v['final']['gain'][0],
          'random_gain':v['final']['gain'][1],'nmse':v['final']['nmse'],
          'nmse_ratio':v['final']['nmse']/s['no_adaptation_nmse'],
          'discrepancy_ratio':v['final']['discrepancy']/v['history'][0]['discrepancy'],
          'max_tangent_error_small_epsilon':v['max_tangent_error_by_epsilon'][1],
          'relative_step_difference_gain':v['final_gain_relative_step_difference'][0],
          'relative_step_difference_nmse':v['final_nmse_relative_step_difference'],
          'material_risk_deterioration':v['material_risk_deterioration']})
    out['primary_seed_table'].append(row)
allruns=[v for s in r['seeds'] for v in s['runs']]
out['max_tangent_error_small_epsilon']=max(v['max_tangent_error_by_epsilon'][1] for v in allruns)
out['max_tangent_error_large_epsilon']=max(v['max_tangent_error_by_epsilon'][0] for v in allruns)
out['max_step_difference_modal_gain']=max(v['final_gain_relative_step_difference'][0] for v in allruns)
out['max_step_difference_nmse']=max(v['final_nmse_relative_step_difference'] for v in allruns)
out['max_initial_field_derivative_error']=max(v for s in r['seeds'] for v in s['initial_derivative_checks'].values())
out['max_trajectory_field_derivative_error']=max(t['max_abs'] for v in allruns for result in [v,v['halved_step']] for t in result['directional_checks'])
(root/'SUMMARY.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='primary_seed_table'},indent=2))
for s in out['primary_seed_table']:
    print('SEED',s['seed'],'PRIMARY',s['primary'],'NMSE0',s['no_adaptation_nmse'])
    for v in s['runs']:
        print(v['rule'],v['tau'],'gain %.6g'%v['modal_gain'],'random %.6g'%v['random_gain'],
              'NMSE %.6g'%v['nmse'],'D/D0 %.6g'%v['discrepancy_ratio'])
