from pathlib import Path
import subprocess,json,hashlib,datetime
import numpy as np
HERE=Path(__file__).resolve().parent
PROPOSAL=Path(r'C:\Users\amita\source\repos\Detach Proposal Presentation')
roots=[PROPOSAL,Path(r'C:\Users\amita\source\repos\Project with Assaf'),
       Path(r'C:\Users\amita\.codex\.chatgpt-projects\g-p-6a9fa12b13bc8191acd58cbcf6c03aef')]
patterns=['*.pt','*.pth','*.ckpt','*.safetensors','*.npz','*.csv','*boston*','*Boston*','*Housing*','pyvenv.cfg']
rows=[]
for root in roots:
 cmd=['rg','--files','--hidden',str(root),'-g','!**/.git/**','-g','!**/node_modules/**']
 for pat in patterns:cmd+=['-g',pat]
 run=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8')
 assert run.returncode in [0,1],run.stderr
 paths=[Path(x) for x in run.stdout.splitlines()]
 rows.append({'root':str(root),'matched_files':[{'path':str(p),'bytes':p.stat().st_size} for p in paths],
              'errors':run.stderr,'patterns':patterns})
models=[]
for p in sorted((PROPOSAL/'research/notes/2026-10-22-cycle3').glob('seed_*.npz')):
 with np.load(p,allow_pickle=False) as a:
  models.append({'path':str(p),'parameter_shape':list(a['parameters'].shape),
                 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                 'compatibility':'114-parameter tanh surrogate; not 1291-parameter ReLU tabular model'})
result={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'inventory':rows,'verified_known_checkpoints':models,'checkpoint_files_deserialized':False,
 'labels_loaded':False,'scope_limit':'Only named roots searched; no machine-wide absence claim.',
 'runtime_discovery':[
 {'python':r'C:\Users\amita\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe','torch_spec':False,'jax_spec':False,'tensorflow_spec':False},
 {'python':r'C:\Users\amita\anaconda3\python.exe','torch_spec':True,'import_and_cpu_fixture':'passed; see conformance_results.json'},
 {'python':r'C:\Users\amita\anaconda3\envs\final-dl-env\python.exe','torch_spec':True,'import_tested':False},
 {'python':r'C:\Users\amita\anaconda3\envs\linearizer\python.exe','torch_spec':False}],
 'conda_registry_source':r'C:\Users\amita\.conda\environments.txt'}
assert not (HERE/'inventory_results.json').exists()
(HERE/'inventory_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'roots':[{'root':r['root'],'matches':len(r['matched_files']),
   'non_npz':[m['path'] for m in r['matched_files'] if not m['path'].endswith('.npz')]} for r in rows],
   'known_checkpoint_shapes':[m['parameter_shape'] for m in models]},indent=2))
