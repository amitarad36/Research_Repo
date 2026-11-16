"""Read-only primary-source retrieval and text extraction; never execute notebook code."""
import hashlib
import json
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REV='100524001d890efc08911f3cba8f45f51f4aa7d1'
URL=f'https://raw.githubusercontent.com/nikitadurasov/ittt/{REV}/exps/tabular/vanilla_model.ipynb'
request=urllib.request.Request(URL,headers={'User-Agent':'research-source-audit'})
raw=urllib.request.urlopen(request,timeout=30).read()
notebook=json.loads(raw)
(ROOT/'tabular_vanilla_model.ipynb').write_bytes(raw)
texts=[]
for index,cell in enumerate(notebook['cells']):
    if cell['cell_type']=='code':
        lines=''.join(cell['source']).splitlines()
        texts.append(f'ZERO-BASED CELL {index}\n'+'\n'.join(f'{j+1:4d}: {line}' for j,line in enumerate(lines)))
extracted='\n\n'.join(texts)+'\n'
(ROOT/'tabular_cells.txt').write_text(extracted,encoding='utf-8')
manifest={'retrieved_date':'2026-10-26','revision':REV,'url':URL,
          'browser_url':f'https://github.com/nikitadurasov/ittt/blob/{REV}/exps/tabular/vanilla_model.ipynb',
          'file':'tabular_vanilla_model.ipynb','sha256':hashlib.sha256(raw).hexdigest(),
          'bytes':len(raw),'text_file':'tabular_cells.txt',
          'text_sha256':hashlib.sha256((ROOT/'tabular_cells.txt').read_bytes()).hexdigest(),
          'executed_remote_code':False}
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
print('\n'.join(t for t in texts if t.startswith('ZERO-BASED CELL 13\n')))
