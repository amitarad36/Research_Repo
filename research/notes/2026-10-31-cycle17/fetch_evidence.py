from pathlib import Path
import urllib.request,json,hashlib,datetime
HERE=Path(__file__).resolve().parent
REV='100524001d890efc08911f3cba8f45f51f4aa7d1'
items={'it3_tree.json':f'https://api.github.com/repos/nikitadurasov/ittt/git/trees/{REV}?recursive=1',
       'it3_README.md':f'https://raw.githubusercontent.com/nikitadurasov/ittt/{REV}/README.md'}
report={}
for name,url in items.items():
 p=HERE/name
 assert not p.exists(),name
 req=urllib.request.Request(url,headers={'User-Agent':'research-proposal-audit','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(req,timeout=30) as r:
  raw=r.read();final=r.url
 p.write_bytes(raw)
 report[name]={'url':url,'resolved_url':final,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
(HERE/'fetch_manifest.json').write_text(json.dumps({'revision':REV,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'items':report},indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
