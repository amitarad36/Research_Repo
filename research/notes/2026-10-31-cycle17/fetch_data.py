"""Pin the notebook-linked data; save opaque bytes without parsing targets."""
from pathlib import Path
import urllib.request,json,hashlib,datetime
HERE=Path(__file__).resolve().parent
def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'research-proposal-audit'})
 with urllib.request.urlopen(req,timeout=30) as r:return r.read()
url='https://api.github.com/repos/cvlab-epfl/zigzag/commits?path=exps/data/bostonHousing.txt&per_page=1'
assert not (HERE/'data_manifest.json').exists()
raw=fetch(url);(HERE/'data_commit.json').write_bytes(raw)
meta=json.loads(raw);rev=meta[0]['sha'];assert len(rev)==40
dataurl=f'https://raw.githubusercontent.com/cvlab-epfl/zigzag/{rev}/exps/data/bostonHousing.txt'
data=fetch(dataurl);(HERE/'bostonHousing.txt').write_bytes(data)
manifest={'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'commit_query_url':url,'commit_query_sha256':hashlib.sha256(raw).hexdigest(),
 'data_revision':rev,'url':dataurl,'file':'bostonHousing.txt','sha256':hashlib.sha256(data).hexdigest(),
 'bytes':len(data),'nonempty_lines':sum(bool(l.strip()) for l in data.splitlines()),
 'targets_parsed':False,'training_executed':False}
(HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
