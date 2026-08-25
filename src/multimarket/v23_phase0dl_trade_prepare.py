from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
DEV_DAYS=tuple(date(2026,m,1) for m in range(1,8)); SYMBOLS=("BTCUSDT","ETHUSDT")
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
 return h.hexdigest()
def root(): return Path(__file__).resolve().parents[2]
def bounds(d):
 s=int(datetime(d.year,d.month,d.day,tzinfo=timezone.utc).timestamp()*1_000_000); return s,s+86_400_000_000
def build(bd):
 src=root()/"tools/v23_phase0dl_trade250.cpp"; bd.mkdir(parents=True,exist_ok=True); exe=bd/"v23_phase0dl_trade250"; cxx=shutil.which('g++')
 if not cxx: raise RuntimeError('g++ not found')
 p=subprocess.run([cxx,'-std=c++17','-O3','-DNDEBUG',str(src),'-lz','-o',str(exe)],capture_output=True,text=True)
 if p.returncode: raise RuntimeError(p.stderr)
 return exe
def one(exe,raw,out,d,s):
 src=raw/'trades'/s/f'{d.isoformat()}.csv.gz'; dst=out/s/f'{d.isoformat()}_TRADE250.csv'; dst.parent.mkdir(parents=True,exist_ok=True); a,b=bounds(d)
 p=subprocess.run([str(exe),str(src),str(dst),str(a),str(b)],capture_output=True,text=True); ok=p.returncode==0
 return {'day':d.isoformat(),'symbol':s,'pass':ok,'stderr':p.stderr.strip(),'output':str(dst),'sha256':sha(dst) if ok else None}
def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',default='data/v23_phase0dl_l2_raw'); ap.add_argument('--output-dir',default='evidence/v23/phase0dl_trade250'); ap.add_argument('--build-dir',default='.build/phase0dl'); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args(argv)
 raw,out=Path(a.raw_dir),Path(a.output_dir); exe=build(Path(a.build_dir)); jobs=[(d,s) for d in DEV_DAYS for s in SYMBOLS]; rec=[]
 with ThreadPoolExecutor(max_workers=max(1,min(a.workers,8))) as pool:
  fs=[pool.submit(one,exe,raw,out,d,s) for d,s in jobs]
  for f in as_completed(fs): r=f.result(); rec.append(r); print(r['day'],r['symbol'],'pass='+str(r['pass']),r['stderr'],flush=True)
 rec.sort(key=lambda x:(x['day'],x['symbol'])); fail=[x for x in rec if not x['pass']]; m={'stage':'TRADE250_PREPARATION','development_only':True,'confirmation_analytically_opened':False,'files':rec,'failures':fail,'pass':not fail}; out.mkdir(parents=True,exist_ok=True); (out/'TRADE250_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
 print(f'expected_jobs={len(jobs)} completed={len(rec)-len(fail)} failures={len(fail)}'); print('PHASE0DL_TRADE250=' + ('PASS' if m['pass'] else 'FAIL')); return 0 if m['pass'] else 2
if __name__=='__main__': raise SystemExit(main())
