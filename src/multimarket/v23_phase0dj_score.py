from __future__ import annotations

import argparse
import csv
import json
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SYMBOLS=("BTCUSDT","ETHUSDT")
STREAMS=("markPriceKlines","indexPriceKlines","premiumIndexKlines")
F0_NAMES=("ret1","ret3","qfi1","cfi1","qfi3","qfi5","qfi10","cfi3","cfi5","cfi10","log_qty1","log_qty5","log_count1","log_count5","vwap_pressure_bps","buy_present","sell_present")
HORIZONS=(5,10,30)
ALPHAS=(0.1,1.0,10.0,100.0)
QUANTILES=(0.990,0.995,0.9975,0.999)
COSTS=(5.0,8.0,10.0,12.0,15.0)
FOLDS=((date(2026,6,15),date(2026,6,24)),(date(2026,6,25),date(2026,7,4)),(date(2026,7,5),date(2026,7,14)),(date(2026,7,15),date(2026,7,24)),(date(2026,7,25),date(2026,8,3)))
DEV_START=date(2026,5,26); DEV_END=date(2026,8,3)
WARMUP=60

@dataclass(frozen=True)
class Config:
    block:str
    horizon:int
    alpha:float
    quantile:float

def _utc(d:date)->int:
    return int(datetime(d.year,d.month,d.day,tzinfo=timezone.utc).timestamp())

def _read_zip_rows(path:Path)->list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        members=[m for m in zf.namelist() if not m.endswith('/')]
        if len(members)!=1: raise ValueError(f"expected one member in {path}")
        with zf.open(members[0]) as fh:
            rows=[r for r in csv.reader(line.decode('utf-8').strip() for line in fh) if r]
    if rows and not rows[0][0].lstrip('-').isdigit(): rows=rows[1:]
    return rows

def _load_state(raw:Path,symbol:str)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    times=[]; data={s:[] for s in STREAMS}
    d=DEV_START
    while d<=DEV_END:
        day_times=None
        for s in STREAMS:
            p=raw/s/symbol/'1m'/f"{symbol}-1m-{d.isoformat()}.zip"
            rows=_read_zip_rows(p)
            ot=np.asarray([int(r[0])//1000 for r in rows],dtype=np.int64)
            close=np.asarray([float(r[4]) for r in rows],dtype=np.float64)
            if day_times is None: day_times=ot
            elif not np.array_equal(day_times,ot): raise ValueError(f"state timestamp mismatch {d} {symbol} {s}")
            data[s].append(close)
        times.append(day_times)
        d+=timedelta(days=1)
    t=np.concatenate(times)
    return t,np.concatenate(data['markPriceKlines']),np.concatenate(data['indexPriceKlines']),np.concatenate(data['premiumIndexKlines'])

def _align_state_decisions(ts:np.ndarray,state_open:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Align completed state minutes to the frozen DEV second grid.

    The Phase 0D-H DEV CSV is label10-complete, so a few acquisition-boundary
    seconds can be absent even though its interior grid is contiguous.  We may
    trim state minutes whose :59 decision second lies strictly before the first
    or after the last DEV timestamp.  Any missing :59 decision second inside
    the DEV timestamp range is an integrity failure and must never be filled.
    """
    if len(ts)==0: raise ValueError('empty trade grid')
    if np.any(np.diff(ts)!=1): raise ValueError('trade grid must be contiguous inside DEV range')
    desired=state_open+59
    keep=(desired>=ts[0]) & (desired<=ts[-1])
    aligned=desired[keep]
    if len(aligned)==0: raise ValueError('no state minute overlaps trade grid')
    idx=np.searchsorted(ts,aligned,side='left')
    if np.any(idx>=len(ts)) or not np.array_equal(ts[idx],aligned):
        good=(idx<len(ts))
        observed=np.full(len(aligned),-1,dtype=np.int64)
        observed[good]=ts[idx[good]]
        bad=np.flatnonzero(observed!=aligned)
        first=int(aligned[bad[0]]) if len(bad) else None
        raise ValueError(f'trade grid missing interior minute decision second: {first}')
    return keep,idx,aligned

def _load_trade_minute(dev:Path,state_open:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    with dev.open('r',encoding='utf-8',newline='') as fh: header=next(csv.reader(fh))
    pos={n:i for i,n in enumerate(header)}
    req=('timestamp','price',*F0_NAMES)
    use=tuple(pos[n] for n in req)
    m=np.loadtxt(dev,delimiter=',',skiprows=1,usecols=use,dtype=np.float64,ndmin=2)
    ts=m[:,0].astype(np.int64,copy=False)
    keep,idx,decision=_align_state_decisions(ts,state_open)
    return keep,decision,m[idx,1],m[idx,2:]

def _lag_ret(x:np.ndarray,k:int)->np.ndarray:
    out=np.full(len(x),np.nan); out[k:]=np.log(x[k:]/x[:-k])*10000.0; return out

def _lag_diff(x:np.ndarray,k:int)->np.ndarray:
    out=np.full(len(x),np.nan); out[k:]=x[k:]-x[:-k]; return out

def _prior_z(x:np.ndarray,w:int=60)->np.ndarray:
    out=np.full(len(x),np.nan)
    cs=np.concatenate(([0.0],np.cumsum(x))); cs2=np.concatenate(([0.0],np.cumsum(x*x)))
    for i in range(w,len(x)):
        sm=cs[i]-cs[i-w]; sm2=cs2[i]-cs2[i-w]; mu=sm/w
        var=max(sm2/w-mu*mu,0.0); sd=var**0.5
        out[i]=0.0 if sd==0 else (x[i]-mu)/sd
    return out

def _build_blocks(mark:np.ndarray,index:np.ndarray,premium:np.ndarray,f0:np.ndarray)->dict[str,np.ndarray]:
    basis=(mark/index-1.0)*10000.0
    mr={k:_lag_ret(mark,k) for k in (1,5,15)}; ir={k:_lag_ret(index,k) for k in (1,5,15)}
    bd={k:_lag_diff(basis,k) for k in (1,5,15)}; pd={k:_lag_diff(premium,k) for k in (1,5,15)}
    bz=_prior_z(basis); pz=_prior_z(premium)
    f1=np.column_stack([basis,premium,mr[1],mr[5],mr[15],ir[1],ir[5],ir[15],bd[1],bd[5],bd[15],pd[1],pd[5],pd[15],bz,pz,mr[1]-ir[1],mr[5]-ir[5],mr[15]-ir[15]])
    fmap={n:i for i,n in enumerate(F0_NAMES)}
    inter=np.column_stack([f0[:,fmap['qfi10']]*bz,f0[:,fmap['cfi10']]*bz,f0[:,fmap['qfi10']]*pz,f0[:,fmap['cfi10']]*pz,f0[:,fmap['vwap_pressure_bps']]*bz])
    f2=np.column_stack([f0,f1,inter])
    return {'J0':f0,'J1':f1,'J2':f2}

def _gross(price:np.ndarray,h:int)->np.ndarray:
    out=np.full(len(price),np.nan); out[:-h]=np.log(price[h:]/price[:-h])*10000.0; return out

def _greedy(ix:np.ndarray,h:int)->np.ndarray:
    sel=[]; nxt=-1
    for i in ix.tolist():
        if i>=nxt: sel.append(i); nxt=i+h+1
    return np.asarray(sel,dtype=np.int64)

def _maxdd(p:np.ndarray)->float:
    if len(p)==0:return 0.0
    eq=np.cumsum(p); peaks=np.maximum.accumulate(np.concatenate(([0.0],eq)))[:-1]
    return float(np.max(peaks-eq))

def _metrics(decision:np.ndarray,gross:np.ndarray,pred:np.ndarray,gate:float,h:int,start:int,end:int,start_ts:int,end_ts:int,cost:float,arrays=False)->dict[str,object]:
    loc=_greedy(np.flatnonzero(np.abs(pred)>=gate),h); sel=loc+start
    direction=np.sign(pred[loc]) if len(loc) else np.empty(0); keep=direction!=0; sel=sel[keep]; direction=direction[keep]
    gt=direction*gross[sel] if len(sel) else np.empty(0); net=gt-cost
    wins=net[net>0]; losses=net[net<0]; gp=float(wins.sum()) if len(wins) else 0.0; gl=float(-losses.sum()) if len(losses) else 0.0
    pf=gp/gl if gl>0 else (float('inf') if gp>0 else 0.0); dd=_maxdd(net); total=float(net.sum())
    day0=start_ts//86400; day1=(end_ts-1)//86400; nd=day1-day0+1; cnt=np.zeros(nd,dtype=int); dp=np.zeros(nd)
    if len(sel):
        off=decision[sel]//86400-day0; np.add.at(cnt,off,1); np.add.at(dp,off,net)
    active=cnt>0; r5=np.convolve(dp,np.ones(5),mode='valid') if len(dp)>=5 else np.asarray([dp.sum()])
    out={'trades':int(len(sel)),'gross_bps_trade':float(gt.mean()) if len(gt) else 0.0,'net_bps_trade':float(net.mean()) if len(net) else 0.0,'total_net_bps':total,'profit_factor':float(pf),'max_drawdown_bps':dd,'pnl_to_drawdown':float(total/dd) if dd>0 else (float('inf') if total>0 else 0.0),'median_trades_day_active':float(np.median(cnt[active])) if np.any(active) else 0.0,'positive_active_day_fraction':float(np.mean(dp[active]>0)) if np.any(active) else 0.0,'median_net_bps_day_all':float(np.median(dp)),'worst_5day_rolling_net_bps':float(np.min(r5))}
    if arrays: out.update(trade_net_bps=net.tolist(),daily_counts=cnt.tolist(),daily_pnl_bps=dp.tolist())
    return out

def _survive(a:dict,b:dict)->bool:
    return float(a['net_bps_trade'])>0 and float(a['total_net_bps'])>0 and float(a['profit_factor'])>1 and float(a['median_trades_day_active'])>=2 and float(b['net_bps_trade'])>0 and float(b['total_net_bps'])>0

def _better(a:dict,b:dict|None)->bool:
    if b is None:return True
    am=float(a['m12']['median_net_bps_day_all']); bm=float(b['m12']['median_net_bps_day_all']); scale=max(abs(am),abs(bm),1e-12)
    if abs(am-bm)/scale>0.01:return am>bm
    aw=float(a['m12']['worst_5day_rolling_net_bps']); bw=float(b['m12']['worst_5day_rolling_net_bps'])
    if aw!=bw:return aw>bw
    at=float(a['m12']['median_trades_day_active']); bt=float(b['m12']['median_trades_day_active'])
    if at!=bt:return at>bt
    ad=float(a['m12']['max_drawdown_bps']); bd=float(b['m12']['max_drawdown_bps'])
    if ad!=bd:return ad<bd
    ac:Config=a['cfg']; bc:Config=b['cfg']
    if ac.block!=bc.block:return ac.block=='J1'
    if ac.horizon!=bc.horizon:return ac.horizon<bc.horizon
    return ac.quantile>bc.quantile

def _fit_scaled(Xtr,ytr,Xv,alpha):
    s=StandardScaler().fit(Xtr); z=s.transform(Xtr); zv=s.transform(Xv); m=Ridge(alpha=alpha).fit(z,ytr); return m.predict(z),m.predict(zv)

def _select(decision:np.ndarray,blocks:dict[str,np.ndarray],gross_by_h:dict[int,np.ndarray],eval_start_ts:int,candidate:bool)->tuple[Config|None,dict]:
    search_blocks=('J1','J2') if candidate else ('J0',); best=None; tested=survivors=0
    for h in HORIZONS:
        outer_end=int(np.searchsorted(decision,eval_start_ts-h*60,side='left')); cut=int(outer_end*0.8); cut_ts=int(decision[cut]); train_end=int(np.searchsorted(decision,cut_ts-h*60,side='left'))
        if train_end<=WARMUP or outer_end<=cut: continue
        y=gross_by_h[h]
        for block in search_blocks:
            X=blocks[block]; Xtr=X[WARMUP:train_end]; ytr=y[WARMUP:train_end]; Xv=X[cut:outer_end]
            s=StandardScaler().fit(Xtr); z=s.transform(Xtr); zv=s.transform(Xv)
            for alpha in ALPHAS:
                m=Ridge(alpha=alpha).fit(z,ytr); tp=m.predict(z); vp=m.predict(zv); ab=np.abs(tp)
                for q in QUANTILES:
                    tested+=1; gate=float(np.quantile(ab,q)); start_ts=int(decision[cut]); end_ts=eval_start_ts
                    m12=_metrics(decision,y,vp,gate,h,cut,outer_end,start_ts,end_ts,12.0); m15=_metrics(decision,y,vp,gate,h,cut,outer_end,start_ts,end_ts,15.0)
                    if not _survive(m12,m15):continue
                    survivors+=1; c={'cfg':Config(block,h,alpha,q),'m12':m12,'m15':m15}
                    if _better(c,best):best=c
    if best is None:return None,{'tested':tested,'survivors':0,'reason':'NO_CONFIGURATION'}
    c:Config=best['cfg']; return c,{'tested':tested,'survivors':survivors,'selected':c.__dict__,'selected_inner_12bps':best['m12'],'selected_inner_15bps':best['m15']}

def _outer(decision,blocks,gross_by_h,cfg:Config,sd:date,ed:date)->dict:
    st=_utc(sd); et=_utc(ed+timedelta(days=1)); tr_end=int(np.searchsorted(decision,st-cfg.horizon*60,side='left')); eb=int(np.searchsorted(decision,st)); ee=int(np.searchsorted(decision,et-cfg.horizon*60,side='left'))
    X=blocks[cfg.block]; y=gross_by_h[cfg.horizon]; s=StandardScaler().fit(X[WARMUP:tr_end]); z=s.transform(X[WARMUP:tr_end]); m=Ridge(alpha=cfg.alpha).fit(z,y[WARMUP:tr_end]); tp=m.predict(z); p=m.predict(s.transform(X[eb:ee])); gate=float(np.quantile(np.abs(tp),cfg.quantile))
    return {'config':cfg.__dict__,'absolute_prediction_gate':gate,'costs':{str(int(c)):_metrics(decision,y,p,gate,cfg.horizon,eb,ee,st,et,c,arrays=True) for c in COSTS}}

def _pool(folds:list[dict],key:str,cost='12')->dict:
    pn=[]; dc=[]; dp=[]; foldexp=[]; pos=0
    for f in folds:
        o=f.get(key)
        if not o: foldexp.append(float('-inf')); continue
        m=o['costs'][cost]; vals=[float(x) for x in m['trade_net_bps']]; pn.extend(vals); dc.extend(m['daily_counts']); dp.extend(m['daily_pnl_bps']); foldexp.append(float(m['net_bps_trade'])); pos+=int(float(m['total_net_bps'])>0)
    a=np.asarray(pn); dca=np.asarray(dc,dtype=int); dpa=np.asarray(dp); wins=a[a>0]; losses=a[a<0]; gp=float(wins.sum()) if len(wins) else 0.0; gl=float(-losses.sum()) if len(losses) else 0.0; pf=gp/gl if gl>0 else (float('inf') if gp>0 else 0.0); dd=_maxdd(a); total=float(a.sum()) if len(a) else 0.0; active=dca>0
    return {'trades':int(len(a)),'net_bps_trade':float(a.mean()) if len(a) else 0.0,'total_net_bps':total,'profit_factor':float(pf),'max_drawdown_bps':dd,'pnl_to_drawdown':float(total/dd) if dd>0 else (float('inf') if total>0 else 0.0),'positive_outer_folds':pos,'fold_expectancies':foldexp,'median_trades_day_active':float(np.median(dca[active])) if np.any(active) else 0.0,'positive_active_day_fraction':float(np.mean(dpa[active]>0)) if np.any(active) else 0.0}

def _gate(p12,p15)->bool:
    scored=sum(bool(np.isfinite(float(x))) for x in p12['fold_expectancies'])
    return bool(scored==5 and int(p12['positive_outer_folds'])>=4 and float(p12['net_bps_trade'])>=1 and float(p12['total_net_bps'])>0 and float(p15['net_bps_trade'])>0 and float(p15['total_net_bps'])>0 and float(p12['profit_factor'])>=1.15 and float(p12['positive_active_day_fraction'])>=0.55 and float(p12['pnl_to_drawdown'])>=2 and min(float(x) for x in p12['fold_expectancies'])>=-2 and float(p12['median_trades_day_active'])>=2)

def score_symbol(raw:Path,work:Path,symbol:str)->dict:
    state_open,mark,index,premium=_load_state(raw,symbol)
    state_keep,decision,trade_price,f0=_load_trade_minute(work/f'{symbol}_DEV.csv',state_open)
    mark=mark[state_keep]; index=index[state_keep]; premium=premium[state_keep]
    blocks=_build_blocks(mark,index,premium,f0)
    if not all(np.all(np.isfinite(v[WARMUP:])) for v in blocks.values()): raise ValueError('non-finite features after warmup')
    gross={h:_gross(trade_price,h) for h in HORIZONS}; folds=[]
    for i,(sd,ed) in enumerate(FOLDS,1):
        st=_utc(sd); r_cfg,r_in=_select(decision,blocks,gross,st,False); c_cfg,c_in=_select(decision,blocks,gross,st,True); rec={'fold':i,'eval_start':sd.isoformat(),'eval_end':ed.isoformat(),'reference_inner':r_in,'candidate_inner':c_in}
        if r_cfg: rec['reference_outer']=_outer(decision,blocks,gross,r_cfg,sd,ed)
        if c_cfg: rec['candidate_outer']=_outer(decision,blocks,gross,c_cfg,sd,ed)
        folds.append(rec)
    r12=_pool(folds,'reference_outer','12'); c12=_pool(folds,'candidate_outer','12'); c15=_pool(folds,'candidate_outer','15'); structural=bool(_gate(c12,c15)); incremental=bool(float(c12['net_bps_trade'])>float(r12['net_bps_trade']) and float(c12['total_net_bps'])>float(r12['total_net_bps'])); passed=bool(structural and incremental)
    return {'phase':'V2.3-PHASE0DJ-FUTURES-STATE','symbol':symbol,'development_only':True,'historical_holdout_opened':False,'boundary_state_minutes_trimmed':int(len(state_open)-np.count_nonzero(state_keep)),'folds':folds,'pooled_reference_12bps':r12,'pooled_candidate_12bps':c12,'pooled_candidate_15bps':c15,'candidate_structural_gate':structural,'incremental_vs_J0':incremental,'development_pass':passed,'decision':'CANDIDATE_FREEZE_BEFORE_CONFIRMATION' if passed else 'FAIL_KEEP_HOLDOUT_SEALED'}

def main(argv=None)->int:
    p=argparse.ArgumentParser(); p.add_argument('--raw-dir',default='data/v23_phase0dj_state_raw'); p.add_argument('--work-dir',default='evidence/v23/phase0dh_tf'); p.add_argument('--output-dir',default='evidence/v23/phase0dj_score'); a=p.parse_args(argv); raw=Path(a.raw_dir); work=Path(a.work_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); results=[]
    for s in SYMBOLS:
        print(f'[{s}] Phase 0D-J nested futures-state scoring',flush=True); r=score_symbol(raw,work,s); results.append(r); (out/f'{s}_PHASE0DJ.json').write_text(json.dumps(r,indent=2)+'\n'); print(f"[{s}] pass={r['development_pass']} candidate_trades={r['pooled_candidate_12bps']['trades']} expectancy12={r['pooled_candidate_12bps']['net_bps_trade']:.6f} incremental={r['incremental_vs_J0']}",flush=True)
    cand=[r['symbol'] for r in results if r['development_pass']]; summary={'phase':'V2.3-PHASE0DJ-FUTURES-STATE','development_only':True,'historical_holdout_opened':False,'candidate_targets':cand,'decision':'CANDIDATE_FOUND_FREEZE_BEFORE_CONFIRMATION' if cand else 'FAIL_KEEP_HOLDOUT_SEALED'}; (out/'V23_PHASE0DJ_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n'); print('candidate_targets='+(','.join(cand) if cand else 'NONE')); print('decision='+summary['decision']); return 0

if __name__=='__main__': raise SystemExit(main())