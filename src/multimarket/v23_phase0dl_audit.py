from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import date, datetime, timezone
from pathlib import Path

DEV_DAYS = tuple(date(2026, m, 1) for m in range(1, 8))
SYMBOLS = ("BTCUSDT", "ETHUSDT")
DATA_TYPES = ("incremental_book_L2", "trades")
L2_HEADER = ("exchange","symbol","timestamp","local_timestamp","is_snapshot","side","price","amount")
TRADES_HEADER = ("exchange","symbol","timestamp","local_timestamp","id","side","price","amount")


def _day_bounds_us(d: date) -> tuple[int, int]:
    start = int(datetime(d.year,d.month,d.day,tzinfo=timezone.utc).timestamp()*1_000_000)
    return start, start + 86_400_000_000


def _audit_file(path: Path, data_type: str, symbol: str, day: date) -> dict[str, object]:
    expected = L2_HEADER if data_type == "incremental_book_L2" else TRADES_HEADER
    start_us, end_us = _day_bounds_us(day)
    rows = snapshots = bad = 0
    first_local = last_local = None
    first_exchange = last_exchange = None
    seen_snapshot = False
    pre_snapshot_rows = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            r = csv.reader(fh)
            header = tuple(next(r))
            if header != expected:
                return {"pass":False,"path":str(path),"reason":"HEADER_MISMATCH","header":header,"expected":expected}
            pos = {n:i for i,n in enumerate(header)}
            prev_local = -1
            for row in r:
                rows += 1
                if len(row) != len(header):
                    bad += 1; continue
                try:
                    ex=row[pos['exchange']]; sy=row[pos['symbol']]
                    ts=int(row[pos['timestamp']]); lt=int(row[pos['local_timestamp']])
                    side=row[pos['side']]; price=float(row[pos['price']]); amount=float(row[pos['amount']])
                except Exception:
                    bad += 1; continue
                if ex != "binance-futures" or sy != symbol or lt < start_us or lt >= end_us or lt < prev_local or price <= 0 or amount < 0:
                    bad += 1
                if data_type == "incremental_book_L2":
                    snap = row[pos['is_snapshot']].strip().lower()
                    if snap not in ("true","false"):
                        bad += 1
                    if snap == "true":
                        snapshots += 1; seen_snapshot = True
                    elif not seen_snapshot:
                        pre_snapshot_rows += 1
                    if side not in ("bid","ask"):
                        bad += 1
                else:
                    if side not in ("buy","sell","unknown"):
                        bad += 1
                prev_local = lt
                if first_local is None:
                    first_local=lt; first_exchange=ts
                last_local=lt; last_exchange=ts
        reason = None
        if rows == 0: reason="EMPTY_DATASET"
        elif bad: reason="ROW_INTEGRITY_FAILURE"
        elif data_type == "incremental_book_L2" and snapshots == 0: reason="NO_SNAPSHOT"
        passed = reason is None
        return {
            "pass":passed,"path":str(path),"rows":rows,"bad_rows":bad,
            "snapshots":snapshots if data_type=="incremental_book_L2" else None,
            "pre_snapshot_rows":pre_snapshot_rows if data_type=="incremental_book_L2" else None,
            "first_local_timestamp":first_local,"last_local_timestamp":last_local,
            "first_exchange_timestamp":first_exchange,"last_exchange_timestamp":last_exchange,
            "reason":reason,
        }
    except Exception as exc:
        return {"pass":False,"path":str(path),"reason":"EXCEPTION","error":repr(exc)}


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description="Audit frozen Phase 0D-L development L2 files")
    p.add_argument("--raw-dir",default="data/v23_phase0dl_l2_raw")
    p.add_argument("--output",default="evidence/v23/phase0dl_l2_audit.json")
    a=p.parse_args(argv); root=Path(a.raw_dir)
    recs=[]
    for d in DEV_DAYS:
        for s in SYMBOLS:
            for t in DATA_TYPES:
                path=root/t/s/f"{d.isoformat()}.csv.gz"
                rec=_audit_file(path,t,s,d) if path.exists() else {"pass":False,"path":str(path),"reason":"MISSING_FILE"}
                rec.update(day=d.isoformat(),symbol=s,data_type=t); recs.append(rec)
                print(f"{d} {s} {t} pass={rec['pass']} rows={rec.get('rows')} reason={rec.get('reason')}",flush=True)
    failures=[r for r in recs if not r['pass']]
    out={
        "phase":"V2.3-PHASE0DL-L2-MECHANISM",
        "development_only":True,
        "audited_days":[d.isoformat() for d in DEV_DAYS],
        "sealed_confirmation_day":"2026-08-01",
        "confirmation_analytically_opened":False,
        "files":recs,"failures":failures,"pass":not failures,
    }
    op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(out,indent=2)+"\n")
    print(f"failures={len(failures)}")
    print("PHASE0DL_DATA_AUDIT="+("PASS" if not failures else "FAIL"))
    return 0 if not failures else 2


if __name__=="__main__":
    raise SystemExit(main())
