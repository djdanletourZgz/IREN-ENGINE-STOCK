from __future__ import annotations
import json, sqlite3
from pathlib import Path
from datetime import datetime, timezone

SCHEMA="""
CREATE TABLE IF NOT EXISTS snapshots (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 ts_utc TEXT NOT NULL,
 ticker TEXT NOT NULL,
 price REAL,
 payload_json TEXT NOT NULL
);
"""

def save_snapshot(path: str | Path, ticker: str, price: float | None, payload: dict):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(p)
    try:
        con.executescript(SCHEMA)
        con.execute("INSERT INTO snapshots(ts_utc,ticker,price,payload_json) VALUES(?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(),ticker,price,json.dumps(payload,default=str)))
        con.commit()
    finally:
        con.close()
