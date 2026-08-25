from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


@dataclass
class MarketBundle:
    daily: Dict[str, pd.DataFrame]
    intraday: Dict[str, pd.DataFrame]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns={c: c.title() for c in out.columns})
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out = out[needed].dropna(subset=["Close"])
    return out


def history(ticker: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance no está instalado. Ejecuta: pip install -r requirements.txt")
    obj = yf.Ticker(ticker)
    df = obj.history(period=period, interval=interval, auto_adjust=False, prepost=prepost, actions=False)
    return _clean(df)


def _safe_history(ticker: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
    try:
        return history(ticker, period, interval, prepost=prepost)
    except Exception:
        return pd.DataFrame()


def load_daily(tickers: list[str], period: str = "5y") -> Dict[str, pd.DataFrame]:
    return {t: _safe_history(t, period, "1d") for t in tickers}


def load_intraday(tickers: list[str], period: str = "60d", interval: str = "5m") -> Dict[str, pd.DataFrame]:
    return {t: _safe_history(t, period, interval, prepost=False) for t in tickers}
