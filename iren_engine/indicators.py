from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def add_daily_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    x = df.copy()
    c = x["Close"]
    x[prefix + "ret1"] = c.pct_change(1)
    x[prefix + "ret3"] = c.pct_change(3)
    x[prefix + "ret5"] = c.pct_change(5)
    x[prefix + "ret20"] = c.pct_change(20)
    for n in (8, 21, 50, 200):
        ema = c.ewm(span=n, adjust=False).mean()
        x[prefix + f"dist_ema{n}"] = c / ema - 1
    x[prefix + "rsi14"] = rsi(c, 14) / 100.0
    a = atr(x, 14)
    x[prefix + "atrp"] = a / c
    vol = x["Volume"].replace(0, np.nan)
    x[prefix + "vol_z"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    x[prefix + "range"] = (x["High"] - x["Low"]) / c
    return x


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["Close"]
    x["ret_5m"] = c.pct_change(1)
    x["ret_15m"] = c.pct_change(3)
    x["ret_60m"] = c.pct_change(12)
    x["ema9"] = c.ewm(span=9, adjust=False).mean()
    x["ema21"] = c.ewm(span=21, adjust=False).mean()
    x["dist_ema9"] = c / x["ema9"] - 1
    x["dist_ema21"] = c / x["ema21"] - 1
    x["rsi14"] = rsi(c, 14) / 100.0
    ny = x.copy()
    if ny.index.tz is None:
        ny.index = ny.index.tz_localize("UTC")
    ny_idx = ny.index.tz_convert("America/New_York")
    session = pd.Series(ny_idx.date, index=x.index)
    typical = (x["High"] + x["Low"] + x["Close"]) / 3.0
    pv = typical * x["Volume"]
    cum_pv = pv.groupby(session).cumsum()
    cum_v = x["Volume"].groupby(session).cumsum().replace(0, np.nan)
    x["vwap"] = cum_pv / cum_v
    x["dist_vwap"] = c / x["vwap"] - 1
    vol = x["Volume"].replace(0, np.nan)
    x["vol_z_20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    mins = ny_idx.hour * 60 + ny_idx.minute
    x["session_frac"] = np.clip((np.asarray(mins, dtype=float) - (9*60+30)) / 390.0, 0, 1)
    return x
