from dataclasses import dataclass

@dataclass(frozen=True)
class EngineConfig:
    ticker: str = "IREN"
    btc: str = "BTC-USD"
    qqq: str = "QQQ"
    nvda: str = "NVDA"
    vix: str = "^VIX"
    eurusd: str = "EURUSD=X"
    ai_basket: tuple[str, ...] = ("SOXX", "NVDA", "MU", "MRVL", "AVGO", "AMD", "VRT", "CRWV", "NBIS")
    daily_period: str = "5y"
    intraday_period: str = "60d"
    intraday_interval: str = "5m"
    daily_neighbors: int = 120
    intraday_neighbors: int = 160
    min_daily_samples: int = 50
    min_intraday_samples: int = 60
    v2_lookback_rows: int = 756
    v2_recency_half_life: int = 252
    options_max_dte: int = 45

CFG = EngineConfig()
