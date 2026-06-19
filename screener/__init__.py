"""Stock screener package.

- M1: universe loader.
- M2: data layer — a swappable ``DataProvider`` (``YFinanceProvider``) behind a
  local, date-keyed ``Cache``.
- M3: indicator engine — pure price-derived technicals (momentum, RSI, MACD,
  moving-average structure, relative volume, 52-wk-high distance) and a
  per-ticker ``snapshot``.

Later milestones add the profile/ranking engine (M4).
"""

from .cache import Cache
from .indicators import (
    EmaCrossState,
    distance_from_high,
    ema,
    ema_cross,
    is_stacked,
    latest,
    latest_ema_cross,
    macd,
    macd_latest,
    momentum,
    pct_from_ma,
    price_above_ma,
    relative_volume,
    relative_volume_latest,
    rsi,
    rsi_latest,
    sma,
    snapshot,
    trailing_return,
)
from .provider import DataProvider, Fundamentals, YFinanceProvider
from .universe import load_universe, tickers

__all__ = [
    "load_universe",
    "tickers",
    "DataProvider",
    "YFinanceProvider",
    "Fundamentals",
    "Cache",
    "sma",
    "ema",
    "rsi",
    "macd",
    "trailing_return",
    "relative_volume",
    "latest",
    "momentum",
    "relative_volume_latest",
    "rsi_latest",
    "macd_latest",
    "distance_from_high",
    "price_above_ma",
    "pct_from_ma",
    "is_stacked",
    "ema_cross",
    "latest_ema_cross",
    "EmaCrossState",
    "snapshot",
]
