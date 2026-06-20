"""Stock screener package.

- M1: universe loader.
- M2: data layer — a swappable ``DataProvider`` (``YFinanceProvider``) behind a
  local, date-keyed ``Cache``.
- M3: indicator engine — pure price-derived technicals (momentum, RSI, MACD,
  moving-average structure, relative volume, 52-wk-high distance) and a
  per-ticker ``snapshot``.
- M4: profile + ranking engine — declarative :class:`Profile` configs (hard
  filters + weighted signals) run through one generic pipeline
  (:func:`run_screen`) that assembles features, scores by cross-sectional
  percentile rank, and returns a ranked, explained table.
- M-patterns: descriptive chart-pattern detection — pure swing-pivot zigzag +
  per-shape detectors (double tops, H&S, triangles, wedges, cup & handle) over
  1d/1w/1mo bars, returning picklable :class:`Pattern` shapes (describes, never
  advises).

Later milestones add the Streamlit dashboard (M5).
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
from .engine import (
    apply_filters,
    assemble_features,
    compute_sector_strength,
    run_screen,
    score_and_rank,
)
from .patterns import (
    PATTERN_LABELS,
    Pattern,
    Pivot,
    detect,
    detect_all_timeframes,
    find_pivots,
    human_label,
    resample_ohlc,
)
from .profiles import PROFILES, Filter, Profile, SignalSpec, get_profile
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
    # M4: profiles + ranking engine
    "Profile",
    "SignalSpec",
    "Filter",
    "PROFILES",
    "get_profile",
    "assemble_features",
    "compute_sector_strength",
    "apply_filters",
    "score_and_rank",
    "run_screen",
    # M-patterns: descriptive chart-pattern detection
    "Pattern",
    "Pivot",
    "resample_ohlc",
    "find_pivots",
    "detect",
    "detect_all_timeframes",
    "human_label",
    "PATTERN_LABELS",
]
