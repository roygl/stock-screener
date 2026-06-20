"""Support / resistance levels and a buy-zone band — pivot clustering over price.

Where :mod:`screener.patterns` turns the swing zigzag into *named shapes*, this
module turns the SAME swing pivots into horizontal **price levels**: clusters of
revisited swing highs/lows that act as support (below the last close) or
resistance (above it), plus a single descriptive **buy zone** band. It reuses the
pivot foundation wholesale — :func:`screener.patterns.find_pivots`,
:func:`screener.patterns.resample_ohlc` and the :class:`screener.patterns.Pivot`
dataclass — and adds nothing to the zigzag itself.

It is DESCRIPTIVE only. A :class:`Level` reports where price has clustered and how
firmly (touches, recency, tightness -> a ``[0, 1]`` strength); a :class:`BuyZone`
reports an entry band with its basis. Neither is a buy/sell/hold call or a price
target — the buy zone is an *educational* entry band (the UI carries the
not-advice disclaimer), never an exit/sell call and never execution.

Pipeline (all pure functions; no network, no streamlit, no yfinance):
  1. :func:`screener.patterns.resample_ohlc` — derive a 1d / 1w / 1mo OHLCV frame.
  2. :func:`screener.patterns.find_pivots`   — the swing-high/low zigzag (REUSED).
  3. :func:`support_resistance` — greedy single-linkage clustering of the pivot
     prices into volume-weighted levels, classified vs the last close.
  4. :func:`levels_all_timeframes` / :func:`buy_zone` — public entry points.

Design notes:
- **Greedy single-linkage on price.** Pivots are sorted by price and walked once;
  a cluster is extended while the next price is within
  :data:`CLUSTER_TOL_PCT` of the cluster's RUNNING MEAN, else a new cluster opens.
  Cheap, deterministic, and pure numpy (no scipy) — matching the patterns ethos.
- **Volume-weighted center.** A level's price is the volume-weighted mean of its
  touch prices, so a high-volume revisit pulls the line toward where real trade
  happened (bar volume falls back to ``1.0`` when missing/degenerate).
- **Strength in [0, 1].** The mean of three sub-scores: touch count (more revisits
  = firmer), recency (a more-recent last touch is more relevant), and tightness
  (a level whose touches hug one price is cleaner than a loose smear).
- **Picklability.** :class:`Level`, :class:`LevelSet` and :class:`BuyZone` are
  ``frozen`` dataclasses of only ``(str, float, int, bool, pd.Timestamp, tuple-of-
  frozen)`` fields, so they round-trip through ``st.cache_data`` exactly like
  :class:`screener.patterns.Pattern` / :class:`screener.indicators.EmaCrossState`.
- **Fail-soft.** Short / empty / degenerate / ``None`` frames yield an empty
  :class:`LevelSet` (``last_close`` ``NaN`` when unknowable) or ``None`` buy zone;
  the public entry points wrap their bodies so a bad frame never raises, mirroring
  :func:`screener.patterns.detect`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import patterns
from .indicators import ema

# --- clustering tolerance (the precision dial) ---------------------------
# Two pivot prices belong to the same level if they sit within this fraction of
# the cluster's running mean. Wider for the coarser, bigger-bar weekly/monthly
# frames (the same % wiggle spans more absolute price there), isolated at module
# top like patterns.py's TOL block so it is easy to tune.
CLUSTER_TOL_PCT: dict[str, float] = {"1d": 0.015, "1w": 0.025, "1mo": 0.04}

# Fallback buy-zone band half-widths around a rising 20-EMA (reusing the +1.5%
# pullback idea from engine.pullback_quality: a shallow hold just above the EMA).
_EMA_ZONE_BELOW = 0.01    # 1% under the EMA (the cushion you'd buy into)
_EMA_ZONE_ABOVE = 0.005   # 0.5% over the EMA (still "at" the line)
_EMA_ZONE_SPAN = 10       # bars of look-back to judge the EMA is rising
_EMA_ZONE_LEN = 20        # the 20-EMA per the plan

# Max levels surfaced per side (nearest-first); a band table beyond this is noise.
_MAX_PER_SIDE = 3


def _clamp01(c: float) -> float:
    """Clamp a score into ``[0, 1]`` (mirrors :func:`patterns._clamp01`)."""
    return max(0.0, min(1.0, float(c)))


# --- dataclasses ---------------------------------------------------------
@dataclass(frozen=True)
class Level:
    """One horizontal price level — a cluster of revisited swing pivots.

    ``price`` is the volume-weighted cluster center; ``kind`` is ``"support"``
    (center at/below the last close) or ``"resistance"`` (above it); ``touches``
    is how many pivots fell in the cluster; ``strength`` is a ``[0, 1]`` composite
    of touch count, recency and tightness; ``distance_pct`` is the signed
    ``(price - last_close) / last_close`` (negative below, positive above);
    ``first`` / ``last`` are the earliest / latest touch timestamps; ``timeframe``
    is the frame the level was read on. Frozen so it pickles for ``st.cache_data``.
    """

    price: float
    kind: str  # "support" | "resistance"
    touches: int
    strength: float
    distance_pct: float
    first: pd.Timestamp
    last: pd.Timestamp
    timeframe: str

    def to_dict(self) -> dict:
        """Plain-dict view (Timestamps left as Timestamp; the app formats them)."""
        return asdict(self)


@dataclass(frozen=True)
class LevelSet:
    """The support and resistance levels around one ticker's last close.

    ``supports`` are ordered NEAREST-BELOW first (price descending toward the last
    close); ``resistances`` NEAREST-ABOVE first (price ascending away from it);
    each side capped at three. ``last_close`` is the close the levels were
    classified against (``NaN`` when unknowable); ``timeframe`` is the frame they
    were read on. Frozen / picklable.
    """

    supports: tuple[Level, ...]
    resistances: tuple[Level, ...]
    last_close: float
    timeframe: str

    def to_dict(self) -> dict:
        """Plain-dict view (nested Levels become dicts via ``asdict``)."""
        return asdict(self)


@dataclass(frozen=True)
class BuyZone:
    """A descriptive entry band — NEVER advice, never an exit/sell call.

    ``low`` / ``high`` bound the band; ``basis`` names how it was derived
    (``"nearest support · 3 touches"`` or ``"20-EMA pullback"``); ``in_zone`` is
    whether the last close currently sits inside ``[low, high]``; ``distance_pct``
    is the signed gap from the last close to the NEAREST edge over the last close
    (``0.0`` when inside); ``timeframe`` is the frame it was read on. Frozen /
    picklable.
    """

    low: float
    high: float
    basis: str
    in_zone: bool
    distance_pct: float
    timeframe: str

    def to_dict(self) -> dict:
        return asdict(self)


# --- internal helpers ----------------------------------------------------
def _last_close(df_tf: pd.DataFrame) -> float:
    """Last close of a (resampled) frame as a ``float`` (``NaN`` if unavailable)."""
    if df_tf is None or "close" not in getattr(df_tf, "columns", []) or len(df_tf) == 0:
        return float("nan")
    try:
        return float(df_tf["close"].iloc[-1])
    except (IndexError, ValueError, TypeError):
        return float("nan")


def _bar_volume(df_tf: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Volume at the bar labelled ``ts`` (``1.0`` fallback if missing/degenerate).

    A non-finite or non-positive volume is treated as the neutral weight ``1.0`` so
    the volume-weighted center never divides by zero or gets pulled by a junk bar.
    """
    if "volume" not in getattr(df_tf, "columns", []):
        return 1.0
    try:
        v = float(df_tf["volume"].loc[ts])
    except (KeyError, IndexError, ValueError, TypeError):
        return 1.0
    if not np.isfinite(v) or v <= 0.0:
        return 1.0
    return v


def _cluster_prices(prices: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Greedy single-linkage on a SORTED price array -> list of ``(start, end)``.

    Walks the prices once (they must be ascending); the current cluster is extended
    while the next price is within ``tol`` of the cluster's RUNNING MEAN, else a new
    cluster opens at that price. Returns half-open ``[start, end)`` positional spans
    into ``prices``. Pure / allocation-light.
    """
    spans: list[tuple[int, int]] = []
    n = len(prices)
    if n == 0:
        return spans
    start = 0
    running_sum = float(prices[0])
    count = 1
    for i in range(1, n):
        mean = running_sum / count
        if mean != 0.0 and abs(float(prices[i]) - mean) / abs(mean) <= tol:
            running_sum += float(prices[i])
            count += 1
        else:
            spans.append((start, i))
            start = i
            running_sum = float(prices[i])
            count = 1
    spans.append((start, n))
    return spans


def _level_strength(
    touches: int,
    prices: np.ndarray,
    last_ts: pd.Timestamp,
    center: float,
    tol: float,
    span_first: pd.Timestamp,
    span_last: pd.Timestamp,
) -> float:
    """Composite ``[0, 1]`` strength: mean of touch / recency / tightness scores.

    * ``touch_score``  = ``min(touches / 4, 1)`` — four+ revisits read as firm.
    * ``recency_score``= how late this level's last touch sits within the SERIES
      span ``[span_first, span_last]`` (a level touched at the right edge scores 1,
      one only touched near the left edge scores ~0). Degenerate span -> ``1.0``.
    * ``tightness``    = ``1 - (price range) / (tol * center)`` — touches hugging
      one price score ~1; a level smeared across the whole tolerance band scores 0.
    """
    touch_score = min(touches / 4.0, 1.0)

    total = float((span_last - span_first) / np.timedelta64(1, "ns"))
    if total <= 0.0:
        recency_score = 1.0
    else:
        elapsed = float((last_ts - span_first) / np.timedelta64(1, "ns"))
        recency_score = _clamp01(elapsed / total)

    denom = tol * center
    if denom <= 0.0:
        tightness = 1.0
    else:
        price_range = float(np.max(prices) - np.min(prices))
        tightness = _clamp01(1.0 - price_range / denom)

    return _clamp01((touch_score + recency_score + tightness) / 3.0)


# --- support / resistance ------------------------------------------------
def _support_resistance(df: pd.DataFrame, timeframe: str) -> LevelSet:
    """Unguarded core of :func:`support_resistance` (the public fn wraps it)."""
    tol = CLUSTER_TOL_PCT.get(timeframe, CLUSTER_TOL_PCT["1d"])
    df_tf = patterns.resample_ohlc(df, timeframe)
    last_close = _last_close(df_tf)

    pivots = patterns.find_pivots(df_tf, timeframe=timeframe)
    if len(pivots) < 2:
        return LevelSet(supports=(), resistances=(), last_close=last_close, timeframe=timeframe)

    # Series span (for the recency sub-score): first..last bar of the frame.
    index = df_tf.index
    span_first = pd.Timestamp(index[0])
    span_last = pd.Timestamp(index[-1])

    # Collect (price, timestamp, volume) per pivot, then sort by price for linkage.
    triples = []
    for p in pivots:
        ts = pd.Timestamp(p.date)
        triples.append((float(p.price), ts, _bar_volume(df_tf, ts)))
    triples.sort(key=lambda t: t[0])

    prices = np.array([t[0] for t in triples], dtype="float64")
    timestamps = [t[1] for t in triples]
    volumes = np.array([t[2] for t in triples], dtype="float64")

    levels: list[Level] = []
    for lo, hi in _cluster_prices(prices, tol):
        touches = hi - lo
        if touches < 2:
            continue
        seg_prices = prices[lo:hi]
        seg_vol = volumes[lo:hi]
        seg_ts = timestamps[lo:hi]

        weight_total = float(np.sum(seg_vol))
        if weight_total <= 0.0:  # all-fallback weights can't be zero, but be safe
            center = float(np.mean(seg_prices))
        else:
            center = float(np.sum(seg_prices * seg_vol) / weight_total)

        first = min(seg_ts)
        last = max(seg_ts)
        strength = _level_strength(
            touches, seg_prices, last, center, tol, span_first, span_last
        )

        if np.isfinite(last_close) and last_close != 0.0:
            kind = "support" if center <= last_close else "resistance"
            distance_pct = (center - last_close) / last_close
        else:
            # No usable last close: classify by sign only, distance unknown.
            kind = "support"
            distance_pct = float("nan")

        levels.append(
            Level(
                price=center,
                kind=kind,
                touches=touches,
                strength=strength,
                distance_pct=distance_pct,
                first=first,
                last=last,
                timeframe=timeframe,
            )
        )

    supports = [lv for lv in levels if lv.kind == "support"]
    resistances = [lv for lv in levels if lv.kind == "resistance"]
    # Nearest-first: supports by price DESC (closest below the close first),
    # resistances by price ASC (closest above first); cap each side.
    supports.sort(key=lambda lv: lv.price, reverse=True)
    resistances.sort(key=lambda lv: lv.price)

    return LevelSet(
        supports=tuple(supports[:_MAX_PER_SIDE]),
        resistances=tuple(resistances[:_MAX_PER_SIDE]),
        last_close=last_close,
        timeframe=timeframe,
    )


def support_resistance(df: pd.DataFrame, *, timeframe: str = "1d") -> LevelSet:
    """Cluster swing pivots into support / resistance levels at one ``timeframe``.

    Pure; never raises (the whole body is wrapped in ``try/except`` -> an empty
    :class:`LevelSet`, mirroring :func:`screener.patterns.detect`). Resamples
    internally via :func:`screener.patterns.resample_ohlc`, reads the swing zigzag
    via :func:`screener.patterns.find_pivots`, and on ``< 2`` pivots returns an
    empty set carrying the last close (or ``NaN``). Levels keep only clusters with
    ``>= 2`` touches; each side is ordered nearest-first and capped at three. The
    result is picklable (frozen dataclasses).
    """
    try:
        return _support_resistance(df, timeframe)
    except Exception:
        last_close = float("nan")
        try:
            last_close = _last_close(patterns.resample_ohlc(df, timeframe))
        except Exception:
            pass
        return LevelSet(supports=(), resistances=(), last_close=last_close, timeframe=timeframe)


def levels_all_timeframes(daily_df: pd.DataFrame) -> dict[str, LevelSet]:
    """Support / resistance across weekly / daily / monthly from one daily frame.

    Pure; resamples internally. Returns an ordered dict with EXACTLY the keys
    ``("1w", "1d", "1mo")`` in that order (mirroring
    :func:`screener.patterns.detect_all_timeframes`). On empty/short/None input
    every value is an empty :class:`LevelSet` (each call fail-softs), so the 3-key
    shape is always returned and callers can iterate a stable structure. The result
    is picklable.
    """
    return {tf: support_resistance(daily_df, timeframe=tf) for tf in ("1w", "1d", "1mo")}


# --- buy zone ------------------------------------------------------------
def _buy_zone(df: pd.DataFrame, timeframe: str) -> Optional[BuyZone]:
    """Unguarded core of :func:`buy_zone` (the public fn wraps it)."""
    tol = CLUSTER_TOL_PCT.get(timeframe, CLUSTER_TOL_PCT["1d"])
    df_tf = patterns.resample_ohlc(df, timeframe)
    ls = support_resistance(df, timeframe=timeframe)
    last_close = ls.last_close
    if not np.isfinite(last_close):
        last_close = _last_close(df_tf)
    if not np.isfinite(last_close):
        return None

    low: Optional[float] = None
    high: Optional[float] = None
    basis: Optional[str] = None

    # Primary: the nearest support below the close defines a band UP TO that line.
    if ls.supports:
        s = ls.supports[0]
        high = float(s.price)
        low = float(s.price) * (1.0 - tol)
        basis = f"nearest support · {s.touches} touches"
    else:
        # Fallback: a rising 20-EMA pullback band (only if the EMA is sloping up),
        # reusing the shallow-hold-just-above-the-EMA idea from pullback_quality.
        if df_tf is None or "close" not in getattr(df_tf, "columns", []):
            return None
        e = ema(df_tf["close"], _EMA_ZONE_LEN)
        if len(e) >= _EMA_ZONE_SPAN + 1:
            now = float(e.iloc[-1])
            prior = float(e.iloc[-(_EMA_ZONE_SPAN + 1)])
            if np.isfinite(now) and np.isfinite(prior) and now > prior:
                low = now * (1.0 - _EMA_ZONE_BELOW)
                high = now * (1.0 + _EMA_ZONE_ABOVE)
                basis = "20-EMA pullback"

    if low is None or high is None or basis is None:
        return None

    in_zone = bool(low <= last_close <= high)
    if in_zone:
        distance_pct = 0.0
    else:
        # Signed gap from the last close to the NEAREST edge, over the last close.
        nearest_edge = high if last_close > high else low
        distance_pct = (nearest_edge - last_close) / last_close

    return BuyZone(
        low=float(low),
        high=float(high),
        basis=basis,
        in_zone=in_zone,
        distance_pct=float(distance_pct),
        timeframe=timeframe,
    )


def buy_zone(df: pd.DataFrame, *, timeframe: str = "1d") -> Optional[BuyZone]:
    """A descriptive entry band for one ticker at one ``timeframe`` (or ``None``).

    Pure; never raises (wrapped in ``try/except`` -> ``None``). Primary band: the
    nearest support from :func:`support_resistance` — ``high = support.price``,
    ``low = support.price * (1 - CLUSTER_TOL_PCT[tf])``. Fallback when there is no
    support below the close: a rising 20-EMA pullback band (only if
    :func:`screener.indicators.ema` is sloping up over the look-back). Else
    ``None``. ``in_zone`` is whether the last close sits in ``[low, high]``;
    ``distance_pct`` is the signed gap to the nearest edge (``0.0`` inside). The
    band is DESCRIPTIVE — an educational entry zone, never advice or an exit call.
    The result is picklable.
    """
    try:
        return _buy_zone(df, timeframe)
    except Exception:
        return None
