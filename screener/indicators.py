"""Price-derived technical indicators — a pure-function layer over the canonical frame.

The data layer (:mod:`screener.provider`) hands every ticker a canonical price
frame: columns ``open/high/low/close/volume`` (lower-cased), a tz-naive
``DatetimeIndex`` named ``date``, OLDEST row first, on an adjusted-close basis.
This module turns that frame into the technical signals the profiles screen on
(spec §6): momentum, RSI, MACD, moving-average structure, relative volume, and
distance from the 52-week high. Fundamentals (forward P/E, growth, market cap,
sector) already live in the provider and are NOT recomputed here — this layer
owns only what you can derive from price and volume.

Conventions (one rule, applied throughout):
- **Inputs.** The series-valued cores and ``relative_volume`` take a single
  ``pandas.Series`` (a column lifted off the frame) so they compose freely. The
  scalar snapshot reductions and every MA-structure / cross helper take that
  column directly (``close`` or ``volume``); :func:`snapshot` takes the whole
  price DataFrame. Each signature says which it expects.
- **No lookahead.** Every rolling window uses only bars up to and INCLUDING the
  current one. Averages use ``min_periods == window`` so an under-filled window
  is ``NaN`` (insufficient history is never a partial value) — the sole exception
  is the 52-week-high reduction, which uses ``min_periods=1`` on purpose so a
  recent IPO measures distance from its *available* high (documented on
  :func:`distance_from_high`).
- **Return types.** Series-valued functions return a ``Series`` aligned to the
  input index. Scalar "latest snapshot" functions return ``float`` (``NaN`` when
  undefined) or ``bool`` / ``None``. Nothing here raises on short or empty input;
  callers screening a whole universe get ``NaN``/``None`` and move on.

All functions are pure: no network, no file I/O, no global mutable state. The
``__main__`` smoke block is the only network user (it pulls one ticker via the
provider to print a live :func:`snapshot`, mirroring ``provider.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# Trading-day horizons for EOD daily bars. 52 weeks == 252 sessions.
TRADING_DAYS = {1: 21, 3: 63, 6: 126, 12: 252}
WEEKS_52 = 252

# Default moving-average sets per spec §6: 20/50/150 SMA (long-term & momentum),
# 5/9 EMA (swing). The engine computes any SMA(n)/EMA(n); these are just defaults.
SMA_WINDOWS = (20, 50, 150)
EMA_FAST, EMA_SLOW = 5, 9


# --- series-valued core --------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    """Simple moving average over ``n`` bars (operates on the given series).

    ``min_periods=n`` so the first ``n-1`` values are ``NaN`` — an under-filled
    window never yields a partial mean.
    """
    return s.rolling(window=n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """Exponential moving average, ``span=n``, ``adjust=False``.

    The recursive trader EMA (today = α·price + (1-α)·yesterday, α = 2/(n+1));
    length is preserved and early values warm up from the first bar.
    """
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index over ``n`` bars, bounded to ``[0, 100]``.

    Wilder's smoothing (RMA, α = 1/n) of up/down moves — NOT a simple MA of
    gains/losses, and NOT pandas' ``ewm`` (whose ``adjust=False`` recursion seeds
    from the first bar, not the canonical mean). The first average is the simple
    mean of the first ``n`` changes, placed at index ``n``; thereafter
    ``avg = (avg·(n-1) + x) / n``. The first ``n`` values are ``NaN`` (insufficient
    history). A run with no losses pins to 100, with no gains to 0; a flat stretch
    is ``50``.
    """
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    if len(s) <= n:
        return out  # not enough changes to seed (need n deltas after the first bar)

    delta = s.diff().to_numpy()  # delta[0] is NaN (no prior bar)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Seed at index n: simple average of the first n changes (delta[1..n]).
    avg_gain = gain[1 : n + 1].mean()
    avg_loss = loss[1 : n + 1].mean()
    values = np.full(len(s), np.nan)
    values[n] = _rsi_point(avg_gain, avg_loss)

    # Wilder recursion for every later bar.
    for i in range(n + 1, len(s)):
        avg_gain = (avg_gain * (n - 1) + gain[i]) / n
        avg_loss = (avg_loss * (n - 1) + loss[i]) / n
        values[i] = _rsi_point(avg_gain, avg_loss)

    out.iloc[:] = values
    return out


def _rsi_point(avg_gain: float, avg_loss: float) -> float:
    """One RSI value from smoothed average gain/loss (handles the zero edges)."""
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0  # flat -> 50, pure up-run -> 100
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD as a DataFrame with columns ``macd``, ``signal``, ``histogram``.

    ``macd = ema(s, fast) - ema(s, slow)``; ``signal = ema(macd, signal)``;
    ``histogram = macd - signal``. EMAs warm up from the first bar (no leading
    ``NaN``), so a constant series gives all-zero columns.
    """
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    hist = line - sig
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": hist})


def trailing_return(s: pd.Series, periods: int) -> pd.Series:
    """Return over the trailing ``periods`` bars: ``s / s.shift(periods) - 1``.

    The first ``periods`` values are ``NaN`` (no bar to compare against). A move
    of exactly +10% over the horizon reads ``0.10``.
    """
    return s / s.shift(periods) - 1.0


def relative_volume(volume: pd.Series, n: int = 20) -> pd.Series:
    """Today's volume vs the mean of the PRIOR ``n`` bars (current bar excluded).

    ``volume / volume.shift(1).rolling(n).mean()`` — a ratio of today against its
    recent baseline. Shifting by one keeps the current bar out of its own
    benchmark (no lookahead); the first ``n`` values are ``NaN``. A zero baseline
    (a degenerate all-zero prior window) reads ``NaN`` ("undefined") rather than
    ``inf``, mirroring the zero-guards in :func:`distance_from_high`/:func:`pct_from_ma`.
    """
    prior_mean = volume.shift(1).rolling(window=n, min_periods=n).mean()
    return volume / prior_mean.replace(0.0, np.nan)


# --- scalar snapshot reductions ------------------------------------------
def latest(series: pd.Series) -> float:
    """Last value of a series indicator as a ``float`` (``NaN`` if empty/undefined)."""
    if series is None or len(series) == 0:
        return float("nan")
    return float(series.iloc[-1])


def momentum(close: pd.Series, months: int) -> float:
    """Latest trailing return over a ``{1,3,6,12}``-month horizon (``float``).

    Maps months to sessions via :data:`TRADING_DAYS` (1mo=21 … 12mo=252) and reads
    the last bar of :func:`trailing_return`. ``NaN`` when history is shorter than
    the horizon, or when ``months`` is not one of the four supported horizons.
    """
    periods = TRADING_DAYS.get(months)
    if periods is None:
        return float("nan")
    return latest(trailing_return(close, periods))


def relative_volume_latest(volume: pd.Series, n: int = 20) -> float:
    """Latest bar of :func:`relative_volume` as a ``float`` (``NaN`` if < ``n+1`` bars)."""
    return latest(relative_volume(volume, n))


def distance_from_high(close: pd.Series, window: int = WEEKS_52) -> float:
    """Latest close vs the trailing-``window`` high: ``(last - high) / high`` (``≤ 0``).

    Uses ``min_periods=1`` (the documented exception to the
    ``min_periods == window`` rule) so a recent IPO with fewer than ``window``
    bars measures distance from its *available* high rather than ``NaN``; at a
    fresh high the result is ``0.0``. ``NaN`` only for an empty series.
    """
    if close is None or len(close) == 0:
        return float("nan")
    rolling_max = close.rolling(window=window, min_periods=1).max()
    high = float(rolling_max.iloc[-1])
    if not np.isfinite(high) or high == 0.0:
        return float("nan")
    return float(close.iloc[-1]) / high - 1.0


def rsi_latest(close: pd.Series, n: int = 14) -> float:
    """Latest :func:`rsi` value as a ``float`` (``NaN`` if fewer than ``n+1`` bars)."""
    return latest(rsi(close, n))


def macd_latest(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Latest MACD reading as ``{'macd', 'signal', 'histogram'}`` floats.

    Convenience reduction of :func:`macd` to its final bar; values are ``NaN``
    only for an empty series (EMAs otherwise warm up from the first bar).
    """
    frame = macd(close, fast, slow, signal)
    return {col: latest(frame[col]) for col in ("macd", "signal", "histogram")}


# --- MA-structure helpers (evaluated at the latest bar) ------------------
def _latest_ma(close: pd.Series, n: int, kind: str) -> float:
    """Latest SMA/EMA value (``float``); ``NaN`` when the MA is undefined.

    EMA warms up from the first bar, so it's undefined only on an empty series;
    SMA needs ``n`` bars (``min_periods=n``), so it's ``NaN`` until then.
    """
    if kind == "sma":
        return latest(sma(close, n))
    if kind == "ema":
        return latest(ema(close, n))
    raise ValueError(f"kind must be 'sma' or 'ema', got {kind!r}")


def price_above_ma(close: pd.Series, n: int, kind: str = "sma") -> "bool | None":
    """Is the latest close above its ``n``-bar MA? ``None`` if either is undefined.

    ``kind`` in ``{"sma","ema"}``. Returns ``None`` (not ``False``) when there is
    too little history to define the MA, or the latest close itself is ``NaN``, so
    "unknown" stays distinct from "below" — consistent with :func:`pct_from_ma`,
    which also reads ``NaN`` in those cases.
    """
    ma = _latest_ma(close, n, kind)
    if not np.isfinite(ma) or len(close) == 0 or not np.isfinite(close.iloc[-1]):
        return None
    return bool(float(close.iloc[-1]) > ma)


def pct_from_ma(close: pd.Series, n: int, kind: str = "sma") -> float:
    """Latest close's distance from its ``n``-bar MA: ``(last - ma) / ma`` (``float``).

    Positive above the average, negative below; ``NaN`` if the MA is undefined or
    zero. ``kind`` in ``{"sma","ema"}``.
    """
    ma = _latest_ma(close, n, kind)
    if not np.isfinite(ma) or ma == 0.0:
        return float("nan")
    return float(close.iloc[-1]) / ma - 1.0


def is_stacked(close: pd.Series, windows=SMA_WINDOWS, kind: str = "sma") -> bool:
    """Are the MAs strictly stacked ``MA(w0) > MA(w1) > …`` at the latest bar?

    The classic trend-template alignment (e.g. 20 > 50 > 150 SMA). ``False`` if
    any MA in ``windows`` is undefined (insufficient history), the order breaks,
    or fewer than two windows are given (no stack to establish).
    """
    values = [_latest_ma(close, w, kind) for w in windows]
    if len(values) < 2 or any(not np.isfinite(v) for v in values):
        return False
    return all(a > b for a, b in zip(values, values[1:]))


def ema_cross(close: pd.Series, fast: int = EMA_FAST, slow: int = EMA_SLOW) -> pd.Series:
    """Per-bar fast/slow EMA crossover signal as an ``int`` Series.

    ``+1`` on a bar where the fast EMA crosses ABOVE the slow (fast ≤ slow on the
    prior bar, fast > slow now), ``-1`` on a cross below, ``0`` otherwise. The
    first bar is ``0`` (no prior to compare). Aligned to ``close``'s index.
    """
    f = ema(close, fast)
    s = ema(close, slow)
    diff = f - s
    prev = diff.shift(1)
    cross_up = (prev <= 0) & (diff > 0)
    cross_down = (prev >= 0) & (diff < 0)
    out = pd.Series(0, index=close.index, dtype="int64")
    out[cross_up] = 1
    out[cross_down] = -1
    return out


@dataclass(frozen=True)
class EmaCrossState:
    """Latest fast/slow EMA cross summary.

    ``state``  — ``"bullish"`` if fast > slow at the latest bar, else ``"bearish"``.
    ``event``  — ``"up"`` / ``"down"`` if the cross happened ON the latest bar,
                 else ``"none"``.
    ``bars_since_cross`` — bars since the most recent cross (``0`` if it is the
                 latest bar), or ``None`` if no cross occurred in the series.
    """

    state: str
    event: str
    bars_since_cross: "int | None"

    def to_dict(self) -> dict:
        return asdict(self)


def latest_ema_cross(close: pd.Series, fast: int = EMA_FAST, slow: int = EMA_SLOW) -> EmaCrossState:
    """Summarise the latest fast/slow EMA cross as an :class:`EmaCrossState`.

    Empty input yields ``state="bearish"``, ``event="none"``, ``bars_since_cross
    =None`` (a missing-data-safe default; callers screen on the fields, not on the
    object identity).
    """
    if close is None or len(close) == 0:
        return EmaCrossState(state="bearish", event="none", bars_since_cross=None)

    f = ema(close, fast)
    s = ema(close, slow)
    state = "bullish" if float(f.iloc[-1]) > float(s.iloc[-1]) else "bearish"

    crosses = ema_cross(close, fast, slow)
    last = int(crosses.iloc[-1])
    event = "up" if last == 1 else "down" if last == -1 else "none"

    nonzero = np.flatnonzero(crosses.to_numpy() != 0)
    bars_since = int(len(crosses) - 1 - nonzero[-1]) if len(nonzero) else None
    return EmaCrossState(state=state, event=event, bars_since_cross=bars_since)


# --- integration / demonstration -----------------------------------------
def snapshot(price_df: pd.DataFrame) -> "dict[str, float | bool | None]":
    """Full latest scalar feature set for one ticker (every spec §6 PRICE variable).

    Computes momentum (1/3/6/12-mo), RSI(14), MACD line/signal/histogram, 20-bar
    relative volume, 52-week-high distance, the 20/50/150 SMAs and 5/9 EMAs,
    price-above-SMA flags, the SMA stack flag, and the 5/9 EMA cross state/event —
    keyed exactly as documented below. Missing-data-safe: every value degrades to
    ``NaN``/``None`` (and ``False`` for the stack flag) on a short or empty frame;
    this never raises.
    """
    close = price_df["close"] if "close" in price_df.columns else pd.Series(dtype="float64")
    volume = price_df["volume"] if "volume" in price_df.columns else pd.Series(dtype="float64")

    macd_vals = macd_latest(close)
    cross = latest_ema_cross(close, EMA_FAST, EMA_SLOW)

    return {
        # momentum (trailing returns)
        "momentum_1m": momentum(close, 1),
        "momentum_3m": momentum(close, 3),
        "momentum_6m": momentum(close, 6),
        "momentum_12m": momentum(close, 12),
        # oscillators
        "rsi_14": rsi_latest(close, 14),
        "macd": macd_vals["macd"],
        "macd_signal": macd_vals["signal"],
        "macd_hist": macd_vals["histogram"],
        # volume / extension
        "rel_volume_20": relative_volume_latest(volume, 20),
        "dist_52w_high": distance_from_high(close, WEEKS_52),
        # moving averages (levels)
        "sma_20": _latest_ma(close, 20, "sma"),
        "sma_50": _latest_ma(close, 50, "sma"),
        "sma_150": _latest_ma(close, 150, "sma"),
        "ema_5": _latest_ma(close, EMA_FAST, "ema"),
        "ema_9": _latest_ma(close, EMA_SLOW, "ema"),
        # MA structure
        "price_above_sma_20": price_above_ma(close, 20, "sma"),
        "price_above_sma_50": price_above_ma(close, 50, "sma"),
        "price_above_sma_150": price_above_ma(close, 150, "sma"),
        "sma_stacked_20_50_150": is_stacked(close, SMA_WINDOWS, "sma"),
        # 5/9 EMA cross
        "ema_5_9_state": cross.state,
        "ema_5_9_event": cross.event,
    }


if __name__ == "__main__":  # smoke test: live snapshot for one ticker (mirrors provider.py)
    import logging
    import sys

    from .provider import YFinanceProvider

    logging.basicConfig(level=logging.WARNING)
    symbols = sys.argv[1:] or ["AAPL"]
    provider = YFinanceProvider()

    for sym in symbols:
        prices = provider.price_history(sym)
        snap = snapshot(prices)
        print(f"{sym}  ({len(prices)} bars)")
        for key, value in snap.items():
            if isinstance(value, float):
                print(f"    {key:24} {value:.4f}" if value == value else f"    {key:24} NaN")
            else:
                print(f"    {key:24} {value}")
