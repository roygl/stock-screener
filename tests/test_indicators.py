"""Synthetic, network-free tests for the indicator engine (screener/indicators.py).

No ``pytest`` import and no ``yfinance``: every test is a plain ``test_*`` function
using ``assert`` + ``math.isclose`` so the suite runs BOTH under
``python -m pytest tests/test_indicators.py`` AND standalone as
``python tests/test_indicators.py`` (the ``__main__`` runner counts pass/fail,
prints a summary, and exits non-zero on any failure).

All inputs are hand-built so expected values are checkable by eye; where an exact
closed form is awkward we assert structural / property invariants instead.
"""

import math
import os
import sys

# Put the repo root on sys.path so "import screener" resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from screener import indicators as ind  # noqa: E402


# --- helpers -------------------------------------------------------------
def _series(values):
    """A close-like Series on a daily, oldest-first, tz-naive 'date' index."""
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    idx.name = "date"
    return pd.Series([float(v) for v in values], index=idx)


def _frame(close, volume=None):
    """Minimal canonical price frame from a close series (+ optional volume)."""
    if not isinstance(close, pd.Series):
        close = _series(close)
    if volume is None:
        volume = pd.Series(1_000_000.0, index=close.index)
    elif not isinstance(volume, pd.Series):
        volume = pd.Series([float(v) for v in volume], index=close.index)
    return pd.DataFrame(
        {
            "open": close.values,
            "high": close.values,
            "low": close.values,
            "close": close.values,
            "volume": volume.values,
        },
        index=close.index,
    )


def _ohlc_frame(rows):
    """A canonical price frame from explicit ``(high, low, close)`` rows.

    ``open`` mirrors ``close`` and ``volume`` is constant — only high/low/close
    matter for true-range / ATR. Oldest-first daily 'date' index.
    """
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    idx.name = "date"
    high = [float(h) for h, _, _ in rows]
    low = [float(lo) for _, lo, _ in rows]
    close = [float(c) for _, _, c in rows]
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1_000_000.0] * len(rows),
        },
        index=idx,
    )


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# --- sma -----------------------------------------------------------------
def test_sma_hand_checked():
    out = ind.sma(_series([1, 2, 3, 4, 5]), 3)
    # window of [3,4,5] -> 4.0
    assert math.isclose(out.iloc[-1], 4.0)
    # mean of [1,2,3] = 2, [2,3,4] = 3
    assert math.isclose(out.iloc[2], 2.0)
    assert math.isclose(out.iloc[3], 3.0)


def test_sma_first_n_minus_1_are_nan():
    out = ind.sma(_series([1, 2, 3, 4, 5]), 3)
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])
    assert not math.isnan(out.iloc[2])


def test_sma_constant_series_is_constant():
    out = ind.sma(_series([7] * 10), 4)
    tail = out.dropna()
    assert len(tail) == 7  # first 3 NaN
    assert all(math.isclose(v, 7.0) for v in tail)


def test_sma_length_preserved():
    s = _series(range(1, 21))
    assert len(ind.sma(s, 5)) == len(s)


# --- ema -----------------------------------------------------------------
def test_ema_constant_series_is_constant():
    out = ind.ema(_series([5] * 30), 9)
    assert all(math.isclose(v, 5.0) for v in out)


def test_ema_length_preserved_and_no_leading_nan():
    s = _series(range(1, 31))
    out = ind.ema(s, 9)
    assert len(out) == len(s)
    assert not out.isna().any()  # adjust=False warms up from the first bar


def test_ema_recursive_formula():
    # alpha = 2/(n+1); out_t = alpha*price + (1-alpha)*out_{t-1}; out_0 = price_0.
    s = _series([10, 12, 14])
    out = ind.ema(s, 4)
    alpha = 2.0 / (4 + 1)
    e0 = 10.0
    e1 = alpha * 12 + (1 - alpha) * e0
    e2 = alpha * 14 + (1 - alpha) * e1
    assert math.isclose(out.iloc[0], e0)
    assert math.isclose(out.iloc[1], e1)
    assert math.isclose(out.iloc[2], e2)


# --- rsi -----------------------------------------------------------------
def test_rsi_monotonically_rising_is_100():
    out = ind.rsi(_series(range(1, 40)), 14)
    assert math.isclose(out.iloc[-1], 100.0)


def test_rsi_monotonically_falling_is_0():
    out = ind.rsi(_series(range(40, 1, -1)), 14)
    assert math.isclose(out.iloc[-1], 0.0)


def test_rsi_within_bounds():
    rng = np.random.default_rng(42)
    walk = 100 + np.cumsum(rng.normal(0, 1, 200))
    out = ind.rsi(_series(walk), 14).dropna()
    assert ((out >= 0.0) & (out <= 100.0)).all()


def test_rsi_warmup_is_nan():
    out = ind.rsi(_series(range(1, 40)), 14)
    # need n changes -> first n values NaN, value at index n present.
    assert math.isnan(out.iloc[13])
    assert not math.isnan(out.iloc[14])


def test_rsi_flat_series_is_50():
    out = ind.rsi(_series([50] * 30), 14).dropna()
    assert len(out) > 0
    assert all(math.isclose(v, 50.0) for v in out)


def test_rsi_seed_is_simple_mean_of_first_n_changes():
    # Canonical Wilder seed (index n) = simple mean of the first n deltas.
    # A balanced +1/-1 alternation -> 7 gains & 7 losses of equal size -> RSI 50
    # exactly AT THE SEED BAR (later bars drift via the recursion, so we pin n).
    vals = [100]
    for i in range(1, 40):
        vals.append(vals[-1] + (1 if i % 2 else -1))
    out = ind.rsi(_series(vals), 14)
    assert math.isnan(out.iloc[13])               # warm-up
    assert math.isclose(out.iloc[14], 50.0, abs_tol=1e-9)  # seed bar, balanced


def test_rsi_matches_reference_wilder_loop():
    # Independent canonical implementation (simple-mean seed + Wilder recursion).
    def wilder_rsi(vals, n=14):
        v = np.asarray(vals, dtype="float64")
        d = np.diff(v)
        gain = np.where(d > 0, d, 0.0)
        loss = np.where(d < 0, -d, 0.0)
        out = [float("nan")] * len(v)
        if len(v) <= n:
            return np.array(out)
        ag = gain[:n].mean()
        al = loss[:n].mean()

        def pt(ag, al):
            if al == 0:
                return 50.0 if ag == 0 else 100.0
            return 100.0 - 100.0 / (1.0 + ag / al)

        out[n] = pt(ag, al)
        for i in range(n, len(d)):
            ag = (ag * (n - 1) + gain[i]) / n
            al = (al * (n - 1) + loss[i]) / n
            out[i + 1] = pt(ag, al)
        return np.array(out)

    rng = np.random.default_rng(11)
    vals = 100 + np.cumsum(rng.normal(0, 1.5, 120))
    mine = ind.rsi(_series(vals), 14).to_numpy()
    ref = wilder_rsi(vals, 14)
    mask = ~np.isnan(ref)
    assert (~np.isnan(mine[mask])).all()
    assert np.allclose(mine[mask], ref[mask], atol=1e-9)


# --- macd ----------------------------------------------------------------
def test_macd_constant_series_all_zero():
    out = ind.macd(_series([20] * 60))
    assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in out["macd"])
    assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in out["signal"])
    assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in out["histogram"])


def test_macd_histogram_is_macd_minus_signal():
    s = _series(100 + np.cumsum(np.sin(np.arange(80) / 3.0)))
    out = ind.macd(s)
    recomputed = out["macd"] - out["signal"]
    assert np.allclose(out["histogram"].values, recomputed.values)


def test_macd_columns_present():
    out = ind.macd(_series(range(1, 60)))
    assert list(out.columns) == ["macd", "signal", "histogram"]


def test_macd_line_equals_fast_minus_slow_ema():
    s = _series(100 + np.cumsum(np.ones(60)))
    out = ind.macd(s, fast=12, slow=26, signal=9)
    expected = ind.ema(s, 12) - ind.ema(s, 26)
    assert np.allclose(out["macd"].values, expected.values)


# --- trailing_return / momentum ------------------------------------------
def test_trailing_return_plus_10_percent():
    s = _series([100, 101, 102, 110])  # last vs 3 bars back: 110/100 - 1
    out = ind.trailing_return(s, 3)
    assert math.isclose(out.iloc[-1], 0.10)


def test_trailing_return_warmup_is_nan():
    out = ind.trailing_return(_series([100, 110, 121]), 3)
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[2])  # only 2 prior bars, need 3


def test_momentum_plus_10_percent_over_horizon():
    # +10% exactly over a 21-session (1mo) horizon: 22 bars, last = 1.10 * first.
    n = ind.TRADING_DAYS[1]
    base = [100.0] * (n + 1)
    base[-1] = 110.0
    assert math.isclose(ind.momentum(_series(base), 1), 0.10)


def test_momentum_insufficient_history_is_nan():
    # Fewer bars than the 12mo (252) horizon -> NaN.
    assert _isnan(ind.momentum(_series(range(1, 50)), 12))


def test_momentum_bad_horizon_is_nan():
    assert _isnan(ind.momentum(_series(range(1, 300)), 7))


def test_momentum_all_horizons_compute_on_long_series():
    s = _series(100 + np.arange(300) * 0.5)
    for m in (1, 3, 6, 12):
        assert not _isnan(ind.momentum(s, m))


# --- relative_volume -----------------------------------------------------
def test_relative_volume_hand_example():
    # prior n=3 volumes mean is known; current divided by it.
    vol = _series([10, 20, 30, 40, 200])  # reuse helper for index only
    out = ind.relative_volume(vol, n=3)
    # prior 3 of the last bar = [20,30,40], mean 30; 200/30.
    assert math.isclose(out.iloc[-1], 200.0 / 30.0)


def test_relative_volume_excludes_current_bar():
    # Spike on the current bar must NOT inflate its own denominator.
    vol = _series([100, 100, 100, 100, 1_000_000])
    out = ind.relative_volume(vol, n=3)
    assert math.isclose(out.iloc[-1], 1_000_000.0 / 100.0)


def test_relative_volume_warmup_is_nan():
    vol = _series([10, 20, 30, 40])
    out = ind.relative_volume(vol, n=3)
    # need shift(1)+rolling(3) => first 3 NaN, index 3 is first defined.
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[2])
    assert not math.isnan(out.iloc[3])


def test_relative_volume_latest_matches_series():
    vol = _series([10, 20, 30, 40, 50, 60])
    assert math.isclose(
        ind.relative_volume_latest(vol, n=3), ind.relative_volume(vol, n=3).iloc[-1]
    )


def test_relative_volume_zero_prior_window_is_nan():
    # Degenerate all-zero prior window: 0 baseline reads NaN, not +inf, so it can't
    # sort to the top of a ranker. Mirrors the zero-guards in dist/pct_from_ma.
    vol = _series([0, 0, 0, 0, 500])
    out = ind.relative_volume(vol, n=3)  # prior 3 of last bar = [0,0,0], mean 0
    assert math.isnan(out.iloc[-1])
    assert _isnan(ind.relative_volume_latest(vol, n=3))


# --- distance_from_high --------------------------------------------------
def test_distance_from_high_minus_10_percent():
    # high 100 in window, last 90 -> -0.10
    s = _series([100, 95, 90])
    assert math.isclose(ind.distance_from_high(s, window=252), -0.10)


def test_distance_from_high_at_fresh_high_is_zero():
    s = _series([10, 20, 30, 40, 50])
    assert math.isclose(ind.distance_from_high(s, window=252), 0.0)


def test_distance_from_high_always_non_positive():
    rng = np.random.default_rng(7)
    walk = 100 + np.cumsum(rng.normal(0, 2, 120))
    assert ind.distance_from_high(_series(walk), window=252) <= 0.0


def test_distance_from_high_ipo_fewer_than_window():
    # Only 5 bars but window 252: uses min_periods=1, distance from available high.
    s = _series([10, 12, 15, 11, 13])
    out = ind.distance_from_high(s, window=252)
    assert math.isclose(out, 13.0 / 15.0 - 1.0)
    assert not math.isnan(out)


def test_distance_from_high_empty_is_nan():
    assert _isnan(ind.distance_from_high(pd.Series(dtype="float64")))


def test_distance_from_high_windowed():
    # window=3: high of the last 3 bars only.
    s = _series([100, 1, 2, 4])  # last 3 = [1,2,4], high 4, last 4 -> 0.0
    assert math.isclose(ind.distance_from_high(s, window=3), 0.0)


# --- price_above_ma / pct_from_ma ----------------------------------------
def test_price_above_ma_true_case():
    # rising series: last close above its SMA.
    s = _series(range(1, 21))
    assert ind.price_above_ma(s, 5, "sma") is True


def test_price_above_ma_false_case():
    # falling series: last close below its SMA.
    s = _series(range(20, 0, -1))
    assert ind.price_above_ma(s, 5, "sma") is False


def test_price_above_ma_none_on_insufficient_history():
    s = _series([1, 2, 3])
    assert ind.price_above_ma(s, 5, "sma") is None


def test_price_above_ma_ema_kind():
    s = _series(range(1, 21))
    assert ind.price_above_ma(s, 5, "ema") is True  # EMA defined from bar 1


def test_pct_from_ma_value():
    # constant 100 series: SMA == 100, pct == 0.
    s = _series([100] * 10)
    assert math.isclose(ind.pct_from_ma(s, 5, "sma"), 0.0)


def test_pct_from_ma_above():
    s = _series([10, 10, 10, 10, 13])  # SMA5 = 10.6, last 13
    out = ind.pct_from_ma(s, 5, "sma")
    assert math.isclose(out, 13.0 / 10.6 - 1.0)
    assert out > 0.0


def test_pct_from_ma_nan_on_insufficient_history():
    assert _isnan(ind.pct_from_ma(_series([1, 2, 3]), 5, "sma"))


def test_price_above_ma_none_when_latest_close_is_nan():
    # A NaN latest close is "unknown", not "below": price_above_ma reads None,
    # consistent with pct_from_ma reading NaN (the MA itself is still defined).
    s = _series(list(range(1, 30)) + [np.nan])
    assert ind.price_above_ma(s, 5, "ema") is None
    assert _isnan(ind.pct_from_ma(s, 5, "ema"))


# --- is_stacked ----------------------------------------------------------
def test_is_stacked_true_on_rising_series():
    # Long rising series: short MA > mid MA > long MA at the latest bar.
    s = _series(np.arange(1, 201, dtype="float64"))
    assert ind.is_stacked(s, (20, 50, 150), "sma") is True


def test_is_stacked_false_on_flat_series():
    # Flat: all MAs equal -> not strictly greater.
    s = _series([100] * 200)
    assert ind.is_stacked(s, (20, 50, 150), "sma") is False


def test_is_stacked_false_when_too_short():
    # Shorter than the longest window -> some MA undefined -> False.
    s = _series(np.arange(1, 60, dtype="float64"))
    assert ind.is_stacked(s, (20, 50, 150), "sma") is False


def test_is_stacked_false_on_falling_series():
    s = _series(np.arange(200, 0, -1, dtype="float64"))
    assert ind.is_stacked(s, (20, 50, 150), "sma") is False


def test_is_stacked_false_on_empty_windows():
    # No MAs to compare: there is no stack to establish, so False (not a vacuous
    # all()-over-nothing True), even on a long, cleanly rising series.
    s = _series(np.arange(1, 201, dtype="float64"))
    assert ind.is_stacked(s, (), "sma") is False


# --- ema_cross / latest_ema_cross ----------------------------------------
def _crossover_series():
    """Falling then rising: forces a fast/slow EMA cross UP late in the series."""
    down = list(np.arange(50, 20, -1, dtype="float64"))   # 30 bars down
    up = list(np.arange(20, 60, dtype="float64"))          # 40 bars up
    return _series(down + up)


def test_ema_cross_marks_bullish_cross():
    s = _crossover_series()
    cr = ind.ema_cross(s, fast=5, slow=9)
    ups = cr[cr == 1]
    assert len(ups) >= 1
    # On the +1 bar the fast EMA goes from <= slow to > slow.
    pos = cr.index.get_loc(ups.index[-1])
    f = ind.ema(s, 5)
    sl = ind.ema(s, 9)
    assert f.iloc[pos - 1] <= sl.iloc[pos - 1]
    assert f.iloc[pos] > sl.iloc[pos]


def test_ema_cross_marks_bearish_cross():
    # Rising then falling -> a cross DOWN.
    up = list(np.arange(20, 60, dtype="float64"))
    down = list(np.arange(60, 20, -1, dtype="float64"))
    cr = ind.ema_cross(_series(up + down), fast=5, slow=9)
    assert (cr == -1).any()


def test_ema_cross_first_bar_is_zero():
    cr = ind.ema_cross(_series(range(1, 30)), fast=5, slow=9)
    assert cr.iloc[0] == 0


def test_ema_cross_values_only_in_set():
    cr = ind.ema_cross(_crossover_series(), fast=5, slow=9)
    assert set(cr.unique()).issubset({-1, 0, 1})


def test_latest_ema_cross_bullish_state():
    s = _crossover_series()  # ends rising -> fast > slow
    res = ind.latest_ema_cross(s, fast=5, slow=9)
    assert res.state == "bullish"
    assert res.bars_since_cross is not None
    assert res.bars_since_cross >= 0


def test_latest_ema_cross_bearish_state():
    up = list(np.arange(20, 60, dtype="float64"))
    down = list(np.arange(60, 20, -1, dtype="float64"))
    res = ind.latest_ema_cross(_series(up + down), fast=5, slow=9)
    assert res.state == "bearish"


def test_latest_ema_cross_event_on_cross_bar():
    # Truncate exactly at the bullish cross bar so event == "up", bars_since == 0.
    s = _crossover_series()
    cr = ind.ema_cross(s, fast=5, slow=9)
    first_up = cr[cr == 1].index[0]
    pos = cr.index.get_loc(first_up)
    truncated = s.iloc[: pos + 1]
    res = ind.latest_ema_cross(truncated, fast=5, slow=9)
    assert res.event == "up"
    assert res.bars_since_cross == 0
    assert res.state == "bullish"


def test_latest_ema_cross_no_cross_returns_none():
    # Constant series: fast and slow EMAs are equal every bar, so the difference
    # never changes sign -> no cross at all. (A monotonic series DOES cross at
    # bar 1, because both EMAs seed to the same first value.)
    res = ind.latest_ema_cross(_series([100.0] * 40), fast=5, slow=9)
    assert res.event == "none"
    assert res.bars_since_cross is None
    assert res.state == "bearish"  # fast == slow is not strictly greater


def test_ema_cross_constant_series_has_no_cross():
    cr = ind.ema_cross(_series([42.0] * 30), fast=5, slow=9)
    assert (cr == 0).all()


def test_latest_ema_cross_empty_safe():
    res = ind.latest_ema_cross(pd.Series(dtype="float64"))
    assert res.event == "none"
    assert res.bars_since_cross is None
    assert res.to_dict()["state"] in {"bullish", "bearish"}


# --- true_range / atr ----------------------------------------------------
def test_true_range_hand_checked():
    # Bar 0: no prev close -> high-low = 10-8 = 2.
    # Bar 1: prev_close=9 -> max(12-9=3, |12-9|=3, |9-9|=0) = 3.
    # Bar 2: prev_close=11 -> max(11-7=4, |11-11|=0, |7-11|=4) = 4.
    df = _ohlc_frame([(10, 8, 9), (12, 9, 11), (11, 7, 8)])
    tr = ind.true_range(df)
    assert len(tr) == 3
    assert math.isclose(tr.iloc[0], 2.0)
    assert math.isclose(tr.iloc[1], 3.0)
    assert math.isclose(tr.iloc[2], 4.0)


def test_true_range_gap_beats_high_low():
    # A gap up makes |high - prev_close| exceed the intrabar high-low range.
    df = _ohlc_frame([(10, 9, 10), (20, 19, 20)])  # prev_close 10, high 20
    tr = ind.true_range(df)
    assert math.isclose(tr.iloc[1], 20.0 - 10.0)  # 10, not the 1.0 high-low


def test_true_range_empty_is_empty():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert len(ind.true_range(empty)) == 0


def test_atr_positive_and_length_preserved():
    rng = np.random.default_rng(3)
    close = _series(100 + np.cumsum(rng.normal(0, 1, 60)))
    df = _frame(close)  # high==low==close here, but diffs across bars give TR > 0
    a = ind.atr(df, 14)
    assert len(a) == len(df)
    tail = a.dropna()
    assert len(tail) > 0
    assert (tail >= 0.0).all()


def test_atr_warmup_is_nan_then_defined():
    df = _ohlc_frame([(i + 1.0, i - 0.5, i + 0.5) for i in range(1, 21)])
    a = ind.atr(df, 14)
    assert math.isnan(a.iloc[12])      # first n-1 == 13 values NaN
    assert not math.isnan(a.iloc[13])  # seed at index n-1


def test_atr_constant_range_equals_that_range():
    # Every bar spans exactly 2.0; first bar TR = high-low = 2.0 too, so the
    # Wilder seed and recursion both stay at 2.0.
    rows = [(c + 1.0, c - 1.0, float(c)) for c in range(10, 50)]
    a = ind.atr(_ohlc_frame(rows), 14)
    assert math.isclose(a.iloc[-1], 2.0, abs_tol=1e-9)


def test_atr_latest_matches_series():
    rng = np.random.default_rng(5)
    df = _frame(_series(100 + np.cumsum(rng.normal(0, 1, 50))))
    assert math.isclose(ind.atr_latest(df, 14), ind.atr(df, 14).iloc[-1])


def test_atr_latest_short_frame_is_nan():
    df = _frame([10, 11, 12])  # fewer than n=14 bars
    assert _isnan(ind.atr_latest(df, 14))


# --- consecutive_up_run --------------------------------------------------
def test_consecutive_up_run_known_sequence():
    # ...,7,5,6,7,8,9 -> last five strictly rising (5<6<7<8<9): run of 4 from the
    # 6 onward? Count back: 9>8,8>7,7>6,6>5 -> 4; then 5>7 is False -> stop.
    s = _series([1, 2, 7, 5, 6, 7, 8, 9])
    assert ind.consecutive_up_run(s) == 4


def test_consecutive_up_run_zero_when_last_bar_down():
    s = _series([1, 2, 3, 4, 3])  # last bar is a down-close
    assert ind.consecutive_up_run(s) == 0


def test_consecutive_up_run_zero_when_last_bar_flat():
    s = _series([1, 2, 3, 3])  # last bar equal -> not strictly up
    assert ind.consecutive_up_run(s) == 0


def test_consecutive_up_run_all_rising():
    s = _series(range(1, 11))  # nine up-moves over ten bars
    assert ind.consecutive_up_run(s) == 9


def test_consecutive_up_run_short_input_is_zero():
    assert ind.consecutive_up_run(_series([5])) == 0
    assert ind.consecutive_up_run(pd.Series(dtype="float64")) == 0


# --- extension_state -----------------------------------------------------
def _parabolic_close():
    """A steep, accelerating advance: late bars rip far above the 20/50-EMA."""
    base = list(100 + np.arange(60) * 0.2)            # gentle ramp to seed the MAs
    blow_off = list(base[-1] + np.cumsum(np.arange(1, 26) * 0.9))  # accelerating tail
    return _series(base + blow_off)


def test_extension_state_parabolic_on_steep_accelerating():
    res = ind.extension_state(_frame(_parabolic_close()))
    assert res.state == "parabolic"
    assert res.score >= ind.PARABOLIC_CUT
    assert res.pct_above_ema20 > 0.0
    assert res.up_run >= 1
    assert isinstance(res.to_dict(), dict)


def test_extension_state_normal_on_flat():
    res = ind.extension_state(_frame([100.0] * 120))
    assert res.state == "normal"
    assert math.isclose(res.score, 0.0, abs_tol=1e-9) or res.score < ind.EXTENDED_CUT
    # flat -> sitting on its own EMAs, RSI 50, no up-run
    assert abs(res.pct_above_ema20) < 1e-9
    assert res.up_run == 0


def test_extension_state_not_parabolic_on_volatile_downtrend():
    # A falling stock, volatile enough to light up ATR, must never read parabolic:
    # the pct_above_ema20 > 0 hard floor blocks it.
    rng = np.random.default_rng(9)
    down = 300 - np.arange(120) * 1.5 + rng.normal(0, 3, 120)
    res = ind.extension_state(_frame(_series(down)))
    assert res.state != "parabolic"
    assert res.pct_above_ema20 <= 0.0


def test_extension_state_fail_soft_empty():
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="date"),
    )
    res = ind.extension_state(empty)
    assert res.state == "normal"
    assert math.isclose(res.score, 0.0)
    assert _isnan(res.pct_above_ema20)
    assert _isnan(res.atr_pct)
    assert res.up_run == 0


def test_extension_state_fail_soft_short():
    res = ind.extension_state(_frame([10, 11, 12]))
    assert res.state == "normal"
    assert math.isclose(res.score, 0.0)
    assert res.up_run == 0


def test_extension_state_missing_close_column_safe():
    df = pd.DataFrame({"volume": [1.0, 2.0, 3.0]})
    res = ind.extension_state(df)
    assert res.state == "normal"
    assert _isnan(res.rsi)


# --- snapshot ------------------------------------------------------------
EXPECTED_KEYS = {
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "rel_volume_20", "dist_52w_high",
    "sma_20", "sma_50", "sma_150", "ema_5", "ema_9",
    "price_above_sma_20", "price_above_sma_50", "price_above_sma_150",
    "sma_stacked_20_50_150", "ema_5_9_state", "ema_5_9_event",
    "extension_state", "extension_score",
}


def test_snapshot_has_every_key_on_long_frame():
    rng = np.random.default_rng(1)
    close = _series(100 + np.cumsum(rng.normal(0.1, 1, 300)))
    vol = pd.Series(rng.integers(1_000_000, 5_000_000, 300).astype("float64"), index=close.index)
    snap = ind.snapshot(_frame(close, vol))
    assert set(snap.keys()) == EXPECTED_KEYS


def test_snapshot_values_are_finite_on_long_frame():
    close = _series(100 + np.arange(300) * 0.3)
    snap = ind.snapshot(_frame(close))
    for key in ("momentum_12m", "rsi_14", "macd", "sma_150", "ema_5", "dist_52w_high"):
        assert not _isnan(snap[key]), key
    assert snap["ema_5_9_state"] in {"bullish", "bearish"}
    assert snap["ema_5_9_event"] in {"up", "down", "none"}
    assert isinstance(snap["price_above_sma_20"], bool)
    assert isinstance(snap["sma_stacked_20_50_150"], bool)


def test_snapshot_contains_extension_keys():
    # Assert PRESENCE + types of the two new universe-wide keys (not exact equality
    # to any external key set — engine.SNAPSHOT_KEYS is out of scope here).
    close = _series(100 + np.arange(80) * 0.3)
    snap = ind.snapshot(_frame(close))
    assert "extension_state" in snap
    assert "extension_score" in snap
    assert isinstance(snap["extension_state"], str)
    assert snap["extension_state"] in {"normal", "extended", "parabolic"}
    assert isinstance(snap["extension_score"], float)
    assert 0.0 <= snap["extension_score"] <= 1.0


def test_snapshot_extension_keys_present_on_short_and_empty():
    # Fail-soft: the keys exist even when the frame is too short or empty, with the
    # neutral default ("normal" / 0.0).
    short = ind.snapshot(_frame([10, 11, 12]))
    assert short["extension_state"] == "normal"
    assert math.isclose(short["extension_score"], 0.0)
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="date"),
    )
    snap = ind.snapshot(empty)
    assert "extension_state" in snap and "extension_score" in snap
    assert snap["extension_state"] == "normal"


def test_snapshot_degrades_on_short_frame():
    # 5 bars: long-horizon features are NaN/None, but nothing raises and all keys exist.
    snap = ind.snapshot(_frame([10, 11, 12, 13, 14]))
    assert set(snap.keys()) == EXPECTED_KEYS
    assert _isnan(snap["momentum_12m"])
    assert _isnan(snap["sma_150"])
    assert _isnan(snap["rsi_14"])             # needs 15 bars
    assert snap["price_above_sma_150"] is None
    assert snap["sma_stacked_20_50_150"] is False
    # short-horizon, EMA-based fields still resolve
    assert not _isnan(snap["ema_5"])
    assert snap["ema_5_9_state"] in {"bullish", "bearish"}
    # 52w-high distance uses min_periods=1, so it computes even here
    assert not _isnan(snap["dist_52w_high"])
    assert snap["dist_52w_high"] <= 0.0


def test_snapshot_empty_frame_safe():
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="date"),
    )
    snap = ind.snapshot(empty)
    assert set(snap.keys()) == EXPECTED_KEYS
    assert _isnan(snap["momentum_1m"])
    assert _isnan(snap["dist_52w_high"])
    assert _isnan(snap["sma_20"])
    assert snap["price_above_sma_20"] is None
    assert snap["sma_stacked_20_50_150"] is False
    assert snap["ema_5_9_event"] == "none"


def test_snapshot_missing_volume_column_safe():
    # A frame without 'volume' must not raise; rel_volume_20 -> NaN.
    close = _series(100 + np.arange(60) * 0.2)
    df = pd.DataFrame({"close": close.values}, index=close.index)
    snap = ind.snapshot(df)
    assert set(snap.keys()) == EXPECTED_KEYS
    assert _isnan(snap["rel_volume_20"])
    assert not _isnan(snap["ema_5"])


# --- standalone runner ---------------------------------------------------
def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report any unexpected error
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
    total = passed + failed
    print(f"\n{passed}/{total} passed" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
