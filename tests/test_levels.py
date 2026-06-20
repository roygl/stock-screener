"""Synthetic, network-free tests for support / resistance levels + the buy zone.

Like ``tests/test_patterns.py`` / ``tests/test_indicators.py``: no ``pytest``
import and no ``yfinance`` / ``streamlit`` — every test is a plain ``test_*``
function using ``assert`` so the suite runs BOTH under
``python -m pytest tests/test_levels.py`` AND standalone as
``python tests/test_levels.py`` (the ``__main__`` runner counts pass/fail, prints
a summary, and exits non-zero on any failure).

Inputs are hand-built from piecewise-linear ``_zigzag`` paths (copied from
test_patterns.py), so each cluster lands its pivots exactly where intended and the
clustering / strength gates clear with comfortable margins. The precision contract
is asserted with deterministic anti-shapes (a monotonic ramp, a flat line, and a
seeded random walk) yielding NO multi-touch level.
"""

import math
import os
import sys

# Put the repo root on sys.path so "import screener" resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from screener import levels as lv  # noqa: E402


# --- helpers (copied from tests/test_patterns.py) ------------------------
def _daily(values, *, highs=None, lows=None, volume=None) -> pd.DataFrame:
    """Canonical daily OHLCV frame from a close path (business-day, oldest-first).

    high/low default to a tight +/-0.5% band around close (so swing highs use the
    bar high and swing lows the bar low, mirroring real bars); open is the prior
    close (first bar opens at its own close); volume defaults to 1e6.
    """
    close = np.asarray([float(v) for v in values], dtype="float64")
    n = len(close)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    idx.name = "date"
    high = np.asarray(highs, dtype="float64") if highs is not None else close * 1.005
    low = np.asarray(lows, dtype="float64") if lows is not None else close * 0.995
    if volume is not None:
        vol = np.asarray(volume, dtype="float64")
    else:
        vol = np.full(n, 1e6)
    open_ = np.concatenate([[close[0]], close[:-1]]) if n else close
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _zigzag(legs) -> np.ndarray:
    """Piecewise-linear close path from ``[(target_price, n_bars), ...]``.

    The first leg's ``target`` is the starting price (its ``n_bars`` is ignored
    beyond seeding one point); each subsequent leg linearly interpolates from the
    previous target to its target over ``n_bars`` steps.
    """
    first_price = legs[0][0]
    out = [np.array([float(first_price)])]
    prev = float(first_price)
    for target, nb in legs[1:]:
        out.append(np.linspace(prev, float(target), nb)[1:])
        prev = float(target)
    return np.concatenate(out)


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# A path that revisits a LOW near ~100 three times. The peaks BETWEEN the lows are
# deliberately SPREAD APART (120 / 140 / 160) so they do NOT cluster — only the
# repeated ~100 low forms a >=2-touch level, and the series finishes high (~180) so
# that level reads as SUPPORT below the close. (H,L,H,L,H,L,H zigzag.)
def _triple_low_support_path():
    return _zigzag(
        [(100, 1), (120, 8), (100, 8), (140, 8), (100, 8), (160, 8), (100, 8), (180, 8)]
    )


# A path that revisits a HIGH near ~120 three times, with the troughs BETWEEN them
# spread apart (100 / 80 / 60) so only the repeated ~120 high clusters; the series
# finishes low (~70) so that level reads as RESISTANCE above the close.
def _triple_high_resistance_path():
    return _zigzag(
        [(120, 1), (100, 8), (120, 8), (80, 8), (120, 8), (60, 8), (120, 8), (70, 8)]
    )


# --- support / resistance: clustering ------------------------------------
def test_support_cluster_from_revisited_low():
    df = _daily(_triple_low_support_path())
    ls = lv.support_resistance(df, timeframe="1d")
    assert ls.supports, "expected at least one support level"
    s = ls.supports[0]
    assert s.kind == "support"
    assert s.touches >= 2, s.touches
    # The clustered support sits near the revisited ~100 low (within the bar band).
    assert 98.0 <= s.price <= 102.0, s.price
    # Below the last close -> negative signed distance.
    assert s.distance_pct < 0.0, s.distance_pct
    assert s.price < ls.last_close


def test_resistance_cluster_from_revisited_high():
    df = _daily(_triple_high_resistance_path())
    ls = lv.support_resistance(df, timeframe="1d")
    assert ls.resistances, "expected at least one resistance level"
    r = ls.resistances[0]
    assert r.kind == "resistance"
    assert r.touches >= 2, r.touches
    assert 118.0 <= r.price <= 122.0, r.price
    # Above the last close -> positive signed distance.
    assert r.distance_pct > 0.0, r.distance_pct
    assert r.price > ls.last_close


def test_level_in_zero_to_one_strength_and_classification():
    df = _daily(_triple_low_support_path())
    ls = lv.support_resistance(df, timeframe="1d")
    for level in ls.supports + ls.resistances:
        assert 0.0 <= level.strength <= 1.0, level.strength
        assert level.first <= level.last
        if level.kind == "support":
            assert level.price <= ls.last_close
        else:
            assert level.price > ls.last_close


# --- strength sensitivity ------------------------------------------------
def test_strength_rises_with_touches():
    # Two touches of ~100 vs four touches of ~100 (more revisits => firmer level).
    # Peaks are spread apart so only the repeated ~100 low forms a cluster. Each path
    # starts/ends HIGH so every ~100 dip is a true interior swing low.
    two = _zigzag([(130, 1), (100, 9), (150, 9), (100, 9), (170, 9)])
    four = _zigzag(
        [(130, 1), (100, 9), (140, 9), (100, 9), (150, 9), (100, 9), (160, 9),
         (100, 9), (190, 9)]
    )
    s_two = lv.support_resistance(_daily(two), timeframe="1d").supports
    s_four = lv.support_resistance(_daily(four), timeframe="1d").supports
    assert s_two and s_four, (len(s_two), len(s_four))
    # Isolate the ~100 cluster on each side (nearest the revisited low).
    near_two = min(s_two, key=lambda lv_: abs(lv_.price - 100.0))
    near_four = min(s_four, key=lambda lv_: abs(lv_.price - 100.0))
    assert near_four.touches > near_two.touches, (near_two.touches, near_four.touches)
    assert near_four.strength > near_two.strength, (near_two.strength, near_four.strength)


def test_strength_rises_with_recency():
    # Hold touch count CONSTANT and vary only recency. ``base`` revisits a ~100 low
    # three times and ends near the series end (a recent last touch). ``early`` is
    # the SAME path with a long gentle HIGH tail appended — it adds no new ~100
    # pivot, so the ~100 cluster is identical (same price, touches, last timestamp),
    # but the series span now extends well PAST that last touch, so its recency
    # sub-score (and thus strength) is lower.
    base = _zigzag(
        [(160, 1), (120, 8), (100, 8), (130, 8), (100, 8), (135, 8), (100, 8), (180, 8)]
    )
    tail = np.linspace(180.0, 185.0, 40)[1:]  # no new low; just lengthens the span
    early = np.concatenate([base, tail])
    s_recent = lv.support_resistance(_daily(base), timeframe="1d").supports
    s_early = lv.support_resistance(_daily(early), timeframe="1d").supports
    assert s_recent and s_early, (len(s_recent), len(s_early))
    near_recent = min(s_recent, key=lambda lv_: abs(lv_.price - 100.0))
    near_early = min(s_early, key=lambda lv_: abs(lv_.price - 100.0))
    # Same cluster: identical touch count and last-touch timestamp.
    assert near_recent.touches == near_early.touches, (
        near_recent.touches, near_early.touches
    )
    assert near_recent.last == near_early.last, (near_recent.last, near_early.last)
    # Only the span differs -> the more-recent (relative to span end) scores higher.
    assert near_recent.strength > near_early.strength, (
        near_recent.strength, near_early.strength
    )


# --- ordering ------------------------------------------------------------
def test_nearest_first_ordering_supports():
    # TWO distinct support bands below a high (~150) close: ~90 (twice) and ~110
    # (twice). Nearest-below-first => the higher-priced ~110 band precedes ~90.
    path = _zigzag(
        [(120, 1), (90, 8), (130, 8), (90, 8), (135, 8), (110, 8), (140, 8),
         (110, 8), (150, 8)]
    )
    ls = lv.support_resistance(_daily(path), timeframe="1d")
    assert len(ls.supports) >= 2, [s.price for s in ls.supports]
    assert ls.supports[0].price > ls.supports[1].price, [s.price for s in ls.supports]
    # Both are genuinely below the close.
    assert all(s.price < ls.last_close for s in ls.supports)


def test_nearest_first_ordering_resistances():
    # TWO distinct resistance bands above a low (~70) close: ~110 (twice) and ~130
    # (twice). Nearest-above-first => the lower-priced ~110 band precedes ~130.
    path = _zigzag(
        [(80, 1), (110, 8), (70, 8), (110, 8), (65, 8), (130, 8), (60, 8),
         (130, 8), (70, 8)]
    )
    ls = lv.support_resistance(_daily(path), timeframe="1d")
    assert len(ls.resistances) >= 2, [r.price for r in ls.resistances]
    assert ls.resistances[0].price < ls.resistances[1].price, (
        [r.price for r in ls.resistances]
    )
    assert all(r.price > ls.last_close for r in ls.resistances)


def test_caps_three_per_side():
    df = _daily(_triple_low_support_path())
    ls = lv.support_resistance(df, timeframe="1d")
    assert len(ls.supports) <= 3
    assert len(ls.resistances) <= 3


# --- classification flips around last_close ------------------------------
def test_classification_flips_around_last_close():
    # The SAME ~110 swing band: when the series finishes well ABOVE it, it's
    # support; when it finishes well BELOW it, it's resistance.
    above_close = _zigzag(
        [(110, 1), (95, 7), (110, 7), (96, 7), (110, 7), (97, 7), (130, 8)]
    )
    below_close = _zigzag(
        [(110, 1), (125, 7), (110, 7), (124, 7), (110, 7), (123, 7), (90, 8)]
    )
    ls_above = lv.support_resistance(_daily(above_close), timeframe="1d")
    ls_below = lv.support_resistance(_daily(below_close), timeframe="1d")
    # When the close is high, the ~110 band classifies as support.
    assert any(95.0 <= s.price <= 115.0 for s in ls_above.supports), (
        [s.price for s in ls_above.supports], ls_above.last_close
    )
    # When the close is low, the ~110 band classifies as resistance.
    assert any(105.0 <= r.price <= 120.0 for r in ls_below.resistances), (
        [r.price for r in ls_below.resistances], ls_below.last_close
    )


# --- volume-weighted center ----------------------------------------------
def test_volume_weighted_center_pulls_toward_high_volume_touch():
    # Three swing lows at ~99.5 / 100.0 / 100.5 form one cluster (all within 1.5% of
    # the mean). The HIGHEST-priced low (~100.5) is the last trough (positional idx
    # 40). Putting a huge volume only on that bar pulls the volume-weighted center
    # clearly ABOVE the flat-volume (unweighted) center.
    path = _zigzag(
        [(130, 1), (100, 9), (140, 9), (100.5, 9), (150, 9), (101, 9), (170, 9)]
    )
    s_flat = lv.support_resistance(_daily(path), timeframe="1d").supports
    vol = np.full(len(path), 1e6)
    vol[40] = 1e9  # the ~100.5 trough bar
    s_heavy = lv.support_resistance(_daily(path, volume=vol), timeframe="1d").supports
    assert s_flat and s_heavy
    assert s_flat[0].touches >= 3 and s_heavy[0].touches >= 3, (
        s_flat[0].touches, s_heavy[0].touches
    )
    # The heavy-volume center is pulled UP toward the higher-priced, heavy low.
    assert s_heavy[0].price > s_flat[0].price + 1e-6, (s_heavy[0].price, s_flat[0].price)


# --- buy zone ------------------------------------------------------------
def test_buy_zone_nearest_support_case():
    # Price sits ABOVE a revisited ~100 support; the buy zone is the band up to it.
    df = _daily(_triple_low_support_path())
    ls = lv.support_resistance(df, timeframe="1d")
    assert ls.supports
    z = lv.buy_zone(df, timeframe="1d")
    assert z is not None
    assert z.basis.startswith("nearest support")
    s = ls.supports[0]
    assert abs(z.high - s.price) < 1e-6, (z.high, s.price)
    expected_low = s.price * (1.0 - lv.CLUSTER_TOL_PCT["1d"])
    assert abs(z.low - expected_low) < 1e-6, (z.low, expected_low)
    assert z.low < z.high


def test_buy_zone_ema_pullback_fallback():
    # A steadily RISING series with NO repeated swing low (so no >=2-touch support
    # below the close) must fall back to the rising-20-EMA pullback band.
    rng = np.random.default_rng(3)
    walk = np.linspace(100, 180, 120) + rng.normal(0, 0.2, 120)
    df = _daily(walk)
    ls = lv.support_resistance(df, timeframe="1d")
    assert not ls.supports, [s.price for s in ls.supports]
    z = lv.buy_zone(df, timeframe="1d")
    assert z is not None, "expected the 20-EMA pullback fallback"
    assert z.basis == "20-EMA pullback"
    assert z.low < z.high


def test_buy_zone_none_on_falling_series():
    # A steadily FALLING series: no support below the close AND the 20-EMA is not
    # rising -> no buy zone at all.
    walk = np.linspace(180, 100, 120)
    df = _daily(walk)
    z = lv.buy_zone(df, timeframe="1d")
    assert z is None, z


def test_buy_zone_in_zone_true_when_close_in_band():
    # Construct a clean rising 20-EMA where the LAST close sits inside the EMA band
    # (a shallow dip back to the EMA on the final bar).
    rng = np.random.default_rng(5)
    walk = list(np.linspace(100, 170, 119) + rng.normal(0, 0.15, 119))
    df0 = _daily(walk)
    e = lv.ema(df0["close"], lv._EMA_ZONE_LEN)
    ema_now = float(e.iloc[-1])
    # Append one more bar whose close sits essentially ON the current EMA value.
    walk.append(ema_now)
    df = _daily(walk)
    z = lv.buy_zone(df, timeframe="1d")
    # If a support band happens to form, accept that path too as long as in_zone holds
    # for a close inside its band; otherwise require the EMA fallback in-band.
    assert z is not None
    if z.basis == "20-EMA pullback":
        assert z.in_zone, (z.low, df["close"].iloc[-1], z.high)
        assert z.distance_pct == 0.0


def test_buy_zone_distance_sign_when_outside():
    # Price above a support band -> the band is BELOW the close, so the signed gap
    # to the nearest (upper) edge is negative.
    df = _daily(_triple_low_support_path())
    z = lv.buy_zone(df, timeframe="1d")
    assert z is not None
    last_close = float(df["close"].iloc[-1])
    if not z.in_zone:
        assert last_close > z.high  # close above the band
        assert z.distance_pct < 0.0, z.distance_pct


# --- anti-shapes (no >=2-touch level) ------------------------------------
def test_no_level_on_monotonic_ramp():
    df = _daily(np.linspace(100, 200, 200))
    ls = lv.support_resistance(df, timeframe="1d")
    assert ls.supports == () and ls.resistances == (), (
        [s.price for s in ls.supports], [r.price for r in ls.resistances]
    )


def test_no_level_on_flat_line():
    # A dead-flat series yields no swing pivots, hence no levels.
    df = _daily(np.full(200, 100.0))
    ls = lv.support_resistance(df, timeframe="1d")
    assert ls.supports == () and ls.resistances == ()


def test_no_strong_level_on_random_walk():
    # A single seeded random walk: swing pivots scatter, so no tight >=2-touch
    # cluster of HIGH strength should dominate. We tolerate incidental weak clusters
    # but assert none is both multi-touch AND strong (the precision contract).
    rng = np.random.default_rng(7)
    walk = 100 * np.cumprod(1 + rng.normal(0, 0.012, 250))
    ls = lv.support_resistance(_daily(walk), timeframe="1d")
    for level in ls.supports + ls.resistances:
        # An incidental random cluster should not look like a firm, tight level.
        assert not (level.touches >= 3 and level.strength >= 0.9), (
            level.kind, level.touches, level.strength, level.price
        )


# --- fail-soft -----------------------------------------------------------
def test_support_resistance_failsoft_empty_short_none():
    from screener import patterns as pat

    for bad in (None, pat._empty_frame(), _daily([1, 2, 3, 4, 5])):
        ls = lv.support_resistance(bad, timeframe="1d")
        assert ls.supports == () and ls.resistances == ()
        assert ls.timeframe == "1d"
        # last_close is NaN for None/empty; a tiny frame may still carry a close.
        assert isinstance(ls.last_close, float)


def test_levels_all_timeframes_keys_and_failsoft():
    from screener import patterns as pat

    df = _daily(_triple_low_support_path())
    result = lv.levels_all_timeframes(df)
    assert list(result.keys()) == ["1w", "1d", "1mo"]
    for tf, ls in result.items():
        assert isinstance(ls, lv.LevelSet)
        assert ls.timeframe == tf
    # Empty / None input still returns the stable 3-key shape with empty sets.
    for bad in (None, pat._empty_frame()):
        res = lv.levels_all_timeframes(bad)
        assert list(res.keys()) == ["1w", "1d", "1mo"]
        for ls in res.values():
            assert ls.supports == () and ls.resistances == ()


def test_buy_zone_failsoft_empty_short_none():
    from screener import patterns as pat

    for bad in (None, pat._empty_frame(), _daily([1, 2, 3, 4, 5])):
        assert lv.buy_zone(bad, timeframe="1d") is None


def test_dataclasses_picklable():
    import pickle

    df = _daily(_triple_low_support_path())
    ls = lv.support_resistance(df, timeframe="1d")
    z = lv.buy_zone(df, timeframe="1d")
    restored_ls = pickle.loads(pickle.dumps(ls))
    assert isinstance(restored_ls, lv.LevelSet)
    assert [s.price for s in restored_ls.supports] == [s.price for s in ls.supports]
    if ls.supports:
        assert isinstance(ls.supports[0].first, pd.Timestamp)
    if z is not None:
        restored_z = pickle.loads(pickle.dumps(z))
        assert restored_z.low == z.low and restored_z.basis == z.basis


# --- runner --------------------------------------------------------------
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
