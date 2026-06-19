"""Synthetic, network-free EDGE-CASE tests (Milestone 6 validation + polish).

Pins the screener's fail-soft behavior on the awkward inputs the M6 edge audit
enumerated — missing data, thin/zero volume, a recent IPO with too little
history, an all-missing signal column — and, crucially, REGRESSION-PINS the one
documented quality wart: a recent IPO scores the top percentile on
``dist_52w_high`` (its ``min_periods=1`` exception turns thin history into a
confident best-case value), so it can out-rank an established name that is genuinely
off its 52-week high. See README "Known limitations" and the
:func:`screener.indicators.distance_from_high` docstring.

House style mirrors ``tests/test_engine.py`` and ``tests/test_indicators.py``: NO
``pytest`` / ``yfinance`` / ``streamlit`` imports — every test is a plain ``test_*``
function using ``assert`` + ``math.isclose`` so the suite runs BOTH under
``python -m pytest tests/test_edge_cases.py`` AND standalone as
``python tests/test_edge_cases.py`` (the ``__main__`` runner counts pass/fail,
prints a summary, and exits non-zero on any failure). All inputs are synthetic and
share ONE index per frame (see ``test_synthetic_frame_index_trap`` for why that
matters).
"""

import datetime as dt
import math
import os
import sys

# Put the repo root on sys.path so "import screener" resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from screener import engine as eng  # noqa: E402
from screener import indicators as ind  # noqa: E402
from screener import profiles as prof  # noqa: E402
from screener.profiles import Filter, Profile, SignalSpec  # noqa: E402
from screener.provider import DataProvider, Fundamentals  # noqa: E402

# A fixed clock so any earnings-window logic is deterministic.
AS_OF = dt.date(2026, 6, 20)


# --- helpers -------------------------------------------------------------
def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _idx(n, *, start="2025-01-01"):
    """A daily, oldest-first, tz-naive 'date' index of length ``n``."""
    idx = pd.date_range(start, periods=n, freq="D")
    idx.name = "date"
    return idx


def _frame(close, *, volume=None, index=None):
    """A canonical OHLCV frame from a close list/array, all columns on ONE index.

    ``volume`` defaults to a steady 1e6. Passing a list builds it on the same
    index as ``close`` — never mix a free-floating Series in (see the index-trap
    test); that is exactly the bug this helper exists to avoid.
    """
    close = np.asarray(close, dtype="float64")
    index = index if index is not None else _idx(len(close))
    if volume is None:
        volume = np.full(len(close), 1_000_000.0)
    else:
        volume = np.asarray(volume, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _empty_frame():
    """The canonical empty price frame a fail-soft provider returns for a bad ticker."""
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="date"),
    )


def _universe(rows):
    """A universe DataFrame[symbol, name, sector] from (sym, name, sector) tuples."""
    return pd.DataFrame(rows, columns=["symbol", "name", "sector"])


class FakeProvider(DataProvider):
    """Offline :class:`DataProvider`: canned frames / fundamentals / earnings.

    An unknown symbol yields the empty price frame, an (almost) empty Fundamentals,
    and ``None`` earnings — exercising the engine's fail-soft path with no network.
    """

    def __init__(self, frames=None, fundamentals=None, earnings=None):
        self._frames = dict(frames or {})
        self._fundamentals = dict(fundamentals or {})
        self._earnings = dict(earnings or {})

    def price_history(self, symbol, *, lookback_days=730):
        return self._frames.get(symbol.strip().upper(), _empty_frame())

    def fundamentals(self, symbol):
        sym = symbol.strip().upper()
        return self._fundamentals.get(sym, Fundamentals(symbol=sym))

    def earnings_date(self, symbol):
        return self._earnings.get(symbol.strip().upper())


def _rising(n, *, start, step):
    """A clean monotonically-rising close array (no noise) of length ``n``."""
    return start + np.arange(n) * step


# =========================================================================
# Recent IPO — the documented dist_52w_high tilt (REGRESSION-PINNED)
# =========================================================================
def test_ipo_dist_52w_high_is_top_value_at_fresh_high():
    # A 30-bar IPO at its available high reads dist_52w_high == 0.0 (the BEST
    # possible value) via min_periods=1 — even though it has far fewer bars than
    # the 252-session window. This is the root cause of the ranking tilt below.
    ipo = pd.Series(_rising(30, start=10.0, step=1.0), index=_idx(30))
    assert math.isclose(ind.distance_from_high(ipo, window=252), 0.0)


def test_ipo_long_window_signals_are_nan_but_dist_is_finite():
    # The ASYMMETRY that creates the tilt: on a short IPO frame, sma_150 and
    # momentum_12m correctly degrade to NaN (-> neutral 0.5 in scoring), but
    # dist_52w_high stays finite at its best-case 0.0. It is the lone short-history
    # signal that yields a confident top value instead of neutralizing.
    snap = ind.snapshot(_frame(_rising(30, start=10.0, step=1.0)))
    assert _isnan(snap["sma_150"])
    assert _isnan(snap["momentum_12m"])
    assert not _isnan(snap["dist_52w_high"])
    assert math.isclose(snap["dist_52w_high"], 0.0)


def test_ipo_outranks_established_name_on_dist_52w_high():
    # REGRESSION: pins CURRENT (documented) behavior. A thin-history IPO sitting at
    # its available high out-ranks an established name that is genuinely 5% off its
    # real 252-bar high, on a profile that ranks purely on dist_52w_high ("higher").
    # If a future change adds a min-bars guard to distance_from_high, THIS test is
    # the canary — update it (and the README "Known limitations") deliberately.
    ipo = pd.Series(_rising(30, start=10.0, step=1.0), index=_idx(30))

    # Established: ramps to a peak over 200 bars, then drifts 5% below it.
    est_close = np.concatenate([np.linspace(50.0, 100.0, 200), np.linspace(100.0, 95.0, 60)])
    est = pd.Series(est_close, index=_idx(260, start="2024-01-01"))
    assert math.isclose(ind.distance_from_high(est, window=252), -0.05, abs_tol=1e-9)

    provider = FakeProvider(frames={"IPO": _frame(ipo.values, index=ipo.index),
                                    "EST": _frame(est.values, index=est.index)})
    universe = _universe([("IPO", "Ipo Co", "Tech"), ("EST", "Est Co", "Tech")])

    # Isolate the effect: a no-filter profile ranking ONLY on dist_52w_high.
    p = Profile("dt", "DistTest", signals=(SignalSpec("dist_52w_high", 1.0, "higher"),))
    res = eng.run_screen(p, universe, provider, as_of=AS_OF)

    ranks = dict(zip(res["symbol"], res["rank"]))
    assert ranks["IPO"] == 1, "documented tilt: short IPO at its high ranks first"
    assert ranks["EST"] == 2
    ipo_score = float(res.loc[res["symbol"] == "IPO", "score"].iloc[0])
    est_score = float(res.loc[res["symbol"] == "EST", "score"].iloc[0])
    assert ipo_score > est_score


def test_ipo_does_not_crash_any_profile_and_keeps_lead_schema():
    # The whole point of the audit: even a degenerate universe (one IPO + one bad
    # empty-frame ticker) runs all three real profiles WITHOUT raising and always
    # returns the stable lead schema.
    ipo = _frame(_rising(30, start=10.0, step=1.0))
    provider = FakeProvider(frames={"IPO": ipo})  # "BAD" absent -> empty frame
    universe = _universe([("IPO", "Ipo Co", "Tech"), ("BAD", "Bad Co", "Energy")])
    for name in ("long_term", "swing", "momentum"):
        res = eng.run_screen(name, universe, provider, as_of=AS_OF)
        for col in ("symbol", "name", "sector", "score", "rank"):
            assert col in res.columns, f"{name}: missing lead column {col!r}"


# =========================================================================
# Thin / ZERO volume — rel_volume_20 is NaN ("undefined"), never +inf
# =========================================================================
def test_zero_volume_rel_volume_is_nan_not_inf_end_to_end():
    # A name whose prior-20 volume window is entirely zero must read rel_volume_20
    # as NaN (a 0 baseline), NEVER +inf — otherwise it would sort straight to the
    # top of any rel-volume ranker. Verified through assemble_features, not just the
    # bare indicator.
    n = 60
    volume = np.concatenate([np.zeros(n - 1), [500.0]])  # zero baseline, late print
    provider = FakeProvider(frames={"ZED": _frame(_rising(n, start=50.0, step=0.3), volume=volume)})
    feats = eng.assemble_features(_universe([("ZED", "Zed Co", "Tech")]), provider, as_of=AS_OF)
    rv = feats.loc["ZED", "rel_volume_20"]
    assert _isnan(rv)
    assert not np.isposinf(rv)


def test_zero_volume_fails_swing_volume_filter_closed():
    # rel_volume_20 = NaN must FAIL the swing `rel_volume_20 > 2.0` filter (closed),
    # so a zero-volume name is excluded ON THE VOLUME FILTER — not incidentally.
    # Use enough bars (> the 3-month momentum window) that the lone name DOES
    # qualify for its sector's top-3 "leading" cut, so rel_volume_20 is the ONLY
    # filter that can exclude it. (With too few bars momentum_3m is NaN, the
    # leading-sector filter drops it for an unrelated reason, and an inf-ratio
    # regression would slip through this test unnoticed.)
    n = 90
    volume = np.concatenate([np.zeros(n - 1), [500.0]])
    provider = FakeProvider(frames={"ZED": _frame(_rising(n, start=50.0, step=0.3), volume=volume)})
    feats = eng.assemble_features(_universe([("ZED", "Zed Co", "Tech")]), provider, as_of=AS_OF)
    feats = eng.compute_sector_strength(feats)
    # Preconditions that make the exclusion attributable to rel_volume_20 alone:
    assert bool(feats.loc["ZED", "in_leading_sector"]) is True
    assert _isnan(feats.loc["ZED", "rel_volume_20"])
    out = eng.apply_filters(feats, prof.get_profile("swing"))
    assert "ZED" not in set(out.index)


def test_thin_nonzero_volume_is_a_real_finite_ratio():
    # Genuinely thin (1-share) baseline is NOT an error: a real, finite, possibly
    # large ratio is correct-by-design (not guarded to NaN like the zero baseline).
    vol = pd.Series([1.0] * 25 + [50.0], index=_idx(26))
    out = ind.relative_volume_latest(vol, n=20)
    assert math.isfinite(out)
    assert math.isclose(out, 50.0)


# =========================================================================
# Missing data — all-NaN signal COLUMN across the whole universe
# =========================================================================
def test_all_nan_signal_column_neutralizes_to_half_no_skew():
    # When an ENTIRE signal column is NaN across the universe, every row gets the
    # neutral 0.5 for it (no crash, no skew); ranking falls to the other signals.
    df = pd.DataFrame(
        {"good": [1.0, 2.0, 3.0], "missing": [np.nan, np.nan, np.nan]},
        index=pd.Index(["A", "B", "C"], name="symbol"),
    )
    p = Profile(
        "t", "T",
        signals=(SignalSpec("good", 1.0, "higher"), SignalSpec("missing", 1.0, "higher")),
    )
    out = eng.score_and_rank(df, p)
    assert (out["missing_pct"] == 0.5).all()
    # Order is decided entirely by `good` (C top), since `missing` is flat.
    assert list(out.index) == ["C", "B", "A"]
    assert list(out["rank"]) == [1, 2, 3]


def test_signal_column_entirely_absent_is_neutral_not_keyerror():
    # A signal whose column does not exist at all must score neutral 0.5 for every
    # row (the engine substitutes an all-NaN series) rather than raising KeyError.
    df = pd.DataFrame({"present": [1.0, 2.0]}, index=pd.Index(["A", "B"], name="symbol"))
    p = Profile(
        "t", "T",
        signals=(SignalSpec("present", 1.0, "higher"), SignalSpec("ghost", 1.0, "higher")),
    )
    out = eng.score_and_rank(df, p)
    assert "ghost_pct" in out.columns
    assert (out["ghost_pct"] == 0.5).all()
    # And the per-row reasons still carry the absent signal (value NaN).
    reasons_a = dict(out.loc["A", "reasons"])
    assert "ghost" in reasons_a
    assert _isnan(reasons_a["ghost"]["value"])


def test_scored_row_score_is_never_nan():
    # Even a row whose every signal is missing gets a finite score (fillna(0.5)
    # before the weighted sum + clip), so the score column is never NaN and the
    # descending sort is well-defined.
    df = pd.DataFrame(
        {"a": [np.nan, 1.0], "b": [np.nan, 2.0]},
        index=pd.Index(["EMPTY", "REAL"], name="symbol"),
    )
    p = Profile(
        "t", "T",
        signals=(SignalSpec("a", 1.0, "higher"), SignalSpec("b", 1.0, "higher")),
    )
    out = eng.score_and_rank(df, p)
    assert out["score"].notna().all()
    assert math.isclose(float(out.loc["EMPTY", "score"]), 0.5)
    assert ((out["score"] >= 0.0) & (out["score"] <= 1.0)).all()


# =========================================================================
# Bad / empty ticker — never aborts, scores neutral / fails closed
# =========================================================================
def test_empty_frame_ticker_does_not_abort_scan_with_a_good_name():
    # A universe mixing a clean rising name with a bad empty-frame ticker: the good
    # one survives momentum's price_above_sma_50 filter; the bad one fails closed.
    good = _frame(_rising(300, start=40.0, step=0.3))
    funds = {"GOOD": Fundamentals(symbol="GOOD", name="Good Co", sector="Tech", forward_pe=12.0)}
    provider = FakeProvider(frames={"GOOD": good}, fundamentals=funds)
    res = eng.run_screen("momentum", _universe([("GOOD", "Good Co", "Tech"),
                                                ("BAD", "Bad Co", "Tech")]),
                         provider, as_of=AS_OF)
    assert "GOOD" in set(res["symbol"])
    assert "BAD" not in set(res["symbol"])


def test_snapshot_on_empty_frame_is_all_safe_defaults():
    # An empty price frame yields every snapshot key, floats NaN, the price-above
    # flags None, the stack flag False, the cross event "none" — and never raises.
    snap = ind.snapshot(_empty_frame())
    assert _isnan(snap["momentum_1m"])
    assert _isnan(snap["dist_52w_high"])
    assert _isnan(snap["rsi_14"])
    assert snap["price_above_sma_50"] is None
    assert snap["sma_stacked_20_50_150"] is False
    assert snap["ema_5_9_event"] == "none"


# =========================================================================
# Test-authoring guard — the synthetic-index trap the M6 audit hit
# =========================================================================
def test_synthetic_frame_index_trap():
    # Documents (and protects against) the offline-suite trap noted in the M6 audit:
    # mixing a free-floating RangeIndex Series into a DatetimeIndex frame makes
    # pandas align by LABEL — it takes the UNION of the two indexes, so close and
    # volume never coexist on a single row and every row carries at least one NaN.
    # An indicator needing aligned close+volume then silently sees NaN, and a test
    # can "pass" on all-NaN data without exercising the math. The shared-index
    # helper in this file avoids it; this pins the failure mode so the lesson is
    # executable.
    index = _idx(30)
    close = pd.Series(_rising(30, start=10.0, step=1.0), index=index)
    bad_volume = pd.Series(np.full(30, 1_000_000.0))  # default RangeIndex 0..29 — WRONG

    trap = pd.DataFrame({"close": close, "volume": bad_volume})  # index UNION
    assert len(trap) == 60, "the trap: disjoint indexes union to twice the rows"
    # No row has BOTH a close and a volume — exactly why aligned indicators NaN out.
    assert not (trap["close"].notna() & trap["volume"].notna()).any()

    good = _frame(close.values, index=index)  # the right way: one shared index
    assert len(good) == 30
    assert good["close"].notna().all() and good["volume"].notna().all()
    assert math.isclose(good["close"].iloc[-1], close.iloc[-1])


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
