"""Synthetic, framework-free tests for the pure display layer (M5).

Covers every helper in :mod:`screener.display` — the universe-size gate, the
empty/state checks, the four-filter pipeline, table-column selection + the pure
column-config descriptor, the ``reasons`` -> tidy-frame builder, value
formatting, the earnings badge/summary helpers, selection reconciliation, and
all context/empty/disclaimer strings.

Like ``tests/test_engine.py`` and ``tests/test_indicators.py``: NO ``pytest``,
NO ``yfinance``, NO ``streamlit`` import — every test is a plain ``test_*``
function using ``assert`` + ``math.isclose`` so the suite runs standalone as
``python tests/test_display.py`` (the ``__main__`` runner counts pass/fail,
prints a summary, and exits non-zero on any failure). All inputs are hand-built
result frames / ``reasons`` OrderedDicts, plus one optional end-to-end check
that drives the REAL engine through a tiny offline ``FakeProvider`` (mirroring
``tests/test_engine.py``) to prove the pure pipeline consumes the genuine engine
schema — no network anywhere.
"""

import datetime as dt
import math
import os
import sys
from collections import OrderedDict

# Put the repo root on sys.path so "import screener" resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from screener import display  # noqa: E402
from screener.profiles import Filter, Profile, SignalSpec, get_profile  # noqa: E402


# --- hand-built fixtures -------------------------------------------------
def _swing_profile() -> Profile:
    """The real swing profile (flags include earnings_in_window)."""
    return get_profile("swing")


def _momentum_profile() -> Profile:
    return get_profile("momentum")


def _long_term_profile() -> Profile:
    return get_profile("long_term")


def _result_frame(symbols=("AAPL", "MSFT", "NVDA"), *, swing=False):
    """A small, well-formed result frame mirroring the engine's schema.

    Includes the lead columns, a couple of raw signal feature columns for each
    profile family, a ``reasons`` column, and (when ``swing``) the
    ``earnings_in_window`` (bool) + ``days_to_earnings`` (float64 with NaN)
    columns the engine emits for swing.
    """
    n = len(symbols)
    data = {
        "symbol": list(symbols),
        "name": [f"{s} Inc" for s in symbols],
        "sector": ["Technology", "Technology", "Energy"][:n],
        "score": [0.80, 0.55, 0.30][:n],
        "rank": list(range(1, n + 1)),
        # a percent-style feature and a couple raw ratios for table tests
        "momentum_3m": [0.18, 0.05, -0.02][:n],
        "momentum_12m": [0.95, 0.40, 0.10][:n],
        "rel_volume_20": [3.1, 2.4, 5.0][:n],
        "forward_pe": [18.0, 25.0, 30.0][:n],
        "reasons": [OrderedDict() for _ in range(n)],
    }
    if swing:
        data["ema_5_9_cross_score"] = [0.9, 0.5, 0.1][:n]
        data["macd_hist"] = [0.12, -0.03, 0.04][:n]
        data["rsi_health"] = [0.8, 0.6, 0.4][:n]
        data["pullback_quality"] = [0.98, 0.50, 0.20][:n]
        data["sector_strength_score"] = [1.0, 1.0, 0.5][:n]
        data["earnings_in_window"] = [True, False, False][:n]
        data["days_to_earnings"] = [3.0, float("nan"), float("nan")][:n]
    return pd.DataFrame(data)


def _reasons(profile: Profile, *, pcts=None, contribs=None, values=None, with_flags=False):
    """A reasons OrderedDict in the profile's signal order (+ optional flags)."""
    sigs = [s.feature for s in profile.signals]
    pcts = pcts or {f: 0.5 for f in sigs}
    contribs = contribs or {f: 0.1 for f in sigs}
    values = values or {f: 0.1 for f in sigs}
    od = OrderedDict()
    for f in sigs:
        od[f] = {
            "value": values.get(f),
            "percentile": pcts.get(f),
            "contribution": contribs.get(f),
        }
    if with_flags:
        od["flags"] = {"earnings_in_window": True}
    return od


# =========================================================================
# take_universe_slice / universe_size_hint
# =========================================================================
def test_take_universe_slice_clamps():
    uni = pd.DataFrame({"symbol": [f"S{i}" for i in range(10)]})
    assert len(display.take_universe_slice(uni, 100)) == 10   # n > len -> full
    assert len(display.take_universe_slice(uni, 0)) == 1      # n < 1 -> one row
    assert len(display.take_universe_slice(uni, -5)) == 1     # negative -> one row
    sl = display.take_universe_slice(uni, 4)
    assert list(sl["symbol"]) == ["S0", "S1", "S2", "S3"]     # head(n)


def test_universe_size_hint_mentions_count():
    s = display.universe_size_hint(25)
    assert "25" in s and len(s) > 0
    assert "cold" in s.lower() or "min" in s.lower()


# =========================================================================
# is_empty_result / sector_options
# =========================================================================
def test_is_empty_result():
    empty = pd.DataFrame(columns=["symbol", "score"])
    assert display.is_empty_result(empty) is True
    assert display.is_empty_result(_result_frame()) is False


def test_sector_options_sorted_unique():
    df = pd.DataFrame({"sector": ["Energy", "Tech", "Energy", None, "Tech", ""]})
    assert display.sector_options(df) == ["Energy", "Tech"]
    # Missing column -> [].
    assert display.sector_options(pd.DataFrame({"x": [1]})) == []
    # All-null -> [].
    assert display.sector_options(pd.DataFrame({"sector": [None, None]})) == []


# =========================================================================
# apply_filters
# =========================================================================
def test_apply_filters_text_substring_case_insensitive():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA"],
            "name": ["Apple Inc", "Microsoft", float("nan")],
            "score": [0.8, 0.5, 0.3],
        }
    )
    p = _momentum_profile()
    out = display.apply_filters(df, text="aap", sectors=[], min_score=0.0, earnings_only=False, profile=p)
    assert list(out["symbol"]) == ["AAPL"]
    # Empty text passes all (NaN name tolerated, no crash).
    out_all = display.apply_filters(df, text="", sectors=[], min_score=0.0, earnings_only=False, profile=p)
    assert len(out_all) == 3
    # Matching on a NaN name simply doesn't match, but doesn't raise.
    out_nv = display.apply_filters(df, text="nvda", sectors=[], min_score=0.0, earnings_only=False, profile=p)
    assert list(out_nv["symbol"]) == ["NVDA"]


def test_apply_filters_sector_and_min_score():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "name": ["a", "b", "c", "d"],
            "sector": ["Tech", "Energy", "Tech", "Health"],
            "score": [0.9, 0.4, float("nan"), 0.2],
        }
    )
    p = _momentum_profile()
    # Sector membership AND score floor compose.
    out = display.apply_filters(df, text="", sectors=["Tech"], min_score=0.5, earnings_only=False, profile=p)
    assert list(out["symbol"]) == ["A"]  # C is Tech but NaN score dropped at floor>0
    # Empty sectors = all sectors.
    out_all = display.apply_filters(df, text="", sectors=[], min_score=0.0, earnings_only=False, profile=p)
    assert len(out_all) == 4
    # floor==0 keeps the NaN-score row.
    assert "C" in set(out_all["symbol"])


def test_apply_filters_swing_earnings_only_gated():
    swing_df = _result_frame(swing=True)
    swing = _swing_profile()
    # Swing profile + earnings_only -> only in-window rows.
    out = display.apply_filters(swing_df, text="", sectors=[], min_score=0.0, earnings_only=True, profile=swing)
    assert list(out["symbol"]) == ["AAPL"]
    # Non-swing profile ignores earnings_only even if the column exists.
    mom = _momentum_profile()
    out_mom = display.apply_filters(swing_df, text="", sectors=[], min_score=0.0, earnings_only=True, profile=mom)
    assert len(out_mom) == 3  # unfiltered by earnings


def test_apply_filters_resets_index():
    df = _result_frame(swing=True)
    out = display.apply_filters(df, text="", sectors=[], min_score=0.0, earnings_only=True, profile=_swing_profile())
    # Even though we kept only the first row, the index is a fresh RangeIndex.
    assert list(out.index) == list(range(len(out)))
    # And the input frame is untouched (a copy was returned).
    assert list(df.index) == [0, 1, 2]


# =========================================================================
# table_view / column_order / column_config_spec
# =========================================================================
def test_table_view_columns_per_profile():
    # long_term
    lt = _long_term_profile()
    lt_df = _result_frame()
    lt_view = display.table_view(lt_df, lt)
    assert list(lt_view.columns)[:5] == ["rank", "symbol", "name", "sector", "score"]
    assert "reasons" not in lt_view.columns
    assert not any(c.endswith("_pct") for c in lt_view.columns)

    # momentum
    mom = _momentum_profile()
    mom_view = display.table_view(_result_frame(), mom)
    assert "momentum_3m" in mom_view.columns
    assert "reasons" not in mom_view.columns

    # swing: earnings columns appended; reasons/pct excluded
    sw = _swing_profile()
    sw_view = display.table_view(_result_frame(swing=True), sw)
    assert "earnings_in_window" in sw_view.columns
    assert "days_to_earnings" in sw_view.columns
    assert "reasons" not in sw_view.columns
    assert not any(c.endswith("_pct") for c in sw_view.columns)


def test_table_view_missing_column_failsoft():
    # A result frame missing one of the profile's signal columns must not raise.
    df = _result_frame()  # has momentum_3m but not all momentum signals
    mom = _momentum_profile()
    view = display.table_view(df, mom)
    # Only the intersection of signal columns appears; lead cols always present.
    assert "momentum_3m" in view.columns
    assert "momentum_6m" not in view.columns  # absent from the fixture
    assert "rank" in view.columns


def test_column_order_excludes_earnings_for_non_swing():
    # Even if earnings columns exist on the frame, a non-swing profile excludes them.
    df = _result_frame(swing=True)
    mom = _momentum_profile()
    order = display.column_order(mom, df)
    assert "earnings_in_window" not in order
    assert "days_to_earnings" not in order


def test_column_config_spec_kinds():
    sw = _swing_profile()
    spec = display.column_config_spec(sw)
    assert spec["score"]["kind"] == "progress"
    assert math.isclose(spec["score"]["min"], 0.0) and math.isclose(spec["score"]["max"], 1.0)
    assert spec["rank"]["kind"] == "number"
    assert spec["rel_volume_20"]["kind"] == "number" and spec["rel_volume_20"]["format"] == "%.2f"
    # A derived [0,1] score -> progress.
    assert spec["pullback_quality"]["kind"] == "progress"
    # Swing earnings columns.
    assert spec["earnings_in_window"]["kind"] == "checkbox"
    assert spec["days_to_earnings"]["kind"] == "number"
    # A percent-style feature on momentum.
    mom_spec = display.column_config_spec(_momentum_profile())
    assert mom_spec["momentum_3m"]["kind"] == "percent"
    # The percent descriptor must carry the "percent" PRESET, not a printf
    # "%.1f%%" — the preset multiplies the engine fraction ×100 for display,
    # a printf string would render the raw 0.12 and drop the scaling.
    assert mom_spec["momentum_3m"]["format"] == "percent"
    # The spec is a plain dict of plain dicts (no streamlit objects).
    assert isinstance(spec, dict)
    for v in spec.values():
        assert isinstance(v, dict) and "kind" in v


# =========================================================================
# reasons_to_frame / max_contribution / contribution_caption
# =========================================================================
def test_reasons_to_frame_order_and_columns():
    sw = _swing_profile()
    reasons = _reasons(sw, with_flags=True)
    frame = display.reasons_to_frame(reasons, sw)
    assert list(frame.columns) == ["Signal", "Value", "Percentile", "Contribution"]
    # One row per signal, flags excluded.
    assert len(frame) == len(sw.signals)
    # Order preserved (the profile's signal order via humanized labels).
    expected = [display.feature_label(s.feature) for s in sw.signals]
    assert list(frame["Signal"]) == expected


def test_reasons_to_frame_nan_tolerant():
    p = Profile("t", "T", signals=(SignalSpec("momentum_3m", 1.0, "higher"),))
    reasons = OrderedDict()
    reasons["momentum_3m"] = {"value": float("nan"), "percentile": None, "contribution": float("nan")}
    frame = display.reasons_to_frame(reasons, p)
    assert frame.iloc[0]["Value"] == "—"
    assert math.isnan(frame.iloc[0]["Percentile"])
    # Empty / None reasons -> empty frame with the right columns.
    empty = display.reasons_to_frame(OrderedDict(), p)
    assert len(empty) == 0
    assert list(empty.columns) == ["Signal", "Value", "Percentile", "Contribution"]
    assert len(display.reasons_to_frame(None, p)) == 0


def test_contribution_caption_sums_to_score():
    p = Profile(
        "t", "T",
        signals=(SignalSpec("a", 1.0, "higher"), SignalSpec("b", 1.0, "higher")),
    )
    reasons = OrderedDict()
    reasons["a"] = {"value": 1.0, "percentile": 1.0, "contribution": 0.5}
    reasons["b"] = {"value": 1.0, "percentile": 0.5, "contribution": 0.21875}
    reasons["flags"] = {"x": True}  # must be ignored in the sum
    score = 0.71875
    cap = display.contribution_caption(reasons, score)
    # The summed contributions equal the score.
    summed = 0.5 + 0.21875
    assert math.isclose(summed, score)
    assert "0.719" in cap  # both sides round to 0.719


def test_max_contribution_positive():
    p = Profile("t", "T", signals=(SignalSpec("a", 1.0), SignalSpec("b", 1.0)))
    reasons = OrderedDict()
    reasons["a"] = {"value": 1.0, "percentile": 1.0, "contribution": 0.3}
    reasons["b"] = {"value": 1.0, "percentile": 1.0, "contribution": 0.1}
    assert math.isclose(display.max_contribution(reasons), 0.3)
    # All-zero contributions still clamp strictly > 0.
    zero = OrderedDict()
    zero["a"] = {"value": 0, "percentile": 0, "contribution": 0.0}
    assert display.max_contribution(zero) > 0.0
    # Empty reasons -> still > 0.
    assert display.max_contribution(OrderedDict()) > 0.0


# =========================================================================
# format_value
# =========================================================================
def test_format_value_scales():
    assert display.format_value("momentum_3m", 0.12) == "12.0%"
    assert display.format_value("dist_52w_high", -0.05) == "-5.0%"
    assert display.format_value("forward_pe", 18.0) == "18.0"
    assert display.format_value("rel_volume_20", 5.0) == "5.00×"
    assert display.format_value("rsi_14", 100.0) == "100"
    assert display.format_value("pullback_quality", 0.98) == "0.98"
    assert display.format_value("sma_stacked_20_50_150", True) == "Yes"
    assert display.format_value("sma_stacked_20_50_150", False) == "No"
    assert display.format_value("momentum_3m", float("nan")) == "—"
    assert display.format_value("forward_pe", None) == "—"
    assert display.format_value("sma_stacked_20_50_150", None) == "—"


# =========================================================================
# earnings_badge / earnings_badge_series / earnings_summary
# =========================================================================
def test_earnings_badge_scalar():
    assert display.earnings_badge(True, 3.0) == "⚠ Earnings in 3d"
    assert display.earnings_badge(False, 8.0) == ""
    assert display.earnings_badge(True, float("nan")) == ""   # no int(NaN) crash
    assert display.earnings_badge(False, None) == ""
    assert display.earnings_badge(True, None) == ""
    # numpy types tolerated
    assert display.earnings_badge(np.bool_(True), np.float64(5.0)) == "⚠ Earnings in 5d"


def test_earnings_badge_series_and_summary():
    swing_df = _result_frame(swing=True)
    series = display.earnings_badge_series(swing_df)
    assert list(series) == ["⚠ Earnings in 3d", "", ""]
    # Columns absent -> empty-string series of the right length.
    no_earn = _result_frame()  # momentum-style, no earnings columns
    s2 = display.earnings_badge_series(no_earn)
    assert list(s2) == ["", "", ""]
    # Summary counts the in-window names.
    summ = display.earnings_summary(swing_df)
    assert summ is not None and "1 of 3" in summ
    # No in-window rows -> None.
    none_df = _result_frame(swing=True)
    none_df["earnings_in_window"] = [False, False, False]
    assert display.earnings_summary(none_df) is None
    # Column absent -> None.
    assert display.earnings_summary(no_earn) is None


# =========================================================================
# resolve_selection / row_option_label
# =========================================================================
def test_row_option_label():
    view = _result_frame()
    assert display.row_option_label(view, "AAPL") == "AAPL — AAPL Inc"
    # Unknown symbol falls back to the symbol string.
    assert display.row_option_label(view, "ZZZZ") == "ZZZZ"


def test_resolve_selection_precedence():
    view = _result_frame()  # AAPL, MSFT, NVDA
    # Fresh table click wins.
    assert display.resolve_selection(view, "MSFT", "NVDA", "AAPL") == "MSFT"
    # No click -> selectbox value.
    assert display.resolve_selection(view, None, "NVDA", "AAPL") == "NVDA"
    # No click, no selectbox -> prev symbol if still present.
    assert display.resolve_selection(view, None, None, "AAPL") == "AAPL"
    # Prev symbol no longer in view -> default to rank-1 row.
    assert display.resolve_selection(view, None, None, "GONE") == "AAPL"
    # A stale click (not in view) is ignored, falls through.
    assert display.resolve_selection(view, "GONE", "MSFT", None) == "MSFT"
    # One-row view never raises.
    one = _result_frame(symbols=("AAPL",))
    assert display.resolve_selection(one, None, None, None) == "AAPL"


# =========================================================================
# messages / context / disclaimer constants
# =========================================================================
def test_messages_and_context_line():
    assert display.filter_summary(4, 7) == "Showing 4 of 7 matches"
    ctx = display.scan_context_line("Swing", 25, "2026-06-20", 4)
    for token in ("Swing", "25", "2026-06-20", "4"):
        assert token in ctx
    em = display.empty_message("Swing", 25)
    assert "Swing" in em and "25" in em
    fe = display.filtered_empty_message(7)
    assert "7" in fe
    assert isinstance(display.DISCLAIMER_TEXT, str) and len(display.DISCLAIMER_TEXT) > 0
    assert isinstance(display.DISCLAIMER_DETAIL, str) and len(display.DISCLAIMER_DETAIL) > 0


def test_feature_labels_fallback():
    # Unknown feature -> title-cased fallback, never a KeyError.
    assert display.feature_label("some_new_feature") == "Some New Feature"
    # Every scored feature across the three profiles has an explicit label.
    scored = set()
    for name in ("long_term", "swing", "momentum"):
        for s in get_profile(name).signals:
            scored.add(s.feature)
    for feat in scored:
        assert feat in display.FEATURE_LABELS, f"{feat} missing an explicit label"


# =========================================================================
# OPTIONAL end-to-end: real engine via a tiny offline FakeProvider
# =========================================================================
def _make_frame(n=300, *, start=50.0, drift=0.30, noise=0.05, vol_spike=False,
                base_volume=1_000_000.0, falling=False, seed=0):
    """A canonical ~300-bar OHLCV frame (mirrors tests/test_engine.py)."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    idx.name = "date"
    rng = np.random.default_rng(seed)
    if falling:
        close = float(start) + np.arange(n) * (-abs(drift)) + rng.normal(0, noise, n)
        close = np.clip(close, 1.0, None)
    else:
        close = float(start) + np.arange(n) * abs(drift) + rng.normal(0, noise, n)
    volume = np.full(n, float(base_volume))
    if vol_spike:
        volume[-1] = base_volume * 5.0
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": volume},
        index=idx,
    )


def test_end_to_end_with_fakeprovider_optional():
    """Drive the REAL swing engine offline, then run the pure pipeline on it.

    Proves the display helpers consume the genuine engine schema (reasons order,
    earnings columns, score/rank dtypes) — not just the hand-built fixtures.
    Skips silently if the engine/provider imports are unavailable.
    """
    try:
        from screener import engine as eng
        from screener.provider import DataProvider, Fundamentals
    except Exception:  # pragma: no cover - engine always present in this repo
        return

    class FakeProvider(DataProvider):
        def __init__(self, frames, funds, earn):
            self._frames, self._funds, self._earn = frames, funds, earn

        def price_history(self, symbol, *, lookback_days=730):
            k = symbol.strip().upper()
            return self._frames.get(
                k,
                pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], name="date")),
            )

        def fundamentals(self, symbol):
            k = symbol.strip().upper()
            return self._funds.get(k, Fundamentals(symbol=k))

        def earnings_date(self, symbol):
            return self._earn.get(symbol.strip().upper())

    as_of = dt.date(2026, 6, 20)
    rows = [
        ("AAA", "Alpha", "Tech"), ("BBB", "Bravo", "Tech"),
        ("CCC", "Char", "Energy"), ("DDD", "Delta", "Energy"),
        ("EEE", "Echo", "Health"), ("FFF", "Fox", "Utilities"),
    ]
    frames = {
        "AAA": _make_frame(start=40, drift=0.35, vol_spike=True, seed=10),
        "BBB": _make_frame(start=60, drift=0.30, vol_spike=True, seed=11),
        "CCC": _make_frame(start=30, drift=0.28, vol_spike=True, seed=12),
        "DDD": _make_frame(start=80, drift=0.25, vol_spike=True, seed=13),
        "EEE": _make_frame(start=50, drift=0.05, falling=True, seed=14),
        "FFF": _make_frame(start=50, drift=0.05, falling=True, seed=15),
    }
    funds = {
        sym: Fundamentals(symbol=sym, name=name, sector=sec, market_cap=5e11,
                          forward_pe=15.0 + i, trailing_pe=20.0 + i,
                          revenue_growth=0.10 + i * 0.01, earnings_growth=0.12 + i * 0.01)
        for i, (sym, name, sec) in enumerate(rows)
    }
    earn = {"AAA": dt.date(2026, 6, 23)}
    universe = pd.DataFrame(rows, columns=["symbol", "name", "sector"])
    provider = FakeProvider(frames, funds, earn)

    swing = get_profile("swing")
    result = eng.run_screen(swing, universe, provider, as_of=as_of)
    assert len(result) >= 1

    # Pure pipeline consumes the genuine frame end to end.
    view = display.apply_filters(result, text="", sectors=[], min_score=0.0,
                                 earnings_only=False, profile=swing)
    assert list(view.index) == list(range(len(view)))  # index reset

    table = display.table_view(view, swing)
    assert "reasons" not in table.columns
    assert "earnings_in_window" in table.columns  # swing
    assert not any(c.endswith("_pct") for c in table.columns)

    # The first row's genuine reasons -> tidy frame; contributions sum to score.
    row = result.iloc[0]
    reasons = row["reasons"]
    rf = display.reasons_to_frame(reasons, swing)
    assert list(rf["Signal"]) == [display.feature_label(s.feature) for s in swing.signals]
    cap = display.contribution_caption(reasons, row["score"])
    summed = sum(v["contribution"] for k, v in reasons.items() if k != "flags")
    assert math.isclose(summed, float(row["score"]))
    assert f"{float(row['score']):.3f}" in cap

    # Earnings helpers see the real columns.
    series = display.earnings_badge_series(result)
    assert len(series) == len(result)
    summ = display.earnings_summary(result)
    assert summ is None or "report earnings" in summ


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
