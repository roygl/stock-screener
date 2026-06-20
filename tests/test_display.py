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
from dataclasses import dataclass

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

    Includes the lead columns, the headline price columns (``price`` /
    ``change_pct``), a couple of raw signal feature columns for each profile family,
    the universe-wide tactical readouts (``extension_state`` / ``in_buy_zone``), a
    ``reasons`` column, and (when ``swing``) the ``earnings_in_window`` (bool) +
    ``days_to_earnings`` (float64 with NaN) columns the engine emits for swing.
    """
    n = len(symbols)
    data = {
        "symbol": list(symbols),
        "name": [f"{s} Inc" for s in symbols],
        "sector": ["Technology", "Technology", "Energy"][:n],
        "score": [0.80, 0.55, 0.30][:n],
        "rank": list(range(1, n + 1)),
        # headline price scalars (Stage 1): last close + signed daily change fraction
        "price": [195.0, 410.0, 120.0][:n],
        "change_pct": [0.012, -0.008, 0.031][:n],
        # a percent-style feature and a couple raw ratios for table tests
        "momentum_3m": [0.18, 0.05, -0.02][:n],
        "momentum_12m": [0.95, 0.40, 0.10][:n],
        "rel_volume_20": [3.1, 2.4, 5.0][:n],
        "forward_pe": [18.0, 25.0, 30.0][:n],
        # universe-wide tactical readouts (present for every profile's scan)
        "extension_state": ["normal", "extended", "parabolic"][:n],
        "in_buy_zone": [True, False, True][:n],
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


# --- tactical-readout fixtures (S/R, overextension, buy zone) -------------
# Lightweight stand-ins mirroring the screener.levels field contracts. The
# display formatters duck-type via getattr, so these exercise the exact contract
# (signed distance_pct fraction, 0..1 strength, int touches) without importing
# any heavier module — keeping the suite framework-free. One test below also
# builds the REAL levels.* objects to prove the formatters consume them.
@dataclass
class _FakeLevel:
    price: float
    kind: str
    touches: int
    strength: float
    distance_pct: float


@dataclass
class _FakeLevelSet:
    supports: tuple
    resistances: tuple


@dataclass
class _FakeZone:
    low: float
    high: float
    basis: str
    in_zone: bool
    distance_pct: float


def _level_set():
    """A LevelSet-like object with one resistance above and two supports below."""
    res = (_FakeLevel(160.0, "resistance", 3, 0.82, 0.073),)
    sup = (
        _FakeLevel(148.0, "support", 4, 0.91, -0.0067),
        _FakeLevel(140.0, "support", 2, 0.45, -0.0667),
    )
    return _FakeLevelSet(supports=sup, resistances=res)


def _tactical_frame(symbols=("AAA", "BBB", "CCC", "DDD")):
    """A result frame carrying the four universe-wide tactical-readout columns.

    ``extension_state`` spans normal/extended/parabolic; ``in_buy_zone`` mixes
    True/False/NaN; ``dist_to_buy_zone_pct`` is a signed fraction with a NaN.
    """
    n = len(symbols)
    return pd.DataFrame(
        {
            "symbol": list(symbols)[:n],
            "name": [f"{s} Inc" for s in symbols][:n],
            "sector": ["Technology", "Energy", "Health", "Technology"][:n],
            "score": [0.80, 0.55, 0.40, 0.30][:n],
            "rank": list(range(1, n + 1)),
            "extension_state": ["normal", "extended", "parabolic", "normal"][:n],
            "extension_score": [0.10, 0.45, 0.72, 0.05][:n],
            "in_buy_zone": [True, False, False, float("nan")][:n],
            "dist_to_buy_zone_pct": [0.0, 0.031, -0.012, float("nan")][:n],
        }
    )


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


def test_apply_filters_ticker_symbol_only():
    # The dedicated ticker box matches the SYMBOL only — never the company name.
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA"],
            "name": ["Apple Inc", "Microsoft", "NVIDIA"],
            "score": [0.8, 0.5, 0.3],
        }
    )
    p = _momentum_profile()
    # Case-insensitive substring on the symbol.
    out = display.apply_filters(df, text="", ticker="aap", sectors=[], min_score=0.0,
                                earnings_only=False, profile=p)
    assert list(out["symbol"]) == ["AAPL"]
    # "apple" would match the NAME via `text`, but NOT via `ticker` (symbol-only).
    out_name = display.apply_filters(df, text="", ticker="apple", sectors=[], min_score=0.0,
                                     earnings_only=False, profile=p)
    assert len(out_name) == 0
    # Empty ticker keeps everything.
    out_all = display.apply_filters(df, text="", ticker="", sectors=[], min_score=0.0,
                                    earnings_only=False, profile=p)
    assert len(out_all) == 3
    # ticker AND text compose (both must hold); index resets after the drop.
    out_both = display.apply_filters(df, text="nvidia", ticker="nv", sectors=[], min_score=0.0,
                                     earnings_only=False, profile=p)
    assert list(out_both["symbol"]) == ["NVDA"]
    assert list(out_both.index) == list(range(len(out_both)))


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


def test_apply_filters_extended_hidden_drops_only_parabolic():
    df = _tactical_frame()  # states: normal, extended, parabolic, normal
    mom = _momentum_profile()
    # Default (off) keeps every row, including the parabolic one.
    keep_all = display.apply_filters(df, text="", sectors=[], min_score=0.0,
                                     earnings_only=False, profile=mom)
    assert len(keep_all) == 4
    # On: only the "parabolic" row is dropped; "extended"/"normal" stay.
    out = display.apply_filters(df, text="", sectors=[], min_score=0.0,
                                earnings_only=False, profile=mom, extended_hidden=True)
    assert list(out["symbol"]) == ["AAA", "BBB", "DDD"]
    assert "parabolic" not in set(out["extension_state"])
    # Fresh RangeIndex after the drop.
    assert list(out.index) == list(range(len(out)))


def test_apply_filters_in_buy_zone_only():
    df = _tactical_frame()  # in_buy_zone: True, False, False, NaN
    mom = _momentum_profile()
    out = display.apply_filters(df, text="", sectors=[], min_score=0.0,
                                earnings_only=False, profile=mom, in_buy_zone_only=True)
    # Only the truthy in_buy_zone row survives; NaN is treated as False (dropped).
    assert list(out["symbol"]) == ["AAA"]
    # Off by default keeps all rows.
    out_off = display.apply_filters(df, text="", sectors=[], min_score=0.0,
                                    earnings_only=False, profile=mom)
    assert len(out_off) == 4


def test_apply_filters_tactical_flags_noop_when_columns_absent():
    # A frame WITHOUT the tactical columns must ignore both new flags (no KeyError).
    df = _result_frame().drop(columns=["extension_state", "in_buy_zone"])
    mom = _momentum_profile()
    out = display.apply_filters(df, text="", sectors=[], min_score=0.0,
                                earnings_only=False, profile=mom,
                                extended_hidden=True, in_buy_zone_only=True)
    assert len(out) == len(df)  # both flags are no-ops, all rows kept


def test_apply_filters_tactical_flags_compose_with_existing():
    # The new flags compose with sector + score filters and the index still resets.
    df = _tactical_frame()
    mom = _momentum_profile()
    out = display.apply_filters(df, text="", sectors=["Technology"], min_score=0.0,
                                earnings_only=False, profile=mom, extended_hidden=True)
    # Technology rows are AAA (normal) and DDD (normal); both survive extended_hidden.
    assert list(out["symbol"]) == ["AAA", "DDD"]
    assert list(out.index) == list(range(len(out)))


# =========================================================================
# table_view / column_order / column_config_spec
# =========================================================================
def test_table_view_columns_per_profile():
    # long_term
    lt = _long_term_profile()
    lt_df = _result_frame()
    lt_view = display.table_view(lt_df, lt)
    # `fit` (0..100) takes the visible score slot; the raw `score` column is dropped.
    assert list(lt_view.columns)[:5] == ["rank", "symbol", "name", "sector", "fit"]
    assert "score" not in lt_view.columns
    # The per-row narrative is the LAST column.
    assert lt_view.columns[-1] == "why"
    assert "reasons" not in lt_view.columns
    # No internal *_pct PERCENTILE columns leak (the headline change_pct is allowed).
    assert not any(c.endswith("_pct") for c in lt_view.columns if c != "change_pct")

    # Compact (default) is lean: NO per-profile signal columns, but the headline
    # price columns and the tactical readouts are present.
    assert "momentum_3m" not in lt_view.columns
    for col in ("price", "change_pct", "extension_state", "in_buy_zone"):
        assert col in lt_view.columns

    # momentum — signals appear only in the Detailed density.
    mom = _momentum_profile()
    assert "momentum_3m" not in display.table_view(_result_frame(), mom).columns
    mom_view = display.table_view(_result_frame(), mom, density="detailed")
    assert "momentum_3m" in mom_view.columns
    assert "reasons" not in mom_view.columns

    # swing: earnings columns appear in Detailed; reasons/pct excluded
    sw = _swing_profile()
    sw_view = display.table_view(_result_frame(swing=True), sw, density="detailed")
    assert "earnings_in_window" in sw_view.columns
    assert "days_to_earnings" in sw_view.columns
    assert "reasons" not in sw_view.columns
    assert not any(c.endswith("_pct") for c in sw_view.columns if c != "change_pct")


def test_table_view_missing_column_failsoft():
    # A result frame missing one of the profile's signal columns must not raise.
    df = _result_frame()  # has momentum_3m but not all momentum signals
    mom = _momentum_profile()
    view = display.table_view(df, mom, density="detailed")
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
# per-ticker external links (tradingview_url / yahoo_url + table threading)
# =========================================================================
def test_link_url_builders_and_class_shares():
    # Bare US large-cap symbols resolve directly on both sites.
    assert display.tradingview_url("AAPL") == "https://www.tradingview.com/chart/?symbol=AAPL"
    assert display.yahoo_url("AAPL") == "https://finance.yahoo.com/quote/AAPL"
    # Class shares: TradingView wants a dot, Yahoo keeps the dash.
    assert display.tradingview_url("BRK-B") == "https://www.tradingview.com/chart/?symbol=BRK.B"
    assert display.yahoo_url("BRK-B") == "https://finance.yahoo.com/quote/BRK-B"
    # Lower-case / whitespace input is canonicalised.
    assert display.tradingview_url(" brk-b ") == "https://www.tradingview.com/chart/?symbol=BRK.B"
    assert display.yahoo_url("aapl") == "https://finance.yahoo.com/quote/AAPL"


def test_column_order_inserts_fit_and_links_after_identity():
    mom = _momentum_profile()
    # Detailed shows signals; the prefix + why-last hold in BOTH densities.
    order = display.column_order(mom, _result_frame(), density="detailed")
    # `fit` takes the visible score slot, then the two link columns, then signals.
    assert order[:7] == ["rank", "symbol", "name", "sector", "fit", "tv_url", "yf_url"]
    assert order.index("tv_url") < order.index("momentum_3m")
    # The synthetic narrative column is LAST.
    assert order[-1] == "why"
    # Gate: a frame with no `symbol` column omits the links (never a KeyError)...
    no_sym = pd.DataFrame({"score": [0.5], "rank": [1]})
    assert "tv_url" not in display.column_order(mom, no_sym)
    # ...but `fit` is still present (it derives from `score`).
    assert "fit" in display.column_order(mom, no_sym)


def test_column_order_compact_vs_detailed_density():
    mom = _momentum_profile()
    df = _result_frame()
    compact = display.column_order(mom, df)                       # default = compact
    detailed = display.column_order(mom, df, density="detailed")
    # Shared prefix in both: identity, fit, links, then headline price columns.
    expected_prefix = ["rank", "symbol", "name", "sector", "fit", "tv_url", "yf_url",
                       "price", "change_pct"]
    assert compact[:9] == expected_prefix
    assert detailed[:9] == expected_prefix
    # Compact carries the tactical readouts but NO profile signals.
    assert "extension_state" in compact and "in_buy_zone" in compact
    assert "momentum_3m" not in compact
    # Detailed adds the signals (between price and the tactical block) and keeps
    # the tactical readouts; `why` is last in both.
    assert "momentum_3m" in detailed
    assert detailed.index("momentum_3m") < detailed.index("extension_state")
    assert "extension_state" in detailed and "in_buy_zone" in detailed
    assert compact[-1] == "why" and detailed[-1] == "why"


def test_all_profile_surfaces_market_cap_in_detailed():
    # The unfiltered "all" lens ranks by market cap; like any signal it shows only
    # in the Detailed density, and the config spec types/labels it.
    all_p = get_profile("all")
    df = _result_frame()
    df["market_cap"] = [3.0e12, 1.0e12, 5.0e11]
    compact = display.column_order(all_p, df)
    detailed = display.column_order(all_p, df, density="detailed")
    assert "market_cap" not in compact
    assert "market_cap" in detailed
    assert "market_cap" in display.table_view(df, all_p, density="detailed").columns
    spec = display.column_config_spec(all_p)
    assert spec["market_cap"]["label"] == "Market Cap"
    assert spec["market_cap"]["kind"] == "number"


def test_table_view_carries_link_columns():
    view = display.table_view(_result_frame(("AAPL", "BRK-B")), _momentum_profile())
    assert "tv_url" in view.columns and "yf_url" in view.columns
    # Real, non-null URLs per row, with the class-share transform applied.
    assert view.loc[view["symbol"] == "BRK-B", "tv_url"].iloc[0].endswith("symbol=BRK.B")
    assert view.loc[view["symbol"] == "BRK-B", "yf_url"].iloc[0].endswith("quote/BRK-B")
    assert view["tv_url"].notna().all() and view["yf_url"].notna().all()


def test_column_config_spec_link_descriptors():
    spec = display.column_config_spec(_momentum_profile())
    for col, brand in (("tv_url", "TradingView"), ("yf_url", "Yahoo")):
        assert spec[col]["kind"] == "link"
        assert spec[col]["label"] == brand
        assert spec[col]["display_text"] == "↗"
        assert spec[col]["help"].strip()


def test_column_config_spec_tactical_descriptors():
    # The universe-wide tactical columns are present for EVERY profile as plain
    # dicts (no streamlit objects): Extension as text, In Buy Zone as a checkbox.
    for prof in (_momentum_profile(), _long_term_profile(), _swing_profile()):
        spec = display.column_config_spec(prof)
        assert spec["extension_state"]["kind"] == "text"
        assert spec["extension_state"]["label"] == "Extension"
        assert spec["extension_state"]["help"].strip()
        assert spec["in_buy_zone"]["kind"] == "checkbox"
        assert spec["in_buy_zone"]["label"] == "In Buy Zone"
        assert spec["in_buy_zone"]["help"].strip()
        # Still plain dicts (the purity boundary holds).
        assert isinstance(spec["extension_state"], dict)
        assert isinstance(spec["in_buy_zone"], dict)


def test_column_config_spec_headline_price_descriptors():
    # price + change_pct are present for EVERY profile: price as a plain number
    # (app.py prefixes "$"), change_pct as a SIGNED percent (app.py colours it).
    for prof in (_momentum_profile(), _long_term_profile(), _swing_profile()):
        spec = display.column_config_spec(prof)
        assert spec["price"]["kind"] == "number"
        assert spec["price"]["format"] == "%.2f"
        assert spec["price"]["label"] == "Price"
        assert spec["price"]["help"].strip()
        assert spec["change_pct"]["kind"] == "percent"
        assert spec["change_pct"]["format"] == "percent"
        assert spec["change_pct"]["signed"] is True
        assert spec["change_pct"]["help"].strip()


# =========================================================================
# reasons_to_frame / max_contribution / contribution_caption
# =========================================================================
def test_reasons_to_frame_order_and_columns():
    sw = _swing_profile()
    reasons = _reasons(sw, with_flags=True)
    frame = display.reasons_to_frame(reasons, sw)
    assert list(frame.columns) == [
        "Signal", "What it measures", "Value", "Percentile", "Contribution",
    ]
    # One row per signal, flags excluded.
    assert len(frame) == len(sw.signals)
    # Order preserved (the profile's signal order via humanized labels).
    expected = [display.feature_label(s.feature) for s in sw.signals]
    assert list(frame["Signal"]) == expected
    # The inline definition column carries each signal's plain-English description.
    assert list(frame["What it measures"]) == [
        display.feature_description(s.feature) for s in sw.signals
    ]
    assert all(frame["What it measures"])  # none empty for real signals


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
    assert list(empty.columns) == [
        "Signal", "What it measures", "Value", "Percentile", "Contribution",
    ]
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
# explanations: descriptions, help copy, narrative, glossary
# =========================================================================
def test_feature_descriptions_parity():
    # Every labelled feature has a non-empty description and vice-versa.
    assert set(display.FEATURE_DESCRIPTIONS) == set(display.FEATURE_LABELS)
    assert all(v.strip() for v in display.FEATURE_DESCRIPTIONS.values())


def test_feature_description_accessor():
    # forward_pe is the lone "lower is better" signal — its description says so.
    assert "cheaper" in display.feature_description("forward_pe")
    assert display.feature_description("does_not_exist") == ""  # safe fallback


def test_help_constants_present():
    for txt in (display.SCORE_HELP, display.PERCENTILE_HELP,
                display.CONTRIBUTION_HELP, display.WHAT_HELP, display.WHY_HELP):
        assert isinstance(txt, str) and txt.strip()


def test_tactical_feature_descriptions_parity():
    # Each new tactical key is BOTH labelled and described (the parity the
    # generic test enforces — pinned explicitly here so a half-added key fails).
    for key in ("extension_state", "extension_score", "in_buy_zone", "dist_to_buy_zone_pct"):
        assert key in display.FEATURE_LABELS, f"{key} missing a label"
        assert key in display.FEATURE_DESCRIPTIONS, f"{key} missing a description"
        assert display.feature_description(key).strip()
        assert display.feature_label(key).strip()


def test_tactical_help_constants_present():
    for txt in (display.LEVELS_HELP, display.EXTENSION_HELP, display.BUY_ZONE_HELP):
        assert isinstance(txt, str) and txt.strip()


def test_buy_zone_help_carries_not_advice_disclaimer():
    # FRAMING GUARD for the relaxed "explicit buy zone" decision: the buy-zone
    # help MUST disclaim advice (substring check tolerant of phrasing).
    blob = display.BUY_ZONE_HELP.lower()
    assert "not financial advice" in blob or "not advice" in blob
    # The caption helper carries the same guardrail in both branches.
    zone = _FakeZone(1.0, 2.0, "nearest support", True, 0.0)
    assert "not financial advice" in display.buy_zone_caption(zone).lower()
    assert "not financial advice" in display.buy_zone_caption(None).lower()


def test_profile_descriptions_for_every_profile():
    from screener.profiles import PROFILES
    for name in PROFILES:
        assert display.profile_description(name).strip()
    assert display.profile_description("nope") == ""


def test_signal_glossary_matches_profile_signals():
    sw = _swing_profile()
    gloss = display.signal_glossary(sw)
    assert [lbl for lbl, _ in gloss] == [display.feature_label(s.feature) for s in sw.signals]
    assert all(desc.strip() for _, desc in gloss)  # every signal documented
    assert display.signal_glossary(None) == []


def test_explain_rank_names_strongest_and_weakest():
    p = Profile(
        "t", "T",
        signals=(SignalSpec("revenue_growth", 1.0), SignalSpec("earnings_growth", 1.0),
                 SignalSpec("momentum_12m", 1.0)),
    )
    reasons = OrderedDict()
    reasons["revenue_growth"] = {"value": 0.2, "percentile": 0.90, "contribution": 0.3}
    reasons["earnings_growth"] = {"value": 0.3, "percentile": 1.00, "contribution": 0.33}
    reasons["momentum_12m"] = {"value": 0.1, "percentile": 0.20, "contribution": 0.07}
    reasons["flags"] = {"x": True}  # ignored
    row = pd.Series({"symbol": "AFL", "rank": 2, "score": 0.7})
    txt = display.explain_rank(row, reasons, p, total=50)
    assert txt.startswith("AFL ranks #2 of 50 —")
    # Strongest two = highest percentile (earnings 1.00, then revenue 0.90).
    assert "strongest on Earnings Growth and Revenue Growth" in txt
    # Weakest = lowest percentile (momentum 0.20).
    assert "weakest on 12M Momentum" in txt


def test_explain_rank_edges():
    # No usable signals but a rank -> just the headline.
    row = pd.Series({"symbol": "ZZZ", "rank": 5})
    assert display.explain_rank(row, OrderedDict()) == "ZZZ ranks #5."
    # No row / no reasons -> empty string (caller renders nothing).
    assert display.explain_rank(None, None) == ""
    # total omitted -> no "of N"; single signal -> no "weakest" clause.
    p = Profile("t", "T", signals=(SignalSpec("revenue_growth", 1.0),))
    reasons = OrderedDict()
    reasons["revenue_growth"] = {"value": 0.2, "percentile": 0.9, "contribution": 0.9}
    txt = display.explain_rank(pd.Series({"symbol": "QQQ", "rank": 1}), reasons, p)
    assert txt == "QQQ ranks #1 — strongest on Revenue Growth."


def test_column_config_spec_carries_help():
    sw = _swing_profile()
    spec = display.column_config_spec(sw)
    assert spec["score"]["help"] == display.SCORE_HELP
    assert spec["rank"]["help"].strip()
    # Each signal column gets its description as a header tooltip.
    for s in sw.signals:
        assert spec[s.feature]["help"] == display.feature_description(s.feature)


# =========================================================================
# "surface what we compute": fit_score / narrative / radar / export_frame
# =========================================================================
def test_fit_score():
    assert display.fit_score(0.0) == 0
    assert display.fit_score(1.0) == 100
    assert display.fit_score(0.719) == 72       # rounds to nearest
    assert display.fit_score(0.5) == 50
    # Clamped + fail-soft (never raises, never outside 0..100).
    assert display.fit_score(1.5) == 100
    assert display.fit_score(-0.2) == 0
    assert display.fit_score(None) == 0
    assert display.fit_score(float("nan")) == 0
    assert display.fit_score(np.float64(0.8)) == 80


def test_narrative_capitalizes_and_stands_alone():
    p = Profile("t", "T",
                signals=(SignalSpec("revenue_growth", 1.0), SignalSpec("earnings_growth", 1.0),
                         SignalSpec("momentum_12m", 1.0)))
    reasons = OrderedDict()
    reasons["revenue_growth"] = {"value": 0.2, "percentile": 0.90, "contribution": 0.3}
    reasons["earnings_growth"] = {"value": 0.3, "percentile": 1.00, "contribution": 0.33}
    reasons["momentum_12m"] = {"value": 0.1, "percentile": 0.20, "contribution": 0.07}
    # Capitalized, full-stopped, no rank/symbol head; same strongest/weakest read
    # as explain_rank (which is now factored on top of the shared clause).
    assert display.narrative(reasons) == (
        "Strongest on Earnings Growth and Revenue Growth, weakest on 12M Momentum."
    )
    # Empty / None reasons -> "".
    assert display.narrative(OrderedDict()) == ""
    assert display.narrative(None) == ""


def test_narrative_series_over_frame():
    p = _momentum_profile()
    reasons = _reasons(p)  # all-0.5 percentiles -> a non-empty clause
    df = pd.DataFrame({"symbol": ["AAA", "BBB"], "reasons": [reasons, OrderedDict()]})
    s = display.narrative_series(df, p)
    assert len(s) == 2 and list(s.index) == [0, 1]
    assert s.iloc[0].startswith("Strongest on")
    assert s.iloc[1] == ""                       # empty reasons -> ""
    # No reasons column -> all-empty series of the right length.
    no_r = pd.DataFrame({"symbol": ["X", "Y", "Z"]})
    assert list(display.narrative_series(no_r, p)) == ["", "", ""]
    # Empty frame -> empty series.
    assert len(display.narrative_series(pd.DataFrame(), p)) == 0


def test_radar_spec_axes_match_reasons():
    p = _momentum_profile()
    pcts = {s.feature: (i + 1) / len(p.signals) for i, s in enumerate(p.signals)}
    spec = display.radar_spec(_reasons(p, pcts=pcts), p)
    # One axis per scored signal, in reasons order; short labels; percentile values.
    assert len(spec["labels"]) == len(p.signals)
    assert len(spec["values"]) == len(p.signals)
    assert spec["labels"][0] == display.radar_label(p.signals[0].feature)
    assert math.isclose(spec["values"][-1], 1.0)
    # A missing percentile -> the neutral 0.5 (engine's missing-signal default).
    r2 = OrderedDict()
    r2["momentum_3m"] = {"value": 0.1, "percentile": None, "contribution": 0.0}
    assert math.isclose(display.radar_spec(r2)["values"][0], 0.5)


def test_radar_svg_structure():
    p = _momentum_profile()
    n = len(p.signals)
    svg = display.radar_svg(display.radar_spec(_reasons(p), p))
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    # One vertex circle per axis; 4 grid rings + 1 data polygon.
    assert svg.count("<circle") == n
    assert svg.count("<polygon") == 5
    # A short axis label is rendered (momentum screens on relative volume).
    assert "Rel Vol" in svg
    # Empty / malformed spec -> "" (the caller then renders nothing).
    assert display.radar_svg({"labels": [], "values": []}) == ""
    assert display.radar_svg({"labels": ["a"], "values": []}) == ""


def test_radar_label_short_and_fallback():
    assert display.radar_label("momentum_12m") == "12M Mom"
    assert display.radar_label("rel_volume_20") == "Rel Vol"
    # Unmapped feature -> the full feature_label.
    assert display.radar_label("trailing_pe") == display.feature_label("trailing_pe")


def test_export_frame_drops_links_keeps_fit_and_why():
    p = _momentum_profile()
    df = _result_frame()                         # AAPL/MSFT/NVDA, scores .80/.55/.30
    df["reasons"] = [_reasons(p), _reasons(p), _reasons(p)]
    exp = display.export_frame(df, p)
    assert "tv_url" not in exp.columns and "yf_url" not in exp.columns
    assert "fit" in exp.columns and "why" in exp.columns
    assert "reasons" not in exp.columns
    # Fit is the 0..100 integer headline (round(score * 100)).
    assert list(exp["fit"]) == [80, 55, 30]
    assert exp["why"].iloc[0].startswith("Strongest on")


def test_column_config_spec_fit_and_why_descriptors():
    spec = display.column_config_spec(_momentum_profile())
    # `score` descriptor retained (0..1 progress) AND a new 0..100 `fit` progress.
    assert spec["score"]["kind"] == "progress" and math.isclose(spec["score"]["max"], 1.0)
    assert spec["fit"]["kind"] == "progress"
    assert math.isclose(spec["fit"]["min"], 0.0) and math.isclose(spec["fit"]["max"], 100.0)
    assert spec["fit"]["format"] == "%d" and spec["fit"]["label"] == "Fit"
    # The narrative column is a text column with a non-empty header tooltip.
    assert spec["why"]["kind"] == "text" and spec["why"]["label"] == "Why"
    assert spec["why"]["help"].strip()


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


def test_format_value_tactical_keys():
    # extension_score: a 0..1 fraction rendered as a 2-dec percent.
    assert display.format_value("extension_score", 0.72) == "72.00%"
    assert display.format_value("extension_score", 0.0) == "0.00%"
    assert display.format_value("extension_score", float("nan")) == "—"
    # dist_to_buy_zone_pct: a SIGNED percent (always carries +/-).
    assert display.format_value("dist_to_buy_zone_pct", 0.031) == "+3.1%"
    assert display.format_value("dist_to_buy_zone_pct", -0.012) == "-1.2%"
    assert display.format_value("dist_to_buy_zone_pct", 0.0) == "+0.0%"
    assert display.format_value("dist_to_buy_zone_pct", None) == "—"
    # change_pct: also a SIGNED percent (green up / red down in the grid).
    assert display.format_value("change_pct", 0.012) == "+1.2%"
    assert display.format_value("change_pct", -0.008) == "-0.8%"
    assert display.format_value("change_pct", float("nan")) == "—"
    # in_buy_zone: a boolean Yes/No like the other bool features.
    assert display.format_value("in_buy_zone", True) == "Yes"
    assert display.format_value("in_buy_zone", False) == "No"
    assert display.format_value("in_buy_zone", float("nan")) == "—"
    # extension_state: the categorical badge text (never a float / KeyError).
    assert display.format_value("extension_state", "parabolic") == "🔴 Parabolic"
    assert display.format_value("extension_state", "normal") == "🟢 Normal"


def test_format_price():
    assert display.format_price(195.0) == "$195.00"
    assert display.format_price(1234.5) == "$1,234.50"
    assert display.format_price(0.0) == "$0.00"
    assert display.format_price(None) == "—"
    assert display.format_price(float("nan")) == "—"
    assert display.format_price("not a number") == "—"


def test_format_market_cap():
    # Human units, one decimal, largest-first.
    assert display.format_market_cap(1.23e12) == "$1.2T"
    assert display.format_market_cap(3.45e11) == "$345.0B"
    assert display.format_market_cap(1.2e7) == "$12.0M"
    # Below $1M -> plain grouped dollars.
    assert display.format_market_cap(5e5) == "$500,000"
    # Degenerate / unknown caps fail soft.
    assert display.format_market_cap(0) == "—"
    assert display.format_market_cap(-5) == "—"
    assert display.format_market_cap(None) == "—"
    assert display.format_market_cap(float("nan")) == "—"
    assert display.format_value("extension_state", None) == "—"


# =========================================================================
# tactical formatters: signed pct, extension badge/colour, levels frame, buy zone
# =========================================================================
def test_format_signed_pct():
    assert display.format_signed_pct(0.032) == "+3.2%"
    assert display.format_signed_pct(-0.010) == "-1.0%"
    assert display.format_signed_pct(0.0) == "+0.0%"
    # Missing / non-finite -> em dash, never a crash.
    assert display.format_signed_pct(None) == "—"
    assert display.format_signed_pct(float("nan")) == "—"
    assert display.format_signed_pct(float("inf")) == "—"
    # numpy float tolerated.
    assert display.format_signed_pct(np.float64(0.05)) == "+5.0%"


def test_extension_badge_and_color():
    assert display.extension_badge("normal") == "🟢 Normal"
    assert display.extension_badge("extended") == "🟠 Extended"
    assert display.extension_badge("parabolic") == "🔴 Parabolic"
    # Case-insensitive; unknown / None falls back to the safe Normal baseline.
    assert display.extension_badge("PARABOLIC") == "🔴 Parabolic"
    assert display.extension_badge("nonsense") == "🟢 Normal"
    assert display.extension_badge(None) == "🟢 Normal"
    # Colours pair with st.badge; unknown/None -> neutral gray.
    assert display.extension_state_color("normal") == "gray"
    assert display.extension_state_color("extended") == "orange"
    assert display.extension_state_color("parabolic") == "red"
    assert display.extension_state_color(None) == "gray"
    assert display.extension_state_color("???") == "gray"


def test_levels_to_frame_columns_and_order():
    frame = display.levels_to_frame(_level_set())
    assert list(frame.columns) == ["Level", "Kind", "Price", "Touches", "Strength", "Distance"]
    # Resistances stack ABOVE supports; each side nearest-first as supplied.
    assert list(frame["Level"]) == ["Resistance 1", "Support 1", "Support 2"]
    assert list(frame["Kind"]) == ["Resistance", "Support", "Support"]
    # Strength stays a numeric 0..1 float (a progress column consumes it).
    assert all(0.0 <= s <= 1.0 for s in frame["Strength"])
    assert math.isclose(float(frame.iloc[1]["Strength"]), 0.91)
    # Touches is an int; Price a float.
    assert int(frame.iloc[0]["Touches"]) == 3
    assert math.isclose(float(frame.iloc[0]["Price"]), 160.0)
    # Distance is the signed-percent STRING (above = +, below = -).
    assert frame.iloc[0]["Distance"] == "+7.3%"
    assert frame.iloc[1]["Distance"] == "-0.7%"


def test_levels_to_frame_failsoft():
    cols = ["Level", "Kind", "Price", "Touches", "Strength", "Distance"]
    # None -> empty frame with the right columns.
    none_f = display.levels_to_frame(None)
    assert len(none_f) == 0 and list(none_f.columns) == cols
    # Empty LevelSet (no supports/resistances) -> empty frame.
    empty_f = display.levels_to_frame(_FakeLevelSet(supports=(), resistances=()))
    assert len(empty_f) == 0 and list(empty_f.columns) == cols
    # A non-finite strength is coerced into [0,1] (progress column stays valid).
    bad = _FakeLevelSet(
        supports=(_FakeLevel(100.0, "support", 2, float("nan"), float("nan")),),
        resistances=(),
    )
    bf = display.levels_to_frame(bad)
    assert len(bf) == 1
    assert math.isfinite(float(bf.iloc[0]["Strength"])) and 0.0 <= float(bf.iloc[0]["Strength"]) <= 1.0
    # A NaN distance renders as the em dash (not "+nan%").
    assert bf.iloc[0]["Distance"] == "—"


def test_format_buy_zone_and_caption():
    zone = _FakeZone(low=145.20, high=148.50, basis="nearest support · 3 touches",
                     in_zone=True, distance_pct=0.0)
    assert display.format_buy_zone(zone) == "$145.20 – $148.50"
    cap = display.buy_zone_caption(zone)
    assert "nearest support · 3 touches" in cap
    # Disclaimer always present (the relaxed-decision guardrail).
    assert "not financial advice" in cap.lower()
    # None zone -> em-dash band, and a caption that STILL carries the disclaimer.
    assert display.format_buy_zone(None) == "—"
    none_cap = display.buy_zone_caption(None)
    assert "not financial advice" in none_cap.lower()
    # Non-finite edges -> em dash (never "$nan").
    assert display.format_buy_zone(_FakeZone(float("nan"), 10.0, "x", False, 0.0)) == "—"


def test_tactical_formatters_consume_real_levels_objects():
    """The formatters duck-type the genuine screener.levels dataclasses.

    Proves the field contract matches the real (frozen) classes, not just the
    stand-ins. Skips silently if levels is somehow unavailable (it is pure
    pandas/numpy — no streamlit/yfinance — so this normally runs).
    """
    try:
        from screener.levels import BuyZone, Level, LevelSet
    except Exception:  # pragma: no cover
        return
    ts = pd.Timestamp("2026-01-02")
    res = Level(price=160.0, kind="resistance", touches=3, strength=0.8,
                distance_pct=0.07, first=ts, last=ts, timeframe="1d")
    sup = Level(price=148.0, kind="support", touches=4, strength=0.9,
                distance_pct=-0.01, first=ts, last=ts, timeframe="1d")
    ls = LevelSet(supports=(sup,), resistances=(res,), last_close=150.0, timeframe="1d")
    frame = display.levels_to_frame(ls)
    assert list(frame["Level"]) == ["Resistance 1", "Support 1"]
    assert frame.iloc[0]["Distance"] == "+7.0%"
    zone = BuyZone(low=145.2, high=148.5, basis="nearest support · 4 touches",
                   in_zone=False, distance_pct=-0.011, timeframe="1d")
    assert display.format_buy_zone(zone) == "$145.20 – $148.50"
    assert "not financial advice" in display.buy_zone_caption(zone).lower()


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


def test_selectivity_hint_names_hard_filters():
    swing = get_profile("swing")
    phrases = display.hard_filter_phrases(swing)
    # Both swing hard filters are described, with the rel-volume threshold rendered
    # and the leading-sector clause present.
    blob = " | ".join(phrases)
    assert "2" in blob and "×" in blob
    assert any("sector" in p for p in phrases)
    # No phrase carries its own parentheses (they nest inside the hint's paren).
    assert all("(" not in p and ")" not in p for p in phrases)

    hint = display.selectivity_hint(swing, 35, 500)
    assert "35 of 500" in hint
    assert "Swing" in hint
    assert "not a data error" in hint
    # Exactly one balanced parenthetical, no nesting.
    assert hint.count("(") == 1 and hint.count(")") == 1

    # A profile with no hard filters yields no hint.
    nofilt = Profile(name="x", label="X", filters=(), signals=(SignalSpec("momentum_1m", 1.0),))
    assert display.selectivity_hint(nofilt, 5, 5) == ""

    # Generic fallback for an unknown numeric filter never raises and keeps the value.
    custom = Profile(name="c", label="C", filters=(Filter("rsi_14", ">", 50.0),))
    cph = display.hard_filter_phrases(custom)
    assert cph and "50" in cph[0]


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

    table = display.table_view(view, swing, density="detailed")
    assert "reasons" not in table.columns
    assert "earnings_in_window" in table.columns  # swing (Detailed reveals signals)
    assert not any(c.endswith("_pct") for c in table.columns if c != "change_pct")

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
