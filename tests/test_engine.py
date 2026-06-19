"""Synthetic, network-free tests for the profile + ranking engine.

Covers :mod:`screener.profiles` (declarative configs + registry) and
:mod:`screener.engine` (feature assembly, sector strength, hard filters, the
PURE cross-sectional percentile scorer, and the end-to-end ``run_screen``).

Like ``tests/test_indicators.py``: no ``pytest`` import and no ``yfinance`` — every
test is a plain ``test_*`` function using ``assert`` + ``math.isclose`` so the suite
runs BOTH under ``python -m pytest tests/test_engine.py`` AND standalone as
``python tests/test_engine.py`` (the ``__main__`` runner counts pass/fail, prints a
summary, and exits non-zero on any failure).

A network-free :class:`FakeProvider` (a real :class:`~screener.provider.DataProvider`
subclass) serves canned ~300-bar synthetic OHLCV frames, fundamentals, and earnings
dates for a small synthetic universe, so ``assemble_features`` / ``run_screen`` run
fully offline and deterministically (``as_of`` is always pinned).
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
from screener import profiles as prof  # noqa: E402
from screener.profiles import Filter, Profile, SignalSpec  # noqa: E402
from screener.provider import DataProvider, Fundamentals  # noqa: E402


# --- helpers -------------------------------------------------------------
def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _make_frame(
    n=300,
    *,
    start=50.0,
    drift=0.30,
    noise=0.05,
    vol_spike=False,
    base_volume=1_000_000.0,
    falling=False,
    flat=False,
    seed=0,
):
    """A canonical ~300-bar OHLCV frame so ``snapshot`` is fully populated.

    ``vol_spike`` blows up only the LAST bar's volume (5x the steady baseline) so
    ``rel_volume_20`` clears the swing ``> 2.0`` filter. ``falling`` / ``flat``
    let a ticker FAIL a trend filter (e.g. ``price_above_sma_50``) on purpose.
    """
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    idx.name = "date"
    rng = np.random.default_rng(seed)
    if flat:
        close = np.full(n, float(start)) + rng.normal(0, noise, n)
    elif falling:
        close = float(start) + np.arange(n) * (-abs(drift)) + rng.normal(0, noise, n)
        close = np.clip(close, 1.0, None)
    else:
        close = float(start) + np.arange(n) * abs(drift) + rng.normal(0, noise, n)
    volume = np.full(n, float(base_volume))
    if vol_spike:
        volume[-1] = base_volume * 5.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


class FakeProvider(DataProvider):
    """Offline :class:`DataProvider`: canned frames / fundamentals / earnings.

    Constructed from three dicts keyed by symbol. An unknown symbol yields an
    empty price frame, an (almost) empty Fundamentals, and ``None`` earnings — so
    the engine's fail-soft path is exercised without any network.
    """

    def __init__(self, frames=None, fundamentals=None, earnings=None):
        self._frames = dict(frames or {})
        self._fundamentals = dict(fundamentals or {})
        self._earnings = dict(earnings or {})

    def price_history(self, symbol, *, lookback_days=730):
        sym = symbol.strip().upper()
        if sym in self._frames:
            return self._frames[sym]
        # Unknown / "bad" ticker: empty canonical frame (fail-soft).
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="date"),
        )

    def fundamentals(self, symbol):
        sym = symbol.strip().upper()
        if sym in self._fundamentals:
            return self._fundamentals[sym]
        return Fundamentals(symbol=sym)

    def earnings_date(self, symbol):
        return self._earnings.get(symbol.strip().upper())


# A fixed clock so the earnings-window logic is deterministic in every test.
AS_OF = dt.date(2026, 6, 20)


def _universe(rows):
    """A universe DataFrame[symbol, name, sector] from a list of (sym, name, sector)."""
    return pd.DataFrame(rows, columns=["symbol", "name", "sector"])


def _hand_frame(data, index):
    """A features frame indexed by 'symbol' from a column dict."""
    df = pd.DataFrame(data, index=index)
    df.index.name = "symbol"
    return df


# =========================================================================
# profiles.py — declarative configs + registry
# =========================================================================
def test_registry_has_three_profiles():
    assert set(prof.PROFILES) == {"long_term", "swing", "momentum"}
    for name, p in prof.PROFILES.items():
        assert isinstance(p, Profile)
        assert p.name == name


def test_get_profile_case_insensitive_and_unknown_raises():
    assert prof.get_profile("SWING").name == "swing"
    assert prof.get_profile(" Momentum ").name == "momentum"
    raised = False
    try:
        prof.get_profile("nope")
    except KeyError:
        raised = True
    assert raised


def test_signalspec_rejects_bad_direction():
    raised = False
    try:
        SignalSpec("x", 1.0, "sideways")
    except ValueError:
        raised = True
    assert raised


def test_filter_rejects_bad_op():
    raised = False
    try:
        Filter("x", "≈", 1.0)
    except ValueError:
        raised = True
    assert raised


def test_profiles_only_use_higher_or_lower_directions():
    # The scorer only ever sees higher/lower; banded logic is pre-derived.
    for p in prof.PROFILES.values():
        for s in p.signals:
            assert s.direction in {"higher", "lower"}


def test_swing_has_expected_filters_and_flag():
    swing = prof.get_profile("swing")
    feats = {f.feature: (f.op, f.threshold) for f in swing.filters}
    assert feats["rel_volume_20"] == (">", 2.0)
    assert feats["in_leading_sector"][0] == "is_true"
    assert "earnings_in_window" in swing.flags


# =========================================================================
# score_and_rank — the PURE, unit-tested core
# =========================================================================
def test_score_rank_top_value_percentile_is_one():
    # No NaN: the largest value gets percentile 1.0 before weighting.
    df = _hand_frame({"x": [1.0, 2.0, 3.0, 4.0]}, ["A", "B", "C", "D"])
    p = Profile("t", "T", signals=(SignalSpec("x", 1.0, "higher"),))
    out = eng.score_and_rank(df, p)
    assert math.isclose(out.loc["D", "x_pct"], 1.0)
    assert math.isclose(out.loc["A", "x_pct"], 0.25)
    # With one signal, score == its percentile.
    assert math.isclose(out.loc["D", "score"], 1.0)


def test_score_rank_lower_direction_inverts():
    # direction "lower": the SMALLEST value ranks best (rank 1, highest pct).
    df = _hand_frame({"forward_pe": [10.0, 20.0, 30.0, 40.0]}, ["A", "B", "C", "D"])
    p = Profile("t", "T", signals=(SignalSpec("forward_pe", 1.0, "lower"),))
    out = eng.score_and_rank(df, p)
    assert int(out.loc["A", "rank"]) == 1
    assert math.isclose(out.loc["A", "forward_pe_pct"], 0.75)  # 1 - 0.25
    assert math.isclose(out.loc["D", "forward_pe_pct"], 0.0)   # 1 - 1.0
    assert out.loc["A", "score"] > out.loc["D", "score"]


def test_score_rank_missing_is_neutral_half():
    # A NaN value must not skew the ranking and is then set to 0.5 neutral.
    df = _hand_frame({"x": [1.0, 2.0, 3.0, np.nan]}, ["A", "B", "C", "D"])
    p = Profile("t", "T", signals=(SignalSpec("x", 1.0, "higher"),))
    out = eng.score_and_rank(df, p)
    assert math.isclose(out.loc["D", "x_pct"], 0.5)
    # The non-missing values rank only among themselves (top C -> 1.0).
    assert math.isclose(out.loc["C", "x_pct"], 1.0)


def test_score_rank_normalized_weights_and_contribution_sum():
    # Two signals with un-normalized weights (3 and 1) -> normalized 0.75 / 0.25.
    df = _hand_frame(
        {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]}, ["A", "B", "C"]
    )
    p = Profile(
        "t", "T",
        signals=(SignalSpec("a", 3.0, "higher"), SignalSpec("b", 1.0, "higher")),
    )
    out = eng.score_and_rank(df, p)
    # C: a is top (pct 1.0, normwt .75), b is bottom (pct 1/3, normwt .25).
    reasons = dict(out.loc["C", "reasons"])
    assert math.isclose(reasons["a"]["contribution"], 0.75 * 1.0)
    assert math.isclose(reasons["b"]["contribution"], 0.25 * (1.0 / 3.0))
    # Contributions sum to the score (flags excluded).
    contrib_sum = sum(v["contribution"] for k, v in reasons.items() if k != "flags")
    assert math.isclose(contrib_sum, out.loc["C", "score"])
    # And the score stays in [0, 1].
    assert 0.0 <= out.loc["C", "score"] <= 1.0


def test_score_rank_reasons_dict_complete():
    df = _hand_frame({"a": [1.0, 2.0], "b": [5.0, 6.0]}, ["A", "B"])
    p = Profile(
        "t", "T",
        signals=(SignalSpec("a", 1.0, "higher"), SignalSpec("b", 1.0, "higher")),
        flags=("flagcol",),
    )
    df["flagcol"] = [True, False]
    out = eng.score_and_rank(df, p)
    reasons = dict(out.loc["A", "reasons"])
    # Every signal present with the three documented keys.
    for sig in ("a", "b"):
        assert set(reasons[sig].keys()) == {"value", "percentile", "contribution"}
    assert math.isclose(reasons["a"]["value"], 1.0)
    # Fired flags carried under "flags".
    assert "flags" in reasons
    assert reasons["flags"]["flagcol"] is True


def test_score_rank_deterministic_tie_break_by_symbol():
    # Equal scores -> symbol ascending; ranks contiguous 1..n.
    df = _hand_frame({"x": [5.0, 5.0, 5.0]}, ["ZZZ", "AAA", "MMM"])
    p = Profile("t", "T", signals=(SignalSpec("x", 1.0, "higher"),))
    out = eng.score_and_rank(df, p)
    assert list(out.index) == ["AAA", "MMM", "ZZZ"]
    assert list(out["rank"]) == [1, 2, 3]


def test_score_rank_booleans_rank_true_above_false():
    # A bool signal: True ranks above False.
    df = _hand_frame({"flag": [True, False, True, False]}, ["A", "B", "C", "D"])
    p = Profile("t", "T", signals=(SignalSpec("flag", 1.0, "higher"),))
    out = eng.score_and_rank(df, p)
    assert out.loc["A", "flag_pct"] > out.loc["B", "flag_pct"]


def test_score_rank_empty_input_same_columns_no_error():
    empty = pd.DataFrame().rename_axis("symbol")
    p = Profile("t", "T", signals=(SignalSpec("x", 1.0, "higher"),))
    out = eng.score_and_rank(empty, p)
    assert len(out) == 0
    for col in ("x_pct", "score", "rank", "reasons"):
        assert col in out.columns


# =========================================================================
# apply_filters — hard cutoffs, fail-closed
# =========================================================================
def test_apply_filters_gt_and_is_true_cutoffs():
    df = _hand_frame(
        {"rel_volume_20": [3.0, 1.0, 5.0], "flag": [True, True, False]},
        ["A", "B", "C"],
    )
    p = Profile(
        "t", "T",
        filters=(Filter("rel_volume_20", ">", 2.0), Filter("flag", "is_true")),
    )
    out = eng.apply_filters(df, p)
    # A passes both; B fails > 2.0; C fails is_true.
    assert list(out.index) == ["A"]


def test_apply_filters_nan_and_none_fail_closed():
    df = _hand_frame(
        {"rel_volume_20": [3.0, np.nan, 9.0], "flag": [True, True, None]},
        ["A", "B", "C"],
    )
    p = Profile(
        "t", "T",
        filters=(Filter("rel_volume_20", ">", 2.0), Filter("flag", "is_true")),
    )
    out = eng.apply_filters(df, p)
    # B's NaN volume fails closed; C's None flag fails closed; only A survives.
    assert list(out.index) == ["A"]


def test_apply_filters_is_true_rejects_non_true():
    # "is_true" passes ONLY on Python True (not 1, not truthy strings).
    df = _hand_frame({"flag": [True, 1, "yes", False]}, ["A", "B", "C", "D"])
    p = Profile("t", "T", filters=(Filter("flag", "is_true"),))
    out = eng.apply_filters(df, p)
    assert list(out.index) == ["A"]


def test_apply_filters_missing_column_drops_all():
    df = _hand_frame({"x": [1.0, 2.0]}, ["A", "B"])
    p = Profile("t", "T", filters=(Filter("not_a_col", ">", 0.0),))
    out = eng.apply_filters(df, p)
    assert len(out) == 0


def test_apply_filters_empty_input():
    empty = pd.DataFrame().rename_axis("symbol")
    p = Profile("t", "T", filters=(Filter("x", ">", 0.0),))
    out = eng.apply_filters(empty, p)
    assert len(out) == 0


# =========================================================================
# compute_sector_strength — top-N leading by median momentum_3m
# =========================================================================
def test_sector_strength_top3_leading():
    # 4 sectors, distinct medians: Tech > Energy > Health > Util.
    df = _hand_frame(
        {
            "sector": ["Tech", "Tech", "Energy", "Energy", "Health", "Health", "Util", "Util"],
            "momentum_3m": [0.30, 0.40, 0.20, 0.25, 0.10, 0.12, -0.05, -0.10],
        },
        ["T1", "T2", "E1", "E2", "H1", "H2", "U1", "U2"],
    )
    out = eng.compute_sector_strength(df, top_n=3)
    # Medians: Tech .35, Energy .225, Health .11, Util -.075.
    assert math.isclose(out.loc["T1", "sector_median_3m"], 0.35)
    assert int(out.loc["T1", "sector_rank"]) == 1
    assert int(out.loc["U1", "sector_rank"]) == 4
    # Top-3 leading: Tech / Energy / Health in, Util out. (The column is bool
    # dtype, so cells box to numpy.bool_ — compare by value, not identity.)
    assert bool(out.loc["T1", "in_leading_sector"]) is True
    assert bool(out.loc["E1", "in_leading_sector"]) is True
    assert bool(out.loc["H1", "in_leading_sector"]) is True
    assert bool(out.loc["U1", "in_leading_sector"]) is False
    # Strength score higher for the stronger sector.
    assert out.loc["T1", "sector_strength_score"] > out.loc["U1", "sector_strength_score"]


def test_sector_strength_missing_median_not_leading():
    # A sector whose only member has a NaN momentum -> no median -> not leading,
    # NaN strength score (neutralized to 0.5 later by the scorer).
    df = _hand_frame(
        {
            "sector": ["Tech", "Tech", "Mystery"],
            "momentum_3m": [0.30, 0.40, np.nan],
        },
        ["T1", "T2", "M1"],
    )
    out = eng.compute_sector_strength(df, top_n=3)
    assert bool(out.loc["M1", "in_leading_sector"]) is False
    assert _isnan(out.loc["M1", "sector_strength_score"])
    assert _isnan(out.loc["M1", "sector_median_3m"])


def test_sector_strength_empty_input_has_columns():
    empty = pd.DataFrame().rename_axis("symbol")
    out = eng.compute_sector_strength(empty)
    for col in ("sector_median_3m", "sector_rank", "sector_strength_score", "in_leading_sector"):
        assert col in out.columns
    assert len(out) == 0


# =========================================================================
# derived features — in [0, 1], moving in the documented direction
# =========================================================================
def test_pullback_quality_band():
    # Peaks just above the EMAs (~+1.5% offset), decays both ways, clamped [0,1].
    peak = eng.pullback_quality(0.015, 0.015)
    extended = eng.pullback_quality(0.08, 0.08)   # far above -> overextended
    below = eng.pullback_quality(-0.04, -0.04)    # below -> trend breaking
    assert math.isclose(peak, 1.0)
    assert 0.0 <= extended <= 1.0 and 0.0 <= below <= 1.0
    assert peak > extended
    assert peak > below
    # A small cushion (+1.5%) outscores a much larger extension (+8%).
    assert eng.pullback_quality(0.015, 0.015) > eng.pullback_quality(0.08, 0.08)


def test_pullback_quality_nan_passthrough():
    assert _isnan(eng.pullback_quality(float("nan"), 0.01))
    assert _isnan(eng.pullback_quality(0.01, float("nan")))


def test_rsi_health_curve():
    # Peak at 70; rises 30->70; penalizes overbought (>70) and floors at 0.5.
    vals = {r: eng.rsi_health(r) for r in (20, 30, 50, 70, 85, 100)}
    for r, v in vals.items():
        assert 0.0 <= v <= 1.0, (r, v)
    assert math.isclose(vals[30], 0.0)
    assert math.isclose(vals[70], 1.0)              # the peak
    assert math.isclose(vals[50], 0.5)
    assert math.isclose(vals[100], 0.5)             # overbought floor
    assert vals[20] <= vals[30]                     # oversold weak
    # Rising on the way up to 70 ...
    assert vals[50] < vals[70]
    # ... and PENALIZED above 70 (overbought decays back down).
    assert vals[85] < vals[70]
    assert vals[100] < vals[85]


def test_ema_cross_score_direction():
    # Fresh bullish cross = 1.0; decays with bars; bearish = 0.0; floors at 0.1.
    fresh = eng.ema_5_9_cross_score("bullish", "up", 0)
    aged = eng.ema_5_9_cross_score("bullish", "none", 5)
    bearish = eng.ema_5_9_cross_score("bearish", "none", 0)
    unknown = eng.ema_5_9_cross_score("bullish", "none", None)
    for v in (fresh, aged, bearish, unknown):
        assert 0.0 <= v <= 1.0
    assert math.isclose(fresh, 1.0)
    assert math.isclose(bearish, 0.0)
    assert fresh > aged > bearish
    assert math.isclose(unknown, 0.1)               # bullish-but-unknown floor


def test_earnings_window_respects_as_of_and_7_day_window():
    # window = next earnings within 7 calendar days (inclusive), upcoming only.
    assert eng._earnings_window(dt.date(2026, 6, 23), AS_OF) == (True, 3)
    assert eng._earnings_window(dt.date(2026, 6, 27), AS_OF) == (True, 7)   # edge
    assert eng._earnings_window(dt.date(2026, 6, 28), AS_OF) == (False, 8)  # past edge
    assert eng._earnings_window(dt.date(2026, 6, 18), AS_OF) == (False, -2) # past
    assert eng._earnings_window(None, AS_OF) == (False, None)


# =========================================================================
# assemble_features — fail-soft, derived columns, earnings via as_of
# =========================================================================
def test_assemble_features_populates_snapshot_and_derived():
    frame = _make_frame(vol_spike=True, seed=1)
    provider = FakeProvider(
        frames={"AAA": frame},
        fundamentals={
            "AAA": Fundamentals(
                symbol="AAA", name="Alpha", sector="Tech", market_cap=1e12,
                forward_pe=18.0, trailing_pe=22.0, revenue_growth=0.20,
                earnings_growth=0.25,
            )
        },
        earnings={"AAA": dt.date(2026, 6, 23)},
    )
    feats = eng.assemble_features(_universe([("AAA", "Alpha", "Tech")]), provider, as_of=AS_OF)
    assert list(feats.index) == ["AAA"]
    # 21 snapshot keys present and populated.
    for key in eng.SNAPSHOT_KEYS:
        assert key in feats.columns
    assert feats.loc["AAA", "rel_volume_20"] > 2.0
    # Fundamentals joined.
    assert feats.loc["AAA", "forward_pe"] == 18.0
    assert feats.loc["AAA", "sector"] == "Tech"
    # Derived columns present and in range.
    for col in ("pct_from_ema_10", "pct_from_ema_20", "pullback_quality",
                "rsi_health", "ema_5_9_cross_score"):
        assert col in feats.columns
    assert 0.0 <= feats.loc["AAA", "rsi_health"] <= 1.0
    assert 0.0 <= feats.loc["AAA", "ema_5_9_cross_score"] <= 1.0
    # Earnings window driven by as_of (23rd is 3 days out -> in window). The
    # column is bool dtype, so cells box to numpy.bool_ — compare by value.
    assert bool(feats.loc["AAA", "earnings_in_window"]) is True
    assert int(feats.loc["AAA", "days_to_earnings"]) == 3


def test_assemble_features_empty_price_frame_is_failsoft():
    # A ticker with no price frame yields NaN/None features, never raises.
    provider = FakeProvider(frames={})  # every symbol -> empty frame
    feats = eng.assemble_features(_universe([("BAD", "Bad Co", "Energy")]), provider, as_of=AS_OF)
    assert list(feats.index) == ["BAD"]
    assert _isnan(feats.loc["BAD", "momentum_3m"])
    # price_above_* is object dtype (holds None) -> native None survives.
    assert feats.loc["BAD", "price_above_sma_50"] is None
    # sma_stacked / earnings flags are bool dtype -> compare by value.
    assert bool(feats.loc["BAD", "sma_stacked_20_50_150"]) is False
    # Falls back to the universe name/sector when fundamentals are empty.
    assert feats.loc["BAD", "sector"] == "Energy"
    assert bool(feats.loc["BAD", "earnings_in_window"]) is False


def test_assemble_features_earnings_out_of_window_false():
    frame = _make_frame(seed=2)
    provider = FakeProvider(
        frames={"CCC": frame},
        earnings={"CCC": dt.date(2026, 7, 15)},  # ~25 days out
    )
    feats = eng.assemble_features(_universe([("CCC", "Gamma", "Tech")]), provider, as_of=AS_OF)
    assert bool(feats.loc["CCC", "earnings_in_window"]) is False
    assert int(feats.loc["CCC", "days_to_earnings"]) == (dt.date(2026, 7, 15) - AS_OF).days


# =========================================================================
# run_screen — each of the 3 profiles end-to-end via FakeProvider
# =========================================================================
def _swing_ready_universe():
    """A universe + provider where several Tech/Energy names clear the swing gate.

    Swing needs rel_volume_20 > 2.0 AND in_leading_sector. We make two strong
    sectors (Tech, Energy) with rising, volume-spiking members and two weak
    sectors so the leading-sector cut is meaningful.
    """
    rows = [
        ("AAA", "Alpha", "Tech"),
        ("BBB", "Bravo", "Tech"),
        ("CCC", "Char", "Energy"),
        ("DDD", "Delta", "Energy"),
        ("EEE", "Echo", "Health"),
        ("FFF", "Fox", "Utilities"),
    ]
    frames = {
        # Strong sectors: rising + last-bar volume spike -> rel_volume_20 ~5x.
        "AAA": _make_frame(start=40, drift=0.35, vol_spike=True, seed=10),
        "BBB": _make_frame(start=60, drift=0.30, vol_spike=True, seed=11),
        "CCC": _make_frame(start=30, drift=0.28, vol_spike=True, seed=12),
        "DDD": _make_frame(start=80, drift=0.25, vol_spike=True, seed=13),
        # Weak sectors: gently falling, no spike -> excluded by sector / volume.
        "EEE": _make_frame(start=50, drift=0.05, falling=True, seed=14),
        "FFF": _make_frame(start=50, drift=0.05, falling=True, seed=15),
    }
    funds = {
        sym: Fundamentals(
            symbol=sym, name=name, sector=sector, market_cap=5e11,
            forward_pe=15.0 + i, trailing_pe=20.0 + i,
            revenue_growth=0.10 + i * 0.01, earnings_growth=0.12 + i * 0.01,
        )
        for i, (sym, name, sector) in enumerate(rows)
    }
    earnings = {"AAA": dt.date(2026, 6, 23)}  # one name flagged in-window
    return _universe(rows), FakeProvider(frames=frames, fundamentals=funds, earnings=earnings)


def _assert_well_formed_result(result, profile_name):
    """Shared shape checks for a non-empty run_screen result."""
    p = prof.get_profile(profile_name)
    assert len(result) > 0, f"{profile_name}: expected a non-empty ranked frame"
    for col in ("symbol", "name", "sector", "score", "rank", "reasons"):
        assert col in result.columns, f"{profile_name}: missing {col}"
    # symbol is a COLUMN, not the index.
    assert result.index.name != "symbol"
    # score in [0, 1].
    assert ((result["score"] >= 0.0) & (result["score"] <= 1.0)).all()
    # rank contiguous from 1, sorted ascending.
    assert list(result["rank"]) == list(range(1, len(result) + 1))
    # reasons per row: an ordered dict with each scored signal.
    for _, row in result.iterrows():
        reasons = dict(row["reasons"])
        for s in p.signals:
            assert s.feature in reasons
            assert set(reasons[s.feature]) == {"value", "percentile", "contribution"}


def test_run_screen_momentum_end_to_end():
    universe, provider = _swing_ready_universe()
    result = eng.run_screen("momentum", universe, provider, as_of=AS_OF)
    _assert_well_formed_result(result, "momentum")
    # The rising names (price above SMA50) survive momentum's filter; the two
    # falling Health/Utilities names do not.
    assert "EEE" not in set(result["symbol"])
    assert "FFF" not in set(result["symbol"])
    assert {"AAA", "BBB", "CCC", "DDD"}.issubset(set(result["symbol"]))


def test_run_screen_long_term_end_to_end():
    universe, provider = _swing_ready_universe()
    result = eng.run_screen("long_term", universe, provider, as_of=AS_OF)
    _assert_well_formed_result(result, "long_term")
    # long_term needs forward_pe > 0 and price_above_sma_150 (rising names pass).
    assert {"AAA", "BBB", "CCC", "DDD"}.issubset(set(result["symbol"]))


def test_run_screen_swing_end_to_end():
    universe, provider = _swing_ready_universe()
    result = eng.run_screen("swing", universe, provider, as_of=AS_OF)
    _assert_well_formed_result(result, "swing")
    # Only leading-sector (Tech/Energy) + rel_volume>2 names survive.
    survivors = set(result["symbol"])
    assert survivors.issubset({"AAA", "BBB", "CCC", "DDD"})
    assert len(survivors) >= 1
    assert "EEE" not in survivors and "FFF" not in survivors
    # The earnings flag column is present and AAA is flagged in-window.
    assert "earnings_in_window" in result.columns
    aaa = result[result["symbol"] == "AAA"]
    if not aaa.empty:
        assert bool(aaa.iloc[0]["earnings_in_window"]) is True


def test_run_screen_accepts_profile_object():
    universe, provider = _swing_ready_universe()
    result = eng.run_screen(prof.get_profile("momentum"), universe, provider, as_of=AS_OF)
    assert len(result) > 0


# =========================================================================
# empty / degenerate — never raise, empty frame out
# =========================================================================
def test_run_screen_empty_universe_returns_empty():
    empty_universe = pd.DataFrame(columns=["symbol", "name", "sector"])
    provider = FakeProvider()
    result = eng.run_screen("momentum", empty_universe, provider, as_of=AS_OF)
    # Empty in -> empty out, never a crash. On a wholly empty universe the
    # feature frame had no rows, so name/sector aren't materialized; the always-
    # present scoring columns still are.
    assert len(result) == 0
    for col in ("symbol", "score", "rank", "reasons"):
        assert col in result.columns


def test_run_screen_all_filtered_out_returns_empty():
    # Every name is falling (fails momentum's price_above_sma_50) -> empty result.
    rows = [("AAA", "Alpha", "Tech"), ("BBB", "Bravo", "Energy")]
    frames = {
        "AAA": _make_frame(falling=True, seed=20),
        "BBB": _make_frame(falling=True, seed=21),
    }
    provider = FakeProvider(frames=frames)
    result = eng.run_screen("momentum", _universe(rows), provider, as_of=AS_OF)
    assert len(result) == 0
    # Still a well-formed (empty) frame.
    for col in ("symbol", "score", "rank", "reasons"):
        assert col in result.columns


def test_run_screen_bad_ticker_skipped_without_crashing():
    # A universe mixing a good rising name with a "bad" empty-frame ticker: the
    # bad one fails filters / scores neutral, the scan never crashes.
    rows = [("GOOD", "Good Co", "Tech"), ("BAD", "Bad Co", "Tech")]
    frames = {"GOOD": _make_frame(vol_spike=True, seed=22)}  # BAD absent -> empty
    funds = {"GOOD": Fundamentals(symbol="GOOD", name="Good Co", sector="Tech", forward_pe=12.0)}
    provider = FakeProvider(frames=frames, fundamentals=funds)
    result = eng.run_screen("momentum", _universe(rows), provider, as_of=AS_OF)
    # GOOD survives; BAD (empty frame -> price_above_sma_50 None) fails closed.
    assert "GOOD" in set(result["symbol"])
    assert "BAD" not in set(result["symbol"])


def test_assemble_features_empty_universe_no_rows():
    provider = FakeProvider()
    feats = eng.assemble_features(pd.DataFrame(columns=["symbol", "name", "sector"]), provider, as_of=AS_OF)
    assert len(feats) == 0


def test_run_screen_duplicate_symbol_in_universe_no_crash():
    # A duplicated symbol — literal, or a case/whitespace collision that canonical-
    # izes to the same upper()/strip()ed key — must NOT crash (it would otherwise
    # give a non-unique index and blow up score_and_rank's .at[] lookups).
    frame = _make_frame(vol_spike=True, seed=30)
    funds = {"AAA": Fundamentals(symbol="AAA", name="Alpha", sector="Tech", forward_pe=12.0)}
    provider = FakeProvider(frames={"AAA": frame}, fundamentals=funds)
    for rows in (
        [("AAA", "Alpha", "Tech"), ("AAA", "Alpha Dup", "Tech")],   # literal dup
        [("aaa", "Alpha", "Tech"), (" AAA ", "Alpha Dup", "Tech")],  # canonical collision
    ):
        result = eng.run_screen("momentum", _universe(rows), provider, as_of=AS_OF)
        # De-duplicated to a single canonical row; ranked and well-formed.
        assert list(result["symbol"]) == ["AAA"]
        assert list(result["rank"]) == [1]


def test_assemble_features_dedupes_canonical_symbol():
    # The assembled feature frame keeps a UNIQUE symbol index (first row wins).
    frame = _make_frame(seed=31)
    provider = FakeProvider(frames={"AAA": frame})
    feats = eng.assemble_features(
        _universe([("AAA", "First", "Tech"), (" aaa ", "Second", "Energy")]), provider, as_of=AS_OF
    )
    assert list(feats.index) == ["AAA"]
    assert feats.index.is_unique
    # First occurrence wins for the universe fallback fields.
    assert feats.loc["AAA", "sector"] == "Tech"


def test_run_screen_empty_universe_has_stable_lead_schema():
    # Even on a wholly empty universe the result carries the full lead schema, so a
    # downstream consumer can read result['name']/['sector'] without a KeyError.
    empty_universe = pd.DataFrame(columns=["symbol", "name", "sector"])
    result = eng.run_screen("momentum", empty_universe, FakeProvider(), as_of=AS_OF)
    assert len(result) == 0
    for col in ("symbol", "name", "sector", "score", "rank", "reasons"):
        assert col in result.columns


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
