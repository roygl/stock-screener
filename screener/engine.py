"""The profile + ranking engine — one generic pipeline, many declarative lenses.

A :class:`~screener.profiles.Profile` (M4) says *what* to screen on (hard filters
+ weighted signals); this module is the *how*. The flow is five pure-ish stages:

    assemble_features -> compute_sector_strength -> apply_filters
                      -> score_and_rank          -> (tidy result)

- :func:`assemble_features` fans the universe out through the
  :class:`~screener.provider.DataProvider`, turning each ticker's price frame into
  the 21 :func:`~screener.indicators.snapshot` keys, joins the fundamentals
  columns, and adds the engine's DERIVED features (the swing band logic baked into
  higher-is-better ``[0, 1]`` scores, plus the earnings-window flag). One row per
  ticker, indexed by ``symbol``.
- :func:`compute_sector_strength` ranks the GICS sectors by median 3-month member
  return (DECISIONS.md: top-3 = "leading") and writes the per-row sector columns
  the swing filter and signal consume — over the FULL universe, before filtering.
- :func:`apply_filters` drops any row failing ANY of a profile's hard cutoffs; a
  missing/``NaN``/``None`` value FAILS CLOSED.
- :func:`score_and_rank` is the heart and the unit-tested PURE core: each signal
  becomes a cross-sectional percentile in ``[0, 1]`` (direction-adjusted, missing
  -> 0.5 neutral), weighted by the normalized weight and summed into ``score``;
  rows are sorted, ranked, and given a per-row ``reasons`` breakdown.
- :func:`run_screen` wires it together with sensible defaults
  (:func:`~screener.universe.load_universe` + :class:`YFinanceProvider`) and never
  raises on a bad/empty ticker.

Fail-soft everywhere (DECISIONS.md "provider fails soft, per ticker"): a ticker
whose price frame is empty yields ``NaN``/``None`` features and simply fails its
filters or scores neutral — it never aborts the scan. The scorer touches no
network and no clock, so it is fully deterministic on synthetic frames.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from .indicators import latest_ema_cross, pct_from_ma, snapshot
from .profiles import Profile, get_profile
from .provider import DataProvider, YFinanceProvider
from .universe import load_universe

log = logging.getLogger(__name__)

# The 21 keys snapshot() emits, in order — used to seed an all-NaN/None row for a
# ticker whose price frame is empty (so the column schema is stable regardless).
SNAPSHOT_KEYS = (
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "rel_volume_20", "dist_52w_high",
    "sma_20", "sma_50", "sma_150", "ema_5", "ema_9",
    "price_above_sma_20", "price_above_sma_50", "price_above_sma_150",
    "sma_stacked_20_50_150", "ema_5_9_state", "ema_5_9_event",
)

# Fundamentals columns lifted off the Fundamentals dataclass (symbol is the index).
FUNDAMENTAL_KEYS = (
    "name", "sector", "market_cap", "forward_pe", "trailing_pe",
    "revenue_growth", "earnings_growth",
)

# Earnings-window width (DECISIONS / spec §5): a name is "in the window" when its
# next earnings is within this many calendar days, inclusive, of the as-of date.
EARNINGS_WINDOW_DAYS = 7


# --- derived feature helpers (pure; unit-tested) -------------------------
def pullback_quality(pct_from_ema_10: float, pct_from_ema_20: float) -> float:
    """How "healthy" the latest pullback to the 10/20 EMA looks, in ``[0, 1]``.

    Encodes spec §10 "don't chase the overextended": in an uptrend you want price
    sitting JUST above its short EMAs — a shallow pullback that has held — not
    stretched far above (extended, chase risk) and not below (the trend is
    breaking). We average the close's distance from the 10- and 20-bar EMA
    (:func:`screener.indicators.pct_from_ma`) and score that offset with a
    triangular band centred on a small positive cushion:

        peak at  ``CENTRE = +1.5%``  -> 1.0
        zero at  ``CENTRE - WIDTH``  (``-3.5%``, broken below) and
                 ``CENTRE + WIDTH``  (``+6.5%``, overextended), clamped to ``[0, 1]``

    with ``WIDTH = 5%``. So a name 1–2% above its short EMAs scores near 1; one
    8% above (extended) or 4% below (rolling over) scores 0. ``NaN`` if either
    input is ``NaN`` (the engine then neutralises it to 0.5 in scoring).
    """
    if not _finite(pct_from_ema_10) or not _finite(pct_from_ema_20):
        return float("nan")
    offset = (float(pct_from_ema_10) + float(pct_from_ema_20)) / 2.0
    centre, width = 0.015, 0.05
    score = 1.0 - abs(offset - centre) / width
    return float(min(1.0, max(0.0, score)))


def rsi_health(rsi_14: float) -> float:
    """RSI strength rescaled to ``[0, 1]`` with an OVERBOUGHT penalty above 70.

    Rises with momentum up to the classic overbought line, then fades — a name
    pinned at RSI 80 is strong but extended (chase risk, spec §10), so it should
    NOT outscore one sitting in the healthy 60s. The curve (RSI ``r``):

        r <= 30          -> 0.0                     (weak / oversold)
        30 < r <= 70     -> (r - 30) / 40           (ramps 0 -> 1 across 30..70)
        70 < r <= 100    -> 1.0 - (r - 70) / 30 * 0.5 (decays 1.0 -> 0.5 across 70..100)

    so the peak (1.0) is exactly at RSI 70, RSI 50 reads 0.5, and an overbought
    RSI 100 still floors at 0.5 (extended, not disqualifying). ``NaN`` -> ``NaN``.
    """
    if not _finite(rsi_14):
        return float("nan")
    r = float(rsi_14)
    if r <= 30.0:
        return 0.0
    if r <= 70.0:
        return (r - 30.0) / 40.0
    r = min(r, 100.0)
    return 1.0 - (r - 70.0) / 30.0 * 0.5


def ema_5_9_cross_score(state: str, event: str, bars_since_cross: "int | None") -> float:
    """Freshness/strength of the 5/9 EMA cross as a higher-is-better ``[0, 1]``.

    The swing trigger (spec §5) is a *fresh bullish* 5/9 cross. From
    :class:`~screener.indicators.EmaCrossState`:

    - ``state`` bearish (fast below slow) -> ``0.0`` (no long trigger at all).
    - ``state`` bullish -> a freshness score that is ``1.0`` on the cross bar
      (``event == "up"`` / ``bars_since_cross == 0``) and decays linearly over a
      10-bar horizon: ``max(0.0, 1 - bars_since_cross / 10)``. A bullish state
      whose cross is unknown (``bars_since_cross is None``) or older than 10 bars
      scores a small ``0.1`` floor — still above any bearish name, but well below
      a recent cross.

    Pure: a tuple of the cross fields in, a float out (so it is trivially testable
    without constructing price history).
    """
    if state != "bullish":
        return 0.0
    if bars_since_cross is None:
        return 0.1
    horizon = 10.0
    score = 1.0 - float(bars_since_cross) / horizon
    return float(max(0.1, score)) if score > 0.0 else 0.1


def _finite(value) -> bool:
    """True for a real, finite number (rejects ``None``, ``NaN``, ``inf``, bools-ok)."""
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False


# --- feature assembly ----------------------------------------------------
def _empty_feature_row() -> dict:
    """A fully-NaN/None feature row for a ticker with no usable price history.

    Mirrors :func:`screener.indicators.snapshot` on an empty frame (floats ->
    ``NaN``; the two string fields -> their missing-safe defaults; the stack flag
    -> ``False``) plus the derived columns, so the assembled frame's column schema
    is identical whether or not a ticker had data.
    """
    row = {k: float("nan") for k in SNAPSHOT_KEYS}
    row["price_above_sma_20"] = None
    row["price_above_sma_50"] = None
    row["price_above_sma_150"] = None
    row["sma_stacked_20_50_150"] = False
    row["ema_5_9_state"] = "bearish"
    row["ema_5_9_event"] = "none"
    return row


def _earnings_window(
    earnings_on: "dt.date | None", as_of: dt.date
) -> "tuple[bool, int | None]":
    """``(in_window, days_to_earnings)`` for a next-earnings date vs ``as_of``.

    ``days_to_earnings`` is signed (negative if the date is in the past);
    ``in_window`` is ``True`` only for an UPCOMING report within
    :data:`EARNINGS_WINDOW_DAYS` calendar days, inclusive (``0 <= days <= 7``).
    A missing date -> ``(False, None)``.
    """
    if earnings_on is None:
        return False, None
    days = (earnings_on - as_of).days
    return (0 <= days <= EARNINGS_WINDOW_DAYS), days


def assemble_features(
    universe_df: pd.DataFrame, provider: DataProvider, *, as_of: "dt.date | None" = None
) -> pd.DataFrame:
    """Build the one-row-per-ticker feature matrix the engine ranks.

    For every symbol in ``universe_df`` this pulls ``provider.price_history`` and
    reduces it to the 21 :func:`~screener.indicators.snapshot` keys, joins the
    ``provider.fundamentals`` columns, and computes the derived features:

    - ``pct_from_ema_10`` / ``pct_from_ema_20`` — close vs the 10/20-bar EMA;
    - ``pullback_quality`` / ``rsi_health`` / ``ema_5_9_cross_score`` — the swing
      band logic baked into higher-is-better ``[0, 1]`` scores;
    - ``earnings_in_window`` (bool) and ``days_to_earnings`` (int|None) from
      ``provider.earnings_date`` vs ``as_of`` (default ``date.today()``).

    Indexed by ``symbol``, with ``name``/``sector`` preferred from fundamentals and
    falling back to the universe row. FAIL-SOFT: a ticker whose price frame is
    empty (or whose fetch raised) yields an all-NaN/None feature row — it never
    raises and never aborts the scan. ``as_of`` is injected so the earnings/today
    logic is deterministic in tests.
    """
    as_of = as_of or dt.date.today()
    rows: "list[dict]" = []

    for record in universe_df.to_dict("records"):
        symbol = str(record.get("symbol", "")).strip().upper()
        if not symbol:
            continue

        # --- price-derived snapshot (fail-soft) --------------------------
        try:
            prices = provider.price_history(symbol)
        except Exception as exc:  # provider should fail soft, but never abort here
            log.warning("price_history(%s) raised in assemble_features: %s", symbol, exc)
            prices = None

        if prices is not None and len(prices) > 0:
            snap = snapshot(prices)
            close = prices["close"] if "close" in prices.columns else pd.Series(dtype="float64")
            pct_ema_10 = pct_from_ma(close, 10, "ema")
            pct_ema_20 = pct_from_ma(close, 20, "ema")
            cross = latest_ema_cross(close)
            cross_score = ema_5_9_cross_score(
                cross.state, cross.event, cross.bars_since_cross
            )
        else:
            snap = _empty_feature_row()
            pct_ema_10 = float("nan")
            pct_ema_20 = float("nan")
            cross_score = 0.0

        # --- fundamentals (fail-soft) ------------------------------------
        try:
            fundamentals = provider.fundamentals(symbol)
        except Exception as exc:
            log.warning("fundamentals(%s) raised in assemble_features: %s", symbol, exc)
            fundamentals = None

        fund = fundamentals.to_dict() if fundamentals is not None else {}

        # --- earnings window (fail-soft) ---------------------------------
        try:
            earnings_on = provider.earnings_date(symbol)
        except Exception as exc:
            log.warning("earnings_date(%s) raised in assemble_features: %s", symbol, exc)
            earnings_on = None
        in_window, days_to = _earnings_window(earnings_on, as_of)

        row: dict = {"symbol": symbol}
        row.update(snap)
        for key in FUNDAMENTAL_KEYS:
            row[key] = fund.get(key)
        # Prefer fundamentals name/sector; fall back to the universe row.
        row["name"] = fund.get("name") or _clean(record.get("name"))
        row["sector"] = fund.get("sector") or _clean(record.get("sector"))
        # Derived swing features.
        row["pct_from_ema_10"] = pct_ema_10
        row["pct_from_ema_20"] = pct_ema_20
        row["pullback_quality"] = pullback_quality(pct_ema_10, pct_ema_20)
        row["rsi_health"] = rsi_health(snap["rsi_14"])
        row["ema_5_9_cross_score"] = cross_score
        row["earnings_in_window"] = bool(in_window)
        row["days_to_earnings"] = days_to
        rows.append(row)

    if not rows:
        return pd.DataFrame().rename_axis("symbol")

    features = pd.DataFrame(rows).set_index("symbol")
    # Keep the symbol index UNIQUE: a universe with a literally duplicated symbol,
    # or two rows that collide after the upper()/strip() canonicalization above
    # (e.g. "aapl" and " AAPL "), would otherwise produce a non-unique index that
    # makes the downstream `.at[symbol]` lookups return a Series and crash
    # score_and_rank — breaking the "never raises" contract. First occurrence wins
    # (mirrors provider._normalize_prices' `~index.duplicated` guard).
    features = features[~features.index.duplicated(keep="first")]
    return features


def _clean(value) -> "str | None":
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --- sector strength -----------------------------------------------------
def compute_sector_strength(features_df: pd.DataFrame, *, top_n: int = 3) -> pd.DataFrame:
    """Add the sector-strength columns, ranking GICS sectors by 3-mo member return.

    Per DECISIONS.md the swing "leading sector" is the TOP ``top_n`` sectors by the
    MEDIAN ``momentum_3m`` over their members with a valid value. Adds, in place on
    a copy:

    - ``sector_median_3m``     — that sector's median 3-mo momentum (NaN if none);
    - ``sector_rank``          — 1 = strongest sector (NaN if median undefined);
    - ``sector_strength_score``— the sector median percentile-ranked across sectors
      into ``[0, 1]`` (stronger = higher; NaN when the median is undefined, so the
      scorer later neutralises it to 0.5);
    - ``in_leading_sector``    — ``sector_rank <= top_n`` (``False`` if no rank).

    Computed over the FULL universe BEFORE filtering, because swing's hard filter
    is ``in_leading_sector is_true``. Empty input -> empty frame with these columns
    added. Never raises.
    """
    out = features_df.copy()
    extra_cols = ("sector_median_3m", "sector_rank", "sector_strength_score", "in_leading_sector")
    if out.empty:
        for col in extra_cols:
            out[col] = pd.Series(dtype="float64")
        out["in_leading_sector"] = out["in_leading_sector"].astype("object")
        return out

    sector = out["sector"] if "sector" in out.columns else pd.Series(index=out.index, dtype="object")
    mom_3m = pd.to_numeric(out.get("momentum_3m"), errors="coerce")

    # Median 3-mo momentum per sector over members with a valid value.
    valid = pd.DataFrame({"sector": sector, "m3": mom_3m})
    valid = valid[valid["sector"].notna() & valid["m3"].notna()]
    sector_median = valid.groupby("sector")["m3"].median() if not valid.empty else pd.Series(dtype="float64")

    # Rank sectors (1 = strongest) and percentile-score them in [0, 1].
    sector_rank_map = sector_median.rank(ascending=False, method="min")
    sector_score_map = sector_median.rank(pct=True) if not sector_median.empty else pd.Series(dtype="float64")

    out["sector_median_3m"] = sector.map(sector_median).astype("float64")
    out["sector_rank"] = sector.map(sector_rank_map).astype("float64")
    out["sector_strength_score"] = sector.map(sector_score_map).astype("float64")
    out["in_leading_sector"] = out["sector_rank"].apply(
        lambda r: bool(r <= top_n) if pd.notna(r) else False
    )
    return out


# --- hard filters --------------------------------------------------------
def _passes_filter(value, op: str, threshold) -> bool:
    """Evaluate one ``value op threshold`` cutoff; missing/NaN/None FAILS CLOSED.

    ``"is_true"`` passes only on a Python ``True`` (so a ``None`` price-above-MA
    flag, or a ``False``, fails). The comparison ops coerce ``value`` to float and
    fail closed on anything non-finite or non-numeric.
    """
    if op == "is_true":
        return value is True

    if value is None:
        return False
    # Booleans are valid numeric operands (True == 1), but NaN/None are not.
    if isinstance(value, bool):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(numeric):
            return False

    try:
        thresh = float(threshold)
    except (TypeError, ValueError):
        return False

    if op == ">":
        return numeric > thresh
    if op == ">=":
        return numeric >= thresh
    if op == "<":
        return numeric < thresh
    if op == "<=":
        return numeric <= thresh
    if op == "==":
        return numeric == thresh
    return False


def apply_filters(features_df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """Drop every row that fails ANY of ``profile``'s hard :class:`Filter` cutoffs.

    A missing/``NaN``/``None`` value FAILS CLOSED (DECISIONS.md). A filter on a
    feature column that does not exist drops every row (the requirement can't be
    met). Empty input -> empty frame. Returns a copy; never raises.
    """
    if features_df.empty or not profile.filters:
        return features_df.copy()

    mask = pd.Series(True, index=features_df.index)
    for filt in profile.filters:
        if filt.feature not in features_df.columns:
            log.warning(
                "Filter feature %r absent from features; dropping all rows for profile %r",
                filt.feature, profile.name,
            )
            mask &= False
            continue
        column = features_df[filt.feature]
        mask &= column.map(lambda v: _passes_filter(v, filt.op, filt.threshold))

    return features_df[mask].copy()


# --- scoring + ranking (PURE core) ---------------------------------------
def _percentile(column: pd.Series, direction: str) -> pd.Series:
    """Cross-sectional percentile of ``column`` in ``[0, 1]``, missing -> 0.5.

    Uses ``rank(pct=True)`` over the non-missing values so ``NaN`` does NOT skew
    the ranking (pandas ranks NaN as NaN), then fills NaN with the neutral 0.5.
    For ``direction == "lower"`` the percentile is inverted (``1 - pct``) so the
    smallest raw value scores best. Booleans rank fine (``True`` above ``False``).
    """
    numeric = pd.to_numeric(column, errors="coerce")
    pct = numeric.rank(pct=True)
    if direction == "lower":
        pct = 1.0 - pct
    return pct.fillna(0.5)


def score_and_rank(features_df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """Score, sort, rank, and explain — the PURE, unit-tested core of the engine.

    For each :class:`SignalSpec` the feature column is turned into a cross-sectional
    percentile in ``[0, 1]`` (:func:`_percentile`: direction-adjusted, missing ->
    0.5), multiplied by the NORMALIZED weight (``weight / sum(weights)``) and summed
    across signals into ``score`` (itself in ``[0, 1]``). Rows are sorted by
    ``score`` desc then ``symbol`` asc (deterministic) and given a 1-based int
    ``rank``. A ``reasons`` column carries, per row, an ordered dict
    ``{feature: {"value", "percentile", "contribution"}}`` (``contribution ==
    normwt * percentile``; the contributions sum to ``score``) plus any fired
    profile flags under ``"flags"``.

    Adds the percentile column per signal as ``<feature>_pct``. Empty input -> an
    empty frame carrying the same columns. No network, no clock — pure and
    deterministic, so two runs on identical input are identical.
    """
    signals = profile.signals
    base_cols = list(features_df.columns)
    pct_cols = [f"{s.feature}_pct" for s in signals]

    if features_df.empty:
        out = features_df.copy()
        for col in pct_cols:
            out[col] = pd.Series(dtype="float64")
        out["score"] = pd.Series(dtype="float64")
        out["rank"] = pd.Series(dtype="int64")
        out["reasons"] = pd.Series(dtype="object")
        return out

    total_weight = sum(s.weight for s in signals)
    out = features_df.copy()

    # Per-signal percentile + normalized weight; accumulate the weighted score.
    score = pd.Series(0.0, index=out.index)
    norm_weights: "dict[str, float]" = {}
    percentiles: "dict[str, pd.Series]" = {}
    for spec in signals:
        column = out[spec.feature] if spec.feature in out.columns else pd.Series(np.nan, index=out.index)
        pct = _percentile(column, spec.direction)
        norm_w = (spec.weight / total_weight) if total_weight else 0.0
        norm_weights[spec.feature] = norm_w
        percentiles[spec.feature] = pct
        out[f"{spec.feature}_pct"] = pct
        score = score + pct * norm_w

    out["score"] = score.clip(0.0, 1.0)

    # Build the per-row reason breakdown (ordered like the profile's signals).
    flag_names = list(profile.flags)
    reasons: "list[OrderedDict]" = []
    for symbol in out.index:
        detail: "OrderedDict[str, object]" = OrderedDict()
        for spec in signals:
            raw = features_df.at[symbol, spec.feature] if spec.feature in features_df.columns else float("nan")
            pct_val = float(percentiles[spec.feature].at[symbol])
            detail[spec.feature] = {
                "value": _scalarize(raw),
                "percentile": pct_val,
                "contribution": norm_weights[spec.feature] * pct_val,
            }
        if flag_names:
            detail["flags"] = {
                name: _scalarize(features_df.at[symbol, name])
                for name in flag_names
                if name in features_df.columns
            }
        reasons.append(detail)
    out["reasons"] = reasons

    # Deterministic order: score desc, then symbol (the index) asc. Then 1-based
    # rank. We sort on a frame whose index IS the symbol, so a stable secondary
    # sort by that index breaks score ties alphabetically.
    out = out.sort_index(kind="mergesort")  # symbol asc first (the tie-break)
    out = out.sort_values(by="score", ascending=False, kind="mergesort")  # stable: keeps symbol order within a tie
    out["rank"] = range(1, len(out) + 1)
    return out


def _scalarize(value):
    """Plain Python scalar for the reasons dict (NumPy types -> native; NaN kept)."""
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


# --- end-to-end driver ---------------------------------------------------
def run_screen(
    profile: "Profile | str",
    universe_df: "pd.DataFrame | None" = None,
    provider: "DataProvider | None" = None,
    *,
    as_of: "dt.date | None" = None,
) -> pd.DataFrame:
    """Run a profile over a universe and return a tidy, ranked result frame.

    ``profile`` may be a :class:`Profile` or a registry name (both accepted).
    Defaults: ``universe_df = load_universe()``, ``provider = YFinanceProvider()``.
    Pipeline: :func:`assemble_features` -> :func:`compute_sector_strength` ->
    :func:`apply_filters` -> :func:`score_and_rank`. The result has ``symbol`` as a
    COLUMN (not the index) and at least ``symbol, name, sector, score, rank``, the
    scored signal columns, ``reasons``, and any profile flag columns; it is sorted
    by ``rank``.

    NEVER raises on a bad/empty ticker (the provider/feature layer fails soft). If
    every row is filtered out (or the universe is empty) it logs an informative
    message and returns the empty-but-well-formed frame.
    """
    if isinstance(profile, str):
        profile = get_profile(profile)
    if universe_df is None:
        universe_df = load_universe()
    if provider is None:
        provider = YFinanceProvider()

    features = assemble_features(universe_df, provider, as_of=as_of)
    features = compute_sector_strength(features)

    if features.empty:
        log.info("run_screen(%s): empty universe -> empty result.", profile.name)
        return _tidy(score_and_rank(features, profile), profile)

    filtered = apply_filters(features, profile)
    if filtered.empty:
        log.info(
            "run_screen(%s): all %d candidates filtered out -> empty result.",
            profile.name, len(features),
        )
        return _tidy(score_and_rank(filtered, profile), profile)

    ranked = score_and_rank(filtered, profile)
    return _tidy(ranked, profile)


# Leading, human-facing columns; the rest (signal pcts, reasons, flags) follow.
_LEAD_COLUMNS = ("symbol", "name", "sector", "score", "rank")


def _tidy(ranked: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """Move ``symbol`` to a column and order columns lead-first; sort by ``rank``.

    Guarantees the presence of the lead columns plus the scored signal columns,
    ``reasons``, and any profile flag columns even on an empty result, so callers
    (and M5) see a stable schema.
    """
    out = ranked.reset_index()  # 'symbol' index -> column
    if "symbol" not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index": "symbol"})

    # Guarantee a STABLE empty-result schema: on a wholly empty universe the
    # feature frame never materialized name/sector, whereas an all-filtered-out
    # result keeps them. Seed any missing lead column so callers (M5) always see
    # at least symbol/name/sector/score/rank regardless of which empty path ran.
    for col in _LEAD_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")

    signal_features = [s.feature for s in profile.signals]
    flag_cols = [f for f in profile.flags if f in out.columns]
    pct_cols = [f"{f}_pct" for f in signal_features if f"{f}_pct" in out.columns]

    ordered: "list[str]" = []
    for col in _LEAD_COLUMNS:
        if col in out.columns and col not in ordered:
            ordered.append(col)
    for col in signal_features + pct_cols + flag_cols + ["reasons"]:
        if col in out.columns and col not in ordered:
            ordered.append(col)
    # Append any remaining columns (deterministically) so nothing is silently lost.
    for col in out.columns:
        if col not in ordered:
            ordered.append(col)

    out = out[ordered]
    if "rank" in out.columns and len(out) > 0:
        out = out.sort_values("rank", kind="mergesort").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


if __name__ == "__main__":  # smoke test: live screen over a small real sample
    import sys

    logging.basicConfig(level=logging.WARNING)
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "momentum"

    universe = load_universe().head(25)
    provider = YFinanceProvider()
    result = run_screen(profile_name, universe, provider)

    print(f"profile={profile_name!r}  universe={len(universe)}  matched={len(result)}")
    if not result.empty:
        signal_features = [s.feature for s in get_profile(profile_name).signals]
        cols = [c for c in ("symbol", "score", "rank", *signal_features[:2]) if c in result.columns]
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(result.head(15)[cols].to_string(index=False))
    else:
        print("(no names matched the profile's hard filters)")
