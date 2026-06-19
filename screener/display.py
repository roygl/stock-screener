"""Pure, network-free display logic for the M5 Streamlit dashboard.

The dashboard (``app.py``) is *thin*: it owns the Streamlit widget calls and the
session-state orchestration, and nothing else. Everything that is testable
without a browser — filtering the cached result, choosing and formatting the
visible table columns, turning the engine's per-row ``reasons`` OrderedDict into
a tidy frame, the earnings badge / summary strings, selection reconciliation,
and every empty/context message — lives HERE so it can be unit-tested offline.

Hard rule (house style + the M5 spec): this module imports ONLY pandas, numpy,
the stdlib, and :class:`screener.profiles.Profile`. It NEVER imports streamlit.
The one purity boundary is :func:`column_config_spec`, which returns a PLAIN
descriptor dict; ``app.py`` turns that into real ``st.column_config`` objects
(the only place ``st.column_config.*`` can be built).

All helpers take plain pandas/numpy/``Profile`` inputs and return plain
frames/dicts/strings, and tolerate the engine's fail-soft ``NaN``/``None`` cells
(verified against the engine: ``days_to_earnings`` is float64 with ``NaN``;
``score`` float64 0..1; ``rank`` int64; ``reasons`` an ordered dict whose
per-signal contributions sum exactly to ``score``).
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pandas as pd

from .profiles import Profile

# --- module-level copy ---------------------------------------------------
# Short disclaimer rendered in EVERY app state (the sidebar), satisfying the
# spec §9/§10 "describes & ranks, never advises" outcome regardless of scan
# state. The fuller text lives in app.py's disclaimer expander.
DISCLAIMER_TEXT = (
    "This tool describes and ranks US large-cap equities from end-of-day data. "
    "It is not financial advice and gives no buy/sell recommendations."
)

DISCLAIMER_DETAIL = (
    "This screener is an educational research aid, not investment advice. It "
    "ranks and describes stocks from delayed, end-of-day data using mechanical "
    "rules — it does not know your goals, risk tolerance, or circumstances, and "
    "it makes no buy, sell, or hold recommendation. Signals can be wrong or "
    "stale, and past behaviour does not predict future returns. Do your own "
    "research and consult a licensed professional before investing."
)

# Human labels for every feature that can surface in the table or the reasons
# panel. Covers all scored signals across the three profiles plus a few raw
# features the reason breakdown can show; .get(...) with a title-case fallback
# means an unknown key can never KeyError.
FEATURE_LABELS: "dict[str, str]" = {
    # momentum
    "momentum_1m": "1M Momentum",
    "momentum_3m": "3M Momentum",
    "momentum_6m": "6M Momentum",
    "momentum_12m": "12M Momentum",
    # trend / structure
    "sma_stacked_20_50_150": "Trend Stacked (20>50>150)",
    "dist_52w_high": "Distance From 52W High",
    # valuation / growth
    "forward_pe": "Forward P/E",
    "trailing_pe": "Trailing P/E",
    "revenue_growth": "Revenue Growth",
    "earnings_growth": "Earnings Growth",
    # swing derived / flow
    "ema_5_9_cross_score": "5/9 EMA Cross",
    "rel_volume_20": "Relative Volume (20d)",
    "macd_hist": "MACD Histogram",
    "rsi_14": "RSI (14)",
    "rsi_health": "RSI Health",
    "pullback_quality": "Pullback Quality",
    "sector_strength_score": "Sector Strength",
}

# Feature buckets that drive type-aware formatting and the column-config kind.
# Fractions in the engine (0.12 == 12%); momentum_12m can exceed 1.0.
_PERCENT_FEATURES = frozenset(
    {
        "momentum_1m",
        "momentum_3m",
        "momentum_6m",
        "momentum_12m",
        "revenue_growth",
        "earnings_growth",
        "dist_52w_high",
    }
)
# Raw P/E ratios (one decimal, no % or × suffix).
_PE_FEATURES = frozenset({"forward_pe", "trailing_pe"})
# Engine-derived scores already squeezed into [0, 1] (two decimals / progress bar).
_DERIVED_01_FEATURES = frozenset(
    {"ema_5_9_cross_score", "pullback_quality", "rsi_health", "sector_strength_score"}
)
# Boolean features.
_BOOL_FEATURES = frozenset({"sma_stacked_20_50_150"})

# The leading, human-ordered columns shown before a profile's own signals.
_LEAD_VISIBLE = ("rank", "symbol", "name", "sector", "score")

_MISSING = "—"


# --- small numeric helpers ----------------------------------------------
def _is_missing(value) -> bool:
    """True for ``None``/``NaN``/non-finite — the engine's fail-soft sentinels."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        # numpy floats, pandas NA, etc.
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def feature_label(feature: str) -> str:
    """Human label for ``feature`` (explicit map, title-case fallback, no KeyError)."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


# --- universe-size guard -------------------------------------------------
def take_universe_slice(universe: pd.DataFrame, n: int) -> pd.DataFrame:
    """First ``n`` rows of ``universe``, clamped to ``1..len`` — the size gate.

    The only universe-size gate before a scan: ``n`` larger than the universe
    returns the whole thing; ``n < 1`` still returns one row (so a scan always
    has something to do). Equivalent to ``universe.head(n)`` with the bounds
    enforced.
    """
    if universe is None or len(universe) == 0:
        return universe.head(0) if universe is not None else pd.DataFrame()
    bounded = max(1, min(int(n), len(universe)))
    return universe.head(bounded)


def universe_size_hint(n: int) -> str:
    """Plain-English cold/warm ETA bucket for scanning ``n`` names.

    A cold scan hits Yahoo ~3× per name (prices + fundamentals + earnings), so
    the cold estimate scales with ``n`` while a warm (cached) re-run is near
    instant. Buckets are deliberately coarse — they are a guardrail, not a
    promise.
    """
    n = int(n)
    if n <= 25:
        cold = "~1-2 min cold"
    elif n <= 50:
        cold = "~2-4 min cold"
    elif n <= 100:
        cold = "~4-8 min cold"
    else:
        cold = "many minutes cold"
    return f"~{n} names: a few seconds warm, {cold} (first run of the day)."


# --- empty / state checks ------------------------------------------------
def is_empty_result(df: pd.DataFrame) -> bool:
    """True when the engine returned zero rows (tolerant of any column set)."""
    return df is None or len(df) == 0


def sector_options(df: pd.DataFrame) -> "list[str]":
    """Sorted, unique, non-null ``sector`` values (``[]`` if absent/empty)."""
    if df is None or "sector" not in df.columns or len(df) == 0:
        return []
    sectors = df["sector"].dropna().astype(str)
    sectors = sectors[sectors.str.strip() != ""]
    return sorted(sectors.unique().tolist())


# --- the filter pipeline -------------------------------------------------
def _swing_earnings_enabled(profile: Profile, df: pd.DataFrame) -> bool:
    """True only for a swing-style profile with the earnings column present.

    Gated on ``"earnings_in_window" in profile.flags`` (the swing flag), NOT on
    mere column presence — the column exists for all three profiles in a
    non-empty result, so a presence check would wrongly enable swing-only UI for
    momentum/long_term. The column-presence check is the *additional* guard for
    the wholly-empty-universe frame, which drops it.
    """
    return "earnings_in_window" in profile.flags and "earnings_in_window" in df.columns


def apply_filters(
    df: pd.DataFrame,
    *,
    text: str,
    sectors: "list[str]",
    min_score: float,
    earnings_only: bool,
    profile: Profile,
) -> pd.DataFrame:
    """Apply the four sidebar filters to the cached result, purely in pandas.

    Composes (all NaN-safe, all passthrough when "empty"):

    - ``text`` — case-insensitive substring over ``symbol`` + ``name`` (empty
      string keeps everything; a ``NaN`` name simply never matches).
    - ``sectors`` — membership in the chosen sectors (empty list keeps all).
    - ``min_score`` — keep ``score >= min_score``; a ``NaN`` score is kept ONLY
      when the floor is ``0.0`` (so the default never hides fail-soft rows but a
      raised floor does).
    - ``earnings_only`` — SWING ONLY (gated via :func:`_swing_earnings_enabled`):
      keep rows whose ``earnings_in_window`` is ``True``. Ignored for non-swing
      profiles even if the column happens to exist.

    Returns a NEW frame with a fresh ``RangeIndex`` (``reset_index(drop=True)``)
    so positional row-selection from ``st.dataframe`` maps back to a stable
    position. Never mutates the input; never raises.
    """
    if df is None or len(df) == 0:
        return df.copy() if df is not None else pd.DataFrame()

    mask = pd.Series(True, index=df.index)

    # Text: case-insensitive substring over symbol + name.
    needle = (text or "").strip().lower()
    if needle:
        sym = df["symbol"].astype(str).str.lower() if "symbol" in df.columns else pd.Series("", index=df.index)
        name = df["name"].fillna("").astype(str).str.lower() if "name" in df.columns else pd.Series("", index=df.index)
        mask &= sym.str.contains(needle, regex=False) | name.str.contains(needle, regex=False)

    # Sector membership (empty selection = all).
    if sectors:
        if "sector" in df.columns:
            mask &= df["sector"].isin(sectors)
        else:
            mask &= False

    # Minimum score floor.
    floor = float(min_score)
    if "score" in df.columns:
        score = pd.to_numeric(df["score"], errors="coerce")
        if floor <= 0.0:
            # Keep NaN scores at the default floor; otherwise drop only below-floor.
            mask &= score.isna() | (score >= floor)
        else:
            mask &= score >= floor  # NaN >= floor is False -> dropped
    elif floor > 0.0:
        mask &= False

    # Swing-only earnings-in-window filter.
    if earnings_only and _swing_earnings_enabled(profile, df):
        mask &= df["earnings_in_window"].fillna(False).astype(bool)

    return df[mask].reset_index(drop=True)


# --- table column selection ---------------------------------------------
def column_order(profile: Profile, df: pd.DataFrame) -> "list[str]":
    """Ordered visible-column names for the results table.

    Lead columns (``rank, symbol, name, sector, score``) then the profile's RAW
    signal feature columns (intersected with ``df`` so a missing column is just
    skipped, never a KeyError), then — SWING ONLY (flag gate + column present) —
    ``earnings_in_window`` and ``days_to_earnings``. NEVER includes ``reasons``
    or any ``*_pct`` percentile column (those are Arrow-noisy / internal). This
    is the single source of truth for both :func:`table_view` and the
    ``column_order=`` arg passed to ``st.dataframe``.
    """
    cols = [c for c in _LEAD_VISIBLE if df is None or c in df.columns]

    seen = set(cols)
    for spec in profile.signals:
        feat = spec.feature
        if feat in seen:
            continue
        if df is None or feat in df.columns:
            cols.append(feat)
            seen.add(feat)

    if _swing_earnings_enabled(profile, df if df is not None else pd.DataFrame()):
        for extra in ("earnings_in_window", "days_to_earnings"):
            if extra not in seen and (df is None or extra in df.columns):
                cols.append(extra)
                seen.add(extra)
    return cols


def table_view(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """A NEW frame with only the curated, human-ordered scalar columns.

    Selects exactly :func:`column_order` from ``df`` (so it can never include
    ``reasons`` or a ``*_pct`` column) and returns a copy. Fail-soft: a result
    frame missing one of the profile's signal columns still returns the
    intersection without raising.
    """
    cols = column_order(profile, df)
    cols = [c for c in cols if df is not None and c in df.columns]
    return df[cols].copy()


def column_config_spec(profile: Profile) -> "dict[str, dict]":
    """PURE per-column descriptor dict (no streamlit types).

    Maps each visible column to ``{"kind", "label", "format"?, "min"?, "max"?}``
    where ``kind`` is one of ``progress`` / ``number`` / ``percent`` /
    ``checkbox`` / ``text``. ``app.py`` converts each descriptor into the real
    ``st.column_config.*`` object (the one place streamlit may be imported). The
    test asserts on this plain dict, so the formatting contract is verifiable
    without a browser.
    """
    spec: "dict[str, dict]" = {
        "rank": {"kind": "number", "label": "Rank", "format": "%d"},
        "symbol": {"kind": "text", "label": "Symbol"},
        "name": {"kind": "text", "label": "Name"},
        "sector": {"kind": "text", "label": "Sector"},
        "score": {"kind": "progress", "label": "Score", "format": "%.3f", "min": 0.0, "max": 1.0},
    }

    for s in profile.signals:
        feat = s.feature
        label = feature_label(feat)
        if feat in _PERCENT_FEATURES:
            # "percent" is the st.column_config preset (NOT a printf "%.1f%%"):
            # it multiplies the engine's fraction by 100 for display (0.12 ->
            # "12.00%"). app.py passes this straight through to NumberColumn.
            spec[feat] = {"kind": "percent", "label": label, "format": "percent"}
        elif feat in _PE_FEATURES:
            spec[feat] = {"kind": "number", "label": label, "format": "%.1f"}
        elif feat == "rel_volume_20":
            spec[feat] = {"kind": "number", "label": label, "format": "%.2f"}
        elif feat == "rsi_14":
            spec[feat] = {"kind": "number", "label": label, "format": "%.0f"}
        elif feat == "macd_hist":
            spec[feat] = {"kind": "number", "label": label, "format": "%.3f"}
        elif feat in _DERIVED_01_FEATURES:
            spec[feat] = {"kind": "progress", "label": label, "format": "%.2f", "min": 0.0, "max": 1.0}
        elif feat in _BOOL_FEATURES:
            spec[feat] = {"kind": "checkbox", "label": label}
        else:
            spec[feat] = {"kind": "number", "label": label, "format": "%.2f"}

    # Swing-only earnings columns.
    if "earnings_in_window" in profile.flags:
        spec["earnings_in_window"] = {"kind": "checkbox", "label": "Earnings ≤7d"}
        spec["days_to_earnings"] = {"kind": "number", "label": "Days To Earnings", "format": "%d"}
    return spec


# --- value formatting ----------------------------------------------------
def format_value(feature: str, value) -> str:
    """Type-aware string for one feature value (fail-soft on ``NaN``/``None``).

    - percent-style fractions (momentum_*, revenue/earnings_growth,
      dist_52w_high) -> ``f"{v*100:.1f}%"``;
    - ``forward_pe`` / ``trailing_pe`` -> ``f"{v:.1f}"``;
    - ``rel_volume_20`` -> ``f"{v:.2f}×"``;
    - ``rsi_14`` -> ``f"{v:.0f}"``; ``macd_hist`` -> ``f"{v:.3f}"``;
    - derived [0, 1] scores (ema_5_9_cross_score, pullback_quality, rsi_health,
      sector_strength_score) -> ``f"{v:.2f}"``;
    - bool ``sma_stacked_20_50_150`` -> ``"Yes"`` / ``"No"``;
    - missing -> ``"—"``.
    """
    if feature in _BOOL_FEATURES:
        if _is_missing(value):
            return _MISSING
        return "Yes" if bool(value) else "No"

    if _is_missing(value):
        return _MISSING

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if feature in _PERCENT_FEATURES:
        return f"{v * 100:.1f}%"
    if feature in _PE_FEATURES:
        return f"{v:.1f}"
    if feature == "rel_volume_20":
        return f"{v:.2f}×"
    if feature == "rsi_14":
        return f"{v:.0f}"
    if feature == "macd_hist":
        return f"{v:.3f}"
    if feature in _DERIVED_01_FEATURES:
        return f"{v:.2f}"
    # Generic numeric fallback.
    return f"{v:.2f}"


# --- the "why it ranks" reasons table ------------------------------------
def _signal_items(reasons) -> "list[tuple[str, dict]]":
    """The reasons OrderedDict's signal entries (the ``"flags"`` key excluded).

    Preserves insertion order — the engine writes signals in the profile's
    signal order and the engine test asserts it, so we never re-sort.
    """
    if not reasons:
        return []
    return [(k, v) for k, v in reasons.items() if k != "flags" and isinstance(v, dict)]


def reasons_to_frame(reasons, profile: Profile) -> pd.DataFrame:
    """Tidy the per-row ``reasons`` OrderedDict into a display frame.

    Columns ``Signal`` (humanized), ``Value`` (via :func:`format_value`),
    ``Percentile`` (float 0..1), ``Contribution`` (float 0..1), in the
    OrderedDict's signal order, excluding the ``"flags"`` key. The numeric
    Percentile/Contribution stay numeric so ``app.py`` can render them as
    progress bars; ``Value`` is the pre-formatted string. Tolerant of
    ``NaN``/``None`` values and an empty/``None`` ``reasons`` (-> empty frame
    with the right columns).
    """
    columns = ["Signal", "Value", "Percentile", "Contribution"]
    items = _signal_items(reasons)
    if not items:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    rows = []
    for feat, entry in items:
        pct = entry.get("percentile")
        contrib = entry.get("contribution")
        rows.append(
            {
                "Signal": feature_label(feat),
                "Value": format_value(feat, entry.get("value")),
                "Percentile": float(pct) if not _is_missing(pct) else float("nan"),
                "Contribution": float(contrib) if not _is_missing(contrib) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def max_contribution(reasons) -> float:
    """Largest signal contribution (excl ``flags``), clamped strictly ``> 0``.

    Used as the max of the Contribution progress bar so small contributions are
    still visible (a hardcoded 1.0 max would make them look empty). Defaults to a
    small positive number when there are no contributions or all are zero, so the
    progress column never divides by zero.
    """
    items = _signal_items(reasons)
    best = 0.0
    for _, entry in items:
        c = entry.get("contribution")
        if not _is_missing(c):
            c = float(c)
            if c > best:
                best = c
    return best if best > 0.0 else 0.01


def contribution_caption(reasons, score: float) -> str:
    """Assert-the-math caption: per-signal contributions sum to the score.

    e.g. ``"Signal contributions sum to the score (0.719 ≈ 0.719)"``. Sums the
    contributions (excl ``flags``); a missing contribution counts as 0.
    """
    items = _signal_items(reasons)
    total = 0.0
    for _, entry in items:
        c = entry.get("contribution")
        if not _is_missing(c):
            total += float(c)
    score_val = 0.0 if _is_missing(score) else float(score)
    return f"Signal contributions sum to the score ({total:.3f} ≈ {score_val:.3f})"


# --- earnings badge / summary -------------------------------------------
def earnings_badge(in_window, days_to_earnings) -> str:
    """Scalar badge string for one row, or ``""`` when not applicable.

    ``"⚠ Earnings in {d}d"`` only when ``in_window`` is truthy AND
    ``days_to_earnings`` is a finite number (guarded so it NEVER calls
    ``int(NaN)`` — ``days_to_earnings`` is float64 with ``NaN`` in the engine).
    Anything else -> ``""``.
    """
    if _is_missing(in_window) or not bool(in_window):
        return ""
    if _is_missing(days_to_earnings):
        return ""
    try:
        days = float(days_to_earnings)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(days):
        return ""
    return f"⚠ Earnings in {int(days)}d"


def earnings_badge_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized :func:`earnings_badge` over a frame (empty strings if absent).

    Returns an all-empty-string Series (indexed like ``df``) when the earnings
    columns are missing — e.g. a non-swing result or the wholly-empty frame.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype="object")
    if "earnings_in_window" not in df.columns or "days_to_earnings" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return pd.Series(
        [
            earnings_badge(w, d)
            for w, d in zip(df["earnings_in_window"], df["days_to_earnings"])
        ],
        index=df.index,
        dtype="object",
    )


def earnings_summary(df: pd.DataFrame) -> "str | None":
    """Banner text counting in-window names, or ``None`` when 0 / column absent.

    e.g. ``"⚠ 1 of 4 names report earnings within 7 days — elevated event
    risk."``. Returns ``None`` (no banner) when the column is missing or no row
    is flagged.
    """
    if df is None or len(df) == 0 or "earnings_in_window" not in df.columns:
        return None
    flagged = df["earnings_in_window"].fillna(False).astype(bool)
    k = int(flagged.sum())
    if k == 0:
        return None
    n = len(df)
    return f"⚠ {k} of {n} names report earnings within 7 days — elevated event risk."


# --- selection reconciliation -------------------------------------------
def row_option_label(view: pd.DataFrame, symbol: str) -> str:
    """``"SYMBOL — Name"`` label for the inspect selectbox option.

    Looks the name up by symbol VALUE in ``view``; falls back to just the symbol
    if the name is missing or the symbol is not present.
    """
    if view is None or "symbol" not in view.columns:
        return str(symbol)
    match = view.loc[view["symbol"] == symbol]
    if match.empty:
        return str(symbol)
    name = match.iloc[0].get("name") if "name" in view.columns else None
    if _is_missing(name) or str(name).strip() == "":
        return str(symbol)
    return f"{symbol} — {name}"


def resolve_selection(
    view: pd.DataFrame, table_click_symbol, selectbox_symbol, prev_symbol
) -> str:
    """Reconcile the two selection inputs into ONE symbol value.

    Precedence: a fresh table click wins; else the selectbox value; else the
    previous session symbol IF still in ``view["symbol"]``; else default to the
    rank-1 row (``view.iloc[0]["symbol"]``). All candidates are validated against
    the CURRENT ``view`` so a stale symbol (filtered out) is ignored. Never
    indexes an empty view — callers guarantee RESULTS state (non-empty view), but
    if somehow empty this returns ``""`` rather than raising.
    """
    if view is None or "symbol" not in view.columns or len(view) == 0:
        return ""
    present = set(view["symbol"].tolist())

    if table_click_symbol is not None and table_click_symbol in present:
        return table_click_symbol
    if selectbox_symbol is not None and selectbox_symbol in present:
        return selectbox_symbol
    if prev_symbol is not None and prev_symbol in present:
        return prev_symbol
    return view.iloc[0]["symbol"]


# --- captions / context / empty messages --------------------------------
def filter_summary(n_shown: int, n_total: int) -> str:
    """``"Showing {n_shown} of {n_total} matches"``."""
    return f"Showing {int(n_shown)} of {int(n_total)} matches"


def scan_context_line(
    profile_label: str, n_names: int, cache_day: str, n_results: int
) -> str:
    """``"<label> · <n> names · as of <day> · <n_results> matches"``."""
    return f"{profile_label} · {int(n_names)} names · as of {cache_day} · {int(n_results)} matches"


def empty_message(profile_label: str, n_names: int) -> str:
    """Engine-empty warning: nothing cleared the hard filters."""
    return (
        f"No {profile_label} names cleared the profile's hard filters in the "
        f"{int(n_names)} scanned. Try a larger universe size or a different profile."
    )


def filtered_empty_message(n_total: int) -> str:
    """Filtered-empty info: filters hid every row of an otherwise non-empty scan."""
    return (
        f"No rows match the current filters — relax them to see all "
        f"{int(n_total)} results."
    )
