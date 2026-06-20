"""The results-table pipeline: filters, column selection, config spec, export.

The dashboard's tabular core: the universe-size gate, the sidebar filter pipeline,
the synthetic derived columns (links / fit / why), the density-aware column
ordering and selection, the PURE per-column config descriptor (``app.py`` turns it
into real ``st.column_config`` objects), the CSV export frame, selection
reconciliation, and the matches / empty-state captions. Pandas/numpy/stdlib +
:class:`screener.profiles.Profile` only — never streamlit. Re-exported by
:mod:`screener.display`.
"""

from __future__ import annotations

import pandas as pd

from ..profiles import Profile
from ._base import (
    _BOOL_FEATURES,
    _DERIVED_01_FEATURES,
    _PE_FEATURES,
    _PERCENT_FEATURES,
    _is_missing,
)
from .features import feature_description, feature_label, fit_score
from .reasons import narrative_series
from .text import (
    BUY_ZONE_HELP,
    EXTENSION_HELP,
    SCORE_HELP,
    WHY_HELP,
    tradingview_url,
    yahoo_url,
)


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
    extended_hidden: bool = False,
    in_buy_zone_only: bool = False,
) -> pd.DataFrame:
    """Apply the sidebar filters to the cached result, purely in pandas.

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
    - ``extended_hidden`` — when ``True``, DROP rows whose ``extension_state`` is
      ``"parabolic"`` (the steep/overextended names). Only ``"parabolic"`` is hidden;
      ``"extended"`` and ``"normal"`` stay. No-op when the column is absent; a missing
      / ``NaN`` state is treated as non-parabolic (kept), matching the engine's
      ``"normal"`` fail-soft baseline.
    - ``in_buy_zone_only`` — when ``True``, KEEP only rows whose ``in_buy_zone`` is
      truthy. No-op when the column is absent; a missing / ``NaN`` flag is treated as
      ``False`` (dropped), matching the engine's fail-soft baseline.

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

    # Hide overextended (parabolic) names. Only the "parabolic" bucket is dropped;
    # a missing/NaN state is non-parabolic (kept), matching the "normal" baseline.
    if extended_hidden and "extension_state" in df.columns:
        state = df["extension_state"].astype("object")
        is_parabolic = state.apply(
            lambda s: (not _is_missing(s)) and str(s).strip().lower() == "parabolic"
        )
        mask &= ~is_parabolic

    # Keep only rows currently inside the buy zone (NaN/absent flag -> dropped).
    # Elementwise so a mixed object column (True/False/NaN) coerces cleanly without
    # a pandas object-downcast warning; a missing flag is treated as False.
    if in_buy_zone_only and "in_buy_zone" in df.columns:
        in_zone = df["in_buy_zone"].apply(lambda v: (not _is_missing(v)) and bool(v))
        mask &= in_zone

    return df[mask].reset_index(drop=True)


# --- per-ticker external links (pure, derived from `symbol`) -------------
# Synthetic URL columns built from the ticker symbol so the results table can
# jump straight out to a chart/quote. Equities-only and no exchange field in the
# model — both sites resolve bare US large-cap symbols. Class shares differ by
# separator: yfinance/Yahoo use "-" (BRK-B), TradingView uses "." (BRK.B).
_LINK_COLUMNS = ("tv_url", "yf_url")  # inserted right after the `fit` column in column_order


def _with_derived_columns(df: pd.DataFrame, profile=None) -> pd.DataFrame:
    """Return a copy of ``df`` with the synthetic display-only columns added.

    - ``tv_url`` / ``yf_url`` (per-ticker external links) derived from ``symbol``;
    - ``fit`` (0..100 headline number) derived from ``score`` via :func:`fit_score`;
    - ``why`` (per-row plain-English narrative) derived from ``reasons`` when a
      ``profile`` is given, via :func:`narrative_series`.

    Fail-soft: each column is added only when its source column is present, so a
    degenerate frame (no ``symbol`` / ``score`` / ``reasons``) is returned without
    that column and :func:`table_view` can never raise.
    """
    out = df.copy()
    if "symbol" in out.columns:
        out["tv_url"] = out["symbol"].map(tradingview_url)
        out["yf_url"] = out["symbol"].map(yahoo_url)
    if "score" in out.columns:
        out["fit"] = out["score"].map(fit_score)
    if profile is not None and "reasons" in out.columns:
        out["why"] = narrative_series(out, profile)
    return out


# The leading, human-ordered columns shown before a profile's own signals. The
# synthetic ``fit`` (0..100, derived from ``score``) is inserted right after these,
# taking the visible score slot; the raw ``score`` stays in the frame for filtering.
_LEAD_VISIBLE = ("rank", "symbol", "name", "sector")


# --- table column selection ---------------------------------------------
def column_order(profile: Profile, df: pd.DataFrame, *, density: str = "compact") -> "list[str]":
    """Ordered visible-column names for the results table, by display ``density``.

    Shared prefix (both densities): lead columns (``rank, symbol, name, sector``),
    then the synthetic ``fit`` (0..100, taking the visible score slot), the two
    synthetic per-ticker link columns (``tv_url``, ``yf_url``), then the headline
    ``price`` and ``change_pct`` right after the link block.

    - ``density="compact"`` (default, finance-site lean view): the shared prefix,
      then the two universe-wide tactical readouts (``extension_state``,
      ``in_buy_zone``) — and that's it. No per-profile signal columns.
    - ``density="detailed"``: the shared prefix, then the profile's RAW signal
      feature columns, then — SWING ONLY (flag gate + column present) —
      ``earnings_in_window`` / ``days_to_earnings``, then the same two tactical
      readouts.

    The synthetic ``why`` narrative column is ALWAYS last. Columns are intersected
    with ``df`` (a missing column is skipped, never a KeyError). NEVER includes
    ``reasons`` or any ``*_pct`` percentile column. This is the single source of
    truth — including the two tactical columns, which were previously appended in
    ``app.py`` — for both :func:`table_view` and the grid's ``column_order``.
    """
    cols = [c for c in _LEAD_VISIBLE if df is None or c in df.columns]

    # Synthetic `fit` (from `score`) takes the visible score slot, right after the
    # identity block; table_view augments the frame with it before selecting.
    if (df is None or "score" in df.columns) and "fit" not in cols:
        cols.append("fit")

    # Per-ticker external-link columns sit right after the fit/identity block.
    # Synthesised from `symbol`, so gate on it.
    if df is None or "symbol" in df.columns:
        cols += [c for c in _LINK_COLUMNS if c not in cols]

    seen = set(cols)

    # Headline price scalars sit right after the link block, in BOTH densities.
    for extra in ("price", "change_pct"):
        if extra not in seen and (df is None or extra in df.columns):
            cols.append(extra)
            seen.add(extra)

    # Detailed-only: reveal the profile's signal columns (+ swing earnings).
    if density == "detailed":
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

    # Universe-wide tactical readouts in BOTH densities (single source of truth).
    for extra in ("extension_state", "in_buy_zone"):
        if extra not in seen and (df is None or extra in df.columns):
            cols.append(extra)
            seen.add(extra)

    # Synthetic per-row narrative LAST (from `reasons`); kept at the end so the
    # long text never crowds the numeric columns.
    if (df is None or "reasons" in df.columns) and "why" not in seen:
        cols.append("why")
        seen.add("why")
    return cols


def table_view(df: pd.DataFrame, profile: Profile, *, density: str = "compact") -> pd.DataFrame:
    """A NEW frame with only the curated, human-ordered scalar columns.

    Selects exactly :func:`column_order` (at the given ``density``) from ``df`` —
    augmented with the synthetic ``tv_url``/``yf_url`` link columns (so it can never
    include ``reasons`` or a ``*_pct`` column) — and returns a copy. Fail-soft: a
    result frame missing one of the profile's signal columns still returns the
    intersection without raising.
    """
    order = column_order(profile, df, density=density)
    src = _with_derived_columns(df, profile) if df is not None else df
    cols = [c for c in order if df is not None and c in src.columns]
    return src[cols].copy()


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
        "rank": {"kind": "number", "label": "Rank", "format": "%d",
                 "help": "Position in the ranked list (1 = best match for this profile)."},
        "symbol": {"kind": "text", "label": "Symbol"},
        "name": {"kind": "text", "label": "Name"},
        "sector": {"kind": "text", "label": "Sector"},
        "score": {"kind": "progress", "label": "Score", "format": "%.3f", "min": 0.0, "max": 1.0,
                  "help": SCORE_HELP},
        # The headline 0..100 "fit" number that takes the visible score slot (the
        # raw ``score`` descriptor is kept above for the filters / any future use).
        "fit": {"kind": "progress", "label": "Fit", "format": "%d", "min": 0.0, "max": 100.0,
                "help": SCORE_HELP},
        # Per-ticker jump-out links (icon-first: a single "↗"; the header + the
        # hovered URL name the destination). Equities-only, opens in a new tab.
        "tv_url": {"kind": "link", "label": "TradingView", "display_text": "↗",
                   "help": "Open this ticker's interactive chart on TradingView (new tab)."},
        "yf_url": {"kind": "link", "label": "Yahoo", "display_text": "↗",
                   "help": "Open this ticker's Yahoo Finance quote page (new tab)."},
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
        # Every signal column carries its plain-English definition as a header tooltip.
        spec[feat]["help"] = feature_description(feat)

    # Swing-only earnings columns.
    if "earnings_in_window" in profile.flags:
        spec["earnings_in_window"] = {"kind": "checkbox", "label": "Earnings ≤7d"}
        spec["days_to_earnings"] = {"kind": "number", "label": "Days To Earnings", "format": "%d"}

    # The synthetic per-row narrative column (plain-English strengths/weaknesses).
    spec["why"] = {"kind": "text", "label": "Why", "help": WHY_HELP}
    # Universe-wide tactical-readout columns (present for every profile). The
    # Extension cell renders as text (app.py colours the badge via
    # extension_state_color); In Buy Zone is a checkbox like the earnings flags.
    spec["extension_state"] = {
        "kind": "text", "label": "Extension", "help": EXTENSION_HELP,
    }
    spec["in_buy_zone"] = {
        "kind": "checkbox", "label": "In Buy Zone", "help": BUY_ZONE_HELP,
    }
    # Headline price columns (present for every profile, both densities). `price`
    # is a plain number app.py renders with a leading "$"; `change_pct` is a SIGNED
    # percent app.py colours green/red. Both carry their feature-map tooltip.
    spec["price"] = {
        "kind": "number", "label": feature_label("price"), "format": "%.2f",
        "help": feature_description("price"),
    }
    spec["change_pct"] = {
        "kind": "percent", "label": feature_label("change_pct"), "format": "percent",
        "signed": True, "help": feature_description("change_pct"),
    }
    return spec


def export_frame(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """A human-readable frame for CSV download (no link-URL / internal columns).

    Reuses :func:`table_view` at the DETAILED density — so the CSV always carries
    ``fit``, the headline price columns, the profile's signals, any swing earnings
    columns, the tactical readouts, and the ``why`` narrative regardless of the
    on-screen density toggle (the download is for downstream analysis, so it stays
    the full picture) — and drops the ``tv_url`` / ``yf_url`` link columns. Numbers
    kept raw. Fail-soft on a degenerate frame (returns the intersection).
    """
    view = table_view(df, profile, density="detailed")
    drop = [c for c in _LINK_COLUMNS if c in view.columns]
    return view.drop(columns=drop) if drop else view


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


# --- captions / empty messages -------------------------------------------
def filter_summary(n_shown: int, n_total: int) -> str:
    """``"Showing {n_shown} of {n_total} matches"``."""
    return f"Showing {int(n_shown)} of {int(n_total)} matches"


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
