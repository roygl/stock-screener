"""The five ``@st.cache_data`` memos (relocated verbatim from app.py).

One engine memo (:func:`run_cached`, the single cached scan) plus four per-symbol
detail memos. The lazy imports stay INSIDE each function on purpose: the heavy
``screener.engine`` / ``screener.provider`` / pattern / level / indicator modules
are imported only when a scan or a detail panel actually needs them, never at
dashboard import time. ``cache_day`` is part of every key so these in-memory memos
stay aligned with cache.py's DATE-KEYED on-disk cache.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from screener.universe import load_universe


# --- memoized scan -------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_cached(
    profile_name: str, n_names: int, cache_day: str, include_symbol: str = ""
) -> pd.DataFrame:
    """Run the engine once per (profile, size, day, include_symbol) and memoize it.

    The key is the hashable tuple ``(profile_name, n_names, cache_day,
    include_symbol)`` — never the DataFrame or provider, which are rebuilt inside.
    ``cache_day`` is ``date.today().isoformat()``: it aligns this in-memory memo
    with cache.py's DATE-KEYED on-disk cache and ``run_screen``'s ``as_of = today``
    default, so a same-day re-pick of the same key is an instant hit and can never
    diverge from the on-disk price/fundamentals/earnings files. (Do NOT drop the
    day from the key — that would let a stale cross-day result be served.)

    ``include_symbol`` is ``""`` for every NORMAL scan, so normal scans of the same
    (profile, size, day) share ONE memo entry. It is non-empty only when an NL query
    names an in-universe ticker that ranks past ``n_names``: :func:`universe_slice`
    then unions that one row into the scanned slice. Filtering never calls
    ``run_cached`` (it operates on the already-stored scan), so threading
    ``include_symbol`` through the key can never trigger a refetch-on-filter.

    The returned frame's ``reasons`` OrderedDict column pickles cleanly through
    ``st.cache_data`` (verified).
    """
    from screener.engine import run_screen
    from screener.universe import universe_slice

    return run_screen(profile_name, universe_slice(load_universe(), n_names, include_symbol))


@st.cache_data(show_spinner=False)
def events_upcoming(cache_day: str, horizon_days: int) -> pd.DataFrame:
    """Upcoming macro events within ``horizon_days`` of today — memoized per day.

    ``cache_day`` is ``date.today().isoformat()`` (same key convention as
    :func:`run_cached` and the per-symbol memos), so the bundled-CSV read +
    ``days_until`` recompute happen once per session-day and the panel re-uses the
    in-hand frame on every rerun. Keyed on ``cache_day`` (not the ``date`` object)
    because ``st.cache_data`` keys must be hashable/stable; the helper re-derives
    ``date.today()`` inside so the memo and the displayed countdowns can't diverge.
    All logic lives in :mod:`screener.calendar`; this is the network-free,
    date-keyed warm read.
    """
    import datetime as dt                              # import INSIDE the function
    from screener import calendar
    return calendar.upcoming_events(dt.date.today(), horizon_days=horizon_days)


@st.cache_data(show_spinner=False)
def patterns_for_symbol(symbol: str, cache_day: str) -> dict:
    """Detect chart patterns for ONE inspected symbol across 1w/1d/1mo.

    Keyed on (symbol, cache_day) so it aligns with the date-keyed on-disk cache:
    re-fetching an already-scanned symbol's prices the same day is instant. Pure
    output (dict[str, list[Pattern]]) pickles through st.cache_data.
    """
    from screener.provider import YFinanceProvider   # import INSIDE the function
    from screener import patterns
    prices = YFinanceProvider().price_history(symbol)
    return patterns.detect_all_timeframes(prices)


@st.cache_data(show_spinner=False)
def levels_for_symbol(symbol: str, cache_day: str) -> dict:
    """Support/resistance LevelSets for ONE inspected symbol across 1w/1d/1mo.

    Same date-keyed warm-read model as :func:`patterns_for_symbol`. Returns
    ``dict[str, levels.LevelSet]`` (always keys ``1w``/``1d``/``1mo``); the
    frozen ``LevelSet``/``Level`` dataclasses are picklable through
    ``st.cache_data``. Fail-soft: a bad/short frame yields empty LevelSets.
    """
    from screener.provider import YFinanceProvider   # import INSIDE the function
    from screener import levels
    prices = YFinanceProvider().price_history(symbol)
    return levels.levels_all_timeframes(prices)


@st.cache_data(show_spinner=False)
def extension_for_symbol(symbol: str, cache_day: str) -> dict:
    """Overextension/parabolic readout for ONE inspected symbol (daily).

    Returns ``ExtensionState.to_dict()`` (a plain dict) so the richer detail
    fields (pct_above_ema20/50, rsi, up_run, atr_pct) survive ``st.cache_data``
    pickling. Fail-soft: a bad/short frame yields the neutral default
    (state ``"normal"``, score ``0.0``, NaN detail fields).
    """
    from screener.provider import YFinanceProvider   # import INSIDE the function
    from screener import indicators
    prices = YFinanceProvider().price_history(symbol)
    return indicators.extension_state(prices).to_dict()


@st.cache_data(show_spinner=False)
def buy_zone_for_symbol(symbol: str, cache_day: str):
    """Daily buy zone (entry band) for ONE inspected symbol — or ``None``.

    Returns a frozen ``levels.BuyZone`` (picklable) or ``None`` when no support
    sits below the close and the 20-EMA is not rising. Date-keyed warm read,
    same model as :func:`patterns_for_symbol`.
    """
    from screener.provider import YFinanceProvider   # import INSIDE the function
    from screener import levels
    prices = YFinanceProvider().price_history(symbol)
    return levels.buy_zone(prices, timeframe="1d")


@st.cache_data(show_spinner=False)
def mcp_supplement_for_symbol(symbol: str, cache_day: str) -> dict:
    """Supplementary fundamentals + earnings for ONE inspected symbol from an
    OPTIONAL external MCP server — or ``{}`` when the feature is OFF/unavailable.

    Milestone B. OFF by default: returns ``{}`` with NO import-cost beyond the
    cheap ``mcp_provider`` module and NO network unless ``MCP_STOCK_DATA_ENABLED``
    is set (the gate lives in :mod:`screener.mcp_provider`; the heavy ``mcp`` SDK
    is lazy-imported only on an actual call). Date-keyed like the other per-symbol
    memos, so an already-inspected name is an instant warm read. The plain-dict
    result pickles cleanly through ``st.cache_data``. This runs ONLY when a user
    inspects a single ticker — never on the universe-scan path.
    """
    from screener.mcp_provider import supplementary_for_symbol   # import INSIDE
    return supplementary_for_symbol(symbol)
