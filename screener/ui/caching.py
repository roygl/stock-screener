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
def run_cached(profile_name: str, n_names: int, cache_day: str) -> pd.DataFrame:
    """Run the engine once per (profile, size, day) and memoize the result.

    The key is the hashable tuple ``(profile_name, n_names, cache_day)`` — never
    the DataFrame or provider, which are rebuilt inside. ``cache_day`` is
    ``date.today().isoformat()``: it aligns this in-memory memo with cache.py's
    DATE-KEYED on-disk cache and ``run_screen``'s ``as_of = today`` default, so a
    same-day re-pick of the same (profile, size) is an instant hit and can never
    diverge from the on-disk price/fundamentals/earnings files. (Do NOT drop the
    day from the key — that would let a stale cross-day result be served.) The
    returned frame's ``reasons`` OrderedDict column pickles cleanly through
    ``st.cache_data`` (verified).
    """
    from screener.engine import run_screen

    return run_screen(profile_name, load_universe().head(n_names))


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
