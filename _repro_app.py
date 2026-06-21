"""Wrapper around app.py for offline reproduction in the REAL Streamlit runtime.

Injects an all-good offline provider and a small universe BEFORE running app.py, so
the scan needs no network and the full RESULTS path renders fast — while the real
localStorage component and real st_aggrid grid run as in production (the place a
StreamlitDuplicateElementKey would actually fire).
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import screener.engine as eng
import screener.provider as provider
import screener.universe as universe
import screener.ui.caching as caching
from screener.provider import DataProvider, Fundamentals

_SECTORS = ["Technology", "Energy", "Financials", "Health Care", "Industrials"]
_ROWS = [(f"SYM{i:02d}", f"Company {i}", _SECTORS[i % len(_SECTORS)]) for i in range(12)]
_UNIV = pd.DataFrame(_ROWS, columns=["symbol", "name", "sector"])


def _frame(seed):
    n = 320
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    idx.name = "date"
    rng = np.random.default_rng(seed)
    close = 50.0 + np.arange(n) * (0.2 + 0.02 * seed) + rng.normal(0, 0.5, n)
    close = np.clip(close, 1.0, None)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


class AllGoodProvider(DataProvider):
    def price_history(self, symbol, *, lookback_days=730):
        return _frame(abs(hash(symbol)) % 97)

    def fundamentals(self, symbol):
        return Fundamentals(
            symbol=symbol, name=f"{symbol} Inc", sector="Technology",
            market_cap=1e11 + abs(hash(symbol)) % 50 * 1e9,
            forward_pe=20.0, trailing_pe=25.0, revenue_growth=0.12,
            earnings_growth=0.15, industry="Software",
            business_summary="Does things.",
        )

    def earnings_date(self, symbol):
        return dt.date.today() + dt.timedelta(days=5)


# Patch the engine's default provider + the universe loader everywhere they're read.
provider.YFinanceProvider = lambda *a, **k: AllGoodProvider()
eng.YFinanceProvider = lambda *a, **k: AllGoodProvider()
universe.load_universe = lambda: _UNIV.copy()
caching.load_universe = lambda: _UNIV.copy()

# --- instrument duplicate-key detection (idempotent across reruns) --------
import traceback  # noqa: E402

import streamlit.elements.lib.utils as _stutils  # noqa: E402

if not getattr(_stutils, "_dupkey_patched", False):
    _stutils._orig_register = _stutils._register_element_id
    _stutils._seen = {}

    def _patched_register(ctx, element_type, element_id):
        seen = _stutils._seen
        if not getattr(ctx, "widget_user_keys_this_run", None):
            seen.clear()  # start-of-run: the ctx set is empty
        uk = _stutils.user_key_from_element_id(element_id)
        if uk is not None:
            if uk in seen:
                print("\n##### DUPLICATE KEY:", repr(uk), "type=", element_type, flush=True)
                print("---- FIRST site ----\n" + seen[uk], flush=True)
                print("---- SECOND site ----\n" + "".join(traceback.format_stack()), flush=True)
                return  # swallow so the drive can continue and collect every dup
            seen[uk] = "".join(traceback.format_stack()[-9:])
        return _stutils._orig_register(ctx, element_type, element_id)

    _stutils._register_element_id = _patched_register
    _stutils._dupkey_patched = True

# Run the real app.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")) as fh:
    code = compile(fh.read(), "app.py", "exec")
exec(code, {"__name__": "__main__", "__file__": "app.py"})
