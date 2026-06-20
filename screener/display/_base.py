"""Shared private primitives for the ``screener.display`` package.

Internal-only seam: the one numeric sentinel check, the missing-value glyph, and
the feature-bucket frozensets that drive type-aware formatting. They live here so
both :mod:`screener.display.formatting` (``format_value``) and
:mod:`screener.display.tables` (``column_config_spec``) — and everyone else — can
share them without a circular import. Nothing here is part of the public API by
itself; the package re-exports what callers actually reference.

Imports ONLY pandas / numpy / the stdlib (never streamlit), like the rest of the
package.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


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
# Boolean features (Yes/No cells + checkbox columns).
_BOOL_FEATURES = frozenset({"sma_stacked_20_50_150", "in_buy_zone"})
# Already-fraction scores shown as a plain (unsigned) percent with two decimals —
# distinct from _PERCENT_FEATURES (one decimal) and the signed buy-zone distance.
_PERCENT2_FEATURES = frozenset({"extension_score"})
# Signed-percent fractions: a leading "+"/"-" matters (gap above vs below a band).
_SIGNED_PERCENT_FEATURES = frozenset({"dist_to_buy_zone_pct", "change_pct"})

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
