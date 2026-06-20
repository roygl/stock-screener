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

This package was split out of the former single ``screener/display.py`` module
(behaviour-preserving): the logic is unchanged, only grouped by concern into the
``formatting`` / ``features`` / ``tables`` / ``reasons`` / ``tactical`` /
``radar`` / ``text`` submodules (with shared private primitives in ``_base``).
Every public name is re-exported here so ``from screener import display;
display.X`` and ``from screener.display import X`` keep working unchanged.
"""

from __future__ import annotations

# The one allowed internal dependency (kept importable as ``display.Profile``).
from ..profiles import Profile

from .features import (
    FEATURE_DESCRIPTIONS,
    FEATURE_LABELS,
    PROFILE_DESCRIPTIONS,
    feature_description,
    feature_label,
    fit_score,
    profile_description,
)
from .formatting import (
    extension_badge,
    extension_state_color,
    format_market_cap,
    format_price,
    format_signed_pct,
    format_value,
)
from .radar import (
    radar_label,
    radar_spec,
    radar_svg,
)
from .reasons import (
    contribution_caption,
    explain_rank,
    max_contribution,
    narrative,
    narrative_series,
    reasons_to_frame,
    signal_glossary,
)
from .tables import (
    apply_filters,
    column_config_spec,
    column_order,
    empty_message,
    export_frame,
    filter_summary,
    filtered_empty_message,
    is_empty_result,
    resolve_selection,
    row_option_label,
    sector_options,
    table_view,
    take_universe_slice,
)
from .tactical import (
    buy_zone_caption,
    format_buy_zone,
    levels_to_frame,
)
from .text import (
    BUY_ZONE_HELP,
    CONTRIBUTION_HELP,
    DISCLAIMER_DETAIL,
    DISCLAIMER_TEXT,
    EXTENSION_HELP,
    LEVELS_HELP,
    PERCENTILE_HELP,
    SCORE_HELP,
    WHAT_HELP,
    WHY_HELP,
    earnings_badge,
    earnings_badge_series,
    earnings_summary,
    hard_filter_phrases,
    scan_context_line,
    selectivity_hint,
    tradingview_url,
    universe_size_hint,
    yahoo_url,
)

__all__ = [
    # features
    "FEATURE_DESCRIPTIONS",
    "FEATURE_LABELS",
    "PROFILE_DESCRIPTIONS",
    "feature_description",
    "feature_label",
    "fit_score",
    "profile_description",
    # formatting
    "extension_badge",
    "extension_state_color",
    "format_market_cap",
    "format_price",
    "format_signed_pct",
    "format_value",
    # radar
    "radar_label",
    "radar_spec",
    "radar_svg",
    # reasons
    "contribution_caption",
    "explain_rank",
    "max_contribution",
    "narrative",
    "narrative_series",
    "reasons_to_frame",
    "signal_glossary",
    # tables
    "apply_filters",
    "column_config_spec",
    "column_order",
    "empty_message",
    "export_frame",
    "filter_summary",
    "filtered_empty_message",
    "is_empty_result",
    "resolve_selection",
    "row_option_label",
    "sector_options",
    "table_view",
    "take_universe_slice",
    # tactical
    "buy_zone_caption",
    "format_buy_zone",
    "levels_to_frame",
    # text
    "BUY_ZONE_HELP",
    "CONTRIBUTION_HELP",
    "DISCLAIMER_DETAIL",
    "DISCLAIMER_TEXT",
    "EXTENSION_HELP",
    "LEVELS_HELP",
    "PERCENTILE_HELP",
    "SCORE_HELP",
    "WHAT_HELP",
    "WHY_HELP",
    "earnings_badge",
    "earnings_badge_series",
    "earnings_summary",
    "hard_filter_phrases",
    "scan_context_line",
    "selectivity_hint",
    "tradingview_url",
    "universe_size_hint",
    "yahoo_url",
]
