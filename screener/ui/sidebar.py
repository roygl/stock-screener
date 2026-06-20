"""The sidebar: post-scan Filters only (returns None).

After the header-led redesign the high-frequency controls (search, profile, engine,
universe size, Run/Interpret/Refresh, add-ticker) moved into the sticky header
(:mod:`screener.ui.header`). The sidebar now renders ONLY the post-scan Filters
block — the sector multiselect, the minimum-score slider, and the conditional
earnings/extension/buy-zone checkboxes — so it is empty until a scan exists.

Text filtering by symbol / name is owned by the header search box (Decision D1): its
``on_change`` mirrors ``nl_query`` into the plain ``f_text`` session-state value that
``apply_filters`` reads, so the sidebar no longer creates an ``f_text`` widget.

:func:`render_sidebar` returns ``None``; the profile / size the rest of the script
needs are read back out of ``st.session_state`` (the widget keys the header owns).
"""

from __future__ import annotations

import streamlit as st

from screener import display
from screener.profiles import PROFILES, get_profile


# --- sidebar: post-scan filters only -------------------------------------
def render_sidebar(universe) -> None:
    """Render the post-scan Filters block in the sidebar (returns None)."""
    with st.sidebar:
        # Filters appear only after a scan exists.
        if "scan" not in st.session_state:
            return

        st.subheader("Filters")
        scanned = st.session_state["scan"]["df"]
        # The earnings-in-window flag depends on the active profile, now read from the
        # header-owned session_state key (no longer returned by this function).
        profile = get_profile(st.session_state.get("profile_name", next(iter(PROFILES))))

        # Dedicated "filter by ticker": a symbol-only substring narrowing of the cached
        # table (read by apply_filters via results_view, 0 engine calls). Narrower than
        # the header search box (which also matches the company name).
        st.text_input(
            "Ticker",
            key="f_ticker",
            placeholder="e.g. AAPL",
            help="Filter the table to symbols containing this text (case-insensitive). "
                 "The header search box also matches the company name.",
        )
        st.caption("Tip: the header search box up top filters by symbol *or* name.")

        st.multiselect("Sector", options=display.sector_options(scanned), key="f_sectors")
        # Initialize-then-no-default (drop the positional 0.0) so a staged NL score
        # floor is read from session_state without the default-vs-session_state warning.
        if "f_min_score" not in st.session_state:
            st.session_state["f_min_score"] = 0.0
        st.slider("Minimum score", 0.0, 1.0, step=0.01, key="f_min_score")
        if "earnings_in_window" in profile.flags:
            st.checkbox("Only earnings-in-window", key="f_earnings_only")
        # Universe-wide TA filters (columns exist on every profile's scan; gate on
        # presence so they're no-ops if the engine ever omits them).
        if "extension_state" in scanned.columns:
            st.checkbox("Hide overextended (parabolic)", key="f_extended_hidden")
        if "in_buy_zone" in scanned.columns:
            st.checkbox("In buy zone only", key="f_in_buy_zone_only")
