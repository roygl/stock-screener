"""The "Sector heatmap" surface (rendering only).

A descriptive treemap of the CURRENT cached scan: tiles sized by each sector's
combined market cap, coloured by its 3-month momentum (red weak → green strong).
It answers "where is the rotation right now?" at a glance and offers a one-click
drill-down that filters the Screener to a chosen sector — descriptive context,
never buy/sell guidance.

All logic lives in :mod:`screener.sector_heatmap` (pure, offline, no streamlit);
this module only RENDERS its output (the SVG in an isolated iframe, exactly like
the detail panel's radar) and STAGES client-side view/filter changes. It makes
ZERO engine calls: it reads only ``st.session_state["scan"]["df"]`` and the
sector drill-down flows through the existing guarded scan/apply machinery via the
``_pending_*`` staging keys (applied before the relevant widgets render next run).
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from screener import display, sector_heatmap

# Sentinel first option so the drill-down selectbox doesn't fire on every rerun —
# only an explicit pick of a real sector stages the switch.
_PICK_PROMPT = "(pick a sector)"


def render_heatmap_panel() -> None:
    """Render the sector treemap + a drill-down into the Screener.

    Cold state (no scan yet) shows a hint + the disclaimer and returns. Otherwise it
    aggregates the cached scan via :func:`sector_heatmap.sector_summary` and draws the
    squarified treemap in an iframe (the SVG carries its own theme-robust colours). A
    drill-down selectbox + button stages ``_pending_f_sectors`` (applied by
    ``nl_state.apply_pending`` before the sidebar) and ``_pending_view`` (applied by the
    app's pending-view handler before the header) then reruns, so the chosen sector is
    surfaced in the Screener without any engine call. Fail-soft: an empty summary renders
    a caption rather than crashing.
    """
    if "scan" not in st.session_state:
        st.info("Run a scan first to see sector rotation.")
        st.caption(display.DISCLAIMER_TEXT)
        return

    df = st.session_state["scan"]["df"]
    summary = sector_heatmap.sector_summary(df)
    if summary.empty:
        st.caption("No sector data in the current scan.")
        st.caption(display.DISCLAIMER_TEXT)
        return

    # The treemap SVG, centred in an isolated iframe (mirrors detail_panel's radar).
    components.html(
        f'<div style="display:flex;justify-content:center">{sector_heatmap.treemap_svg(summary)}</div>',
        height=560,
    )
    st.caption(
        "Tile size = combined market cap · colour = 3-month momentum "
        "(red weak → green strong)."
    )

    # --- drill-down: jump to the Screener filtered to one sector ----------
    sectors = summary["sector"].tolist()
    col_pick, col_go = st.columns([3, 1])
    with col_pick:
        chosen = st.selectbox(
            "Open a sector in the Screener",
            options=[_PICK_PROMPT, *sectors],
            index=0,
            key="heatmap_drilldown",
            help="Filters the Screener results to the chosen sector (no new scan).",
        )
    with col_go:
        # Vertical nudge so the button baseline lines up with the selectbox.
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        go = st.button("View in Screener", width="stretch", disabled=(chosen == _PICK_PROMPT))

    if go and chosen != _PICK_PROMPT:
        # Stage the client-side filter + the view switch, then rerun. Both are applied
        # BEFORE the widgets that own those keys render next run (f_sectors via
        # nl_state.apply_pending, view via the app's _pending_view handler) — you cannot
        # set a widget key after its widget already rendered this run. NO engine call.
        st.session_state["_pending_f_sectors"] = [chosen]
        st.session_state["_pending_view"] = "Screener"
        st.rerun()

    st.download_button(
        "Download sector summary (CSV)",
        data=summary.to_csv(index=False),
        file_name="sector_summary.csv",
        mime="text/csv",
    )
    st.caption(display.DISCLAIMER_TEXT)
