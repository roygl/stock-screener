"""Stock screener — Streamlit dashboard (Milestone 5).

Thin ORDERED ENTRYPOINT. Every piece of the UI lives in the :mod:`screener.ui`
package; this file only sets the page config and calls those pieces in the exact
top-to-bottom order the original single-file script ran them. The engine is
invoked at EXACTLY one site — inside :func:`screener.ui.scan.run_scan_if_requested`
— so the page never auto-runs a cold full-universe scan on a rerun. Sorting (native
table header-click) and filtering (sidebar widgets) operate purely on the cached
result in ``st.session_state["scan"]["df"]`` and never re-run the engine.

Every pure, testable piece (filtering, formatting, the reasons-table builder,
badge/label/caption helpers, the column-config descriptor, selection
reconciliation, and all the messages) lives in :mod:`screener.display`, which
never imports streamlit.

Run locally:
    .venv/bin/python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

# st.set_page_config MUST be the first Streamlit call (Streamlit enforces this).
st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")

from screener.ui import events_panel, header, heatmap_panel, nl_state, persistence, scan, secrets_bridge, sidebar, transparency
from screener.ui.results_view import render_state_switch
from screener.universe import load_universe

# Bridge any provider key from .streamlit/secrets.toml into os.environ BEFORE the
# header reads provider availability.
secrets_bridge.bridge_secrets_to_env()

# The wordmark + global disclaimer caption now live in the sticky header.

# --- load the universe (surface failures, don't crash) -------------------
try:
    universe = load_universe()
except Exception as exc:  # noqa: BLE001 - show the failure in the UI, not a traceback
    st.error(f"Could not load the ticker universe: {exc}")
    st.stop()

# Engine-independent full-universe sectors — computed ONCE so the natural-language
# agent can canonicalize sector names before any scan exists.
universe_sectors = sorted(universe["sector"].dropna().unique().tolist())

# Seed any choices remembered in the browser (localStorage) into the widget keys
# BEFORE the widgets render. Runs before apply_pending so an explicit NL Interpret
# still wins for this run. Returns the localStorage handle for the write-back below.
remembered = persistence.apply_remembered(universe)

# Apply any staged NL request into the widget keys BEFORE the sidebar creates the
# widgets that own those keys (Streamlit forbids setting a widget key after its
# widget exists).
nl_state.apply_pending(universe)

# A staged view switch (e.g. the heatmap's sector drill-down) is applied before the
# header creates the "view" widget, mirroring the NL _pending_* staging (you cannot
# set a widget key after its widget exists).
if "_pending_view" in st.session_state:
    st.session_state["view"] = st.session_state.pop("_pending_view")

# The sticky header renders BEFORE the scan handler and returns the three action
# flags the downstream handlers consume (it owns the nav, search, profile, engine,
# and universe-size widget keys, all seeded above so they survive a rerun).
interpret_clicked, run_clicked, clear_clicked = header.render_header(universe)

# The slim sidebar (Filters-only) renders after the header; it returns nothing.
sidebar.render_sidebar(universe)

# Persist the current selections back to the browser (no-op if unchanged).
persistence.remember_current(remembered, universe)

# Phase 1 of the NL flow: stage the interpreted request, then rerun (no engine call).
nl_state.handle_interpret(interpret_clicked, universe_sectors)

# Phase 2 / manual: the ONE engine call site; ends in st.rerun() when it fires.
scan.run_scan_if_requested(run_clicked, clear_clicked)

# --- main body: branch on the nav view -----------------------------------
# segmented_control can deselect to None in single mode, so read it None-safe.
view = st.session_state.get("view") or "Screener"
if view == "Screener":
    # NL transparency banner, above the four-state switch.
    transparency.render_nl_banner()
    # The four mutually-exclusive main states.
    render_state_switch()
elif view == "Heatmap":
    # Descriptive sector treemap over the cached scan (no engine call): tiles sized by
    # combined market cap, coloured by 3-month momentum, with a one-click drill-down
    # that stages a sector filter + view switch back into the Screener.
    heatmap_panel.render_heatmap_panel()
elif view == "Events":
    # Pull-based, on-open economic-event surface (no server cron on the host): the
    # next high-impact US macro releases with countdowns + impact tags, rendered from
    # the bundled public-domain CSV (memoized per cache_day), expanded as its own view.
    events_panel.render_events_panel(expanded=True)
