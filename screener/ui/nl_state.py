"""Natural-language staging machinery (relocated verbatim from app.py).

THE CRUX of wiring the NL output into the existing keyed widgets WITHOUT the
Streamlit "created with a default value but also had its value set via Session
State API" warning, and WITHOUT breaking the cold-scan guard. Three pieces, run
at three points of a single script pass:

  1. :func:`apply_pending` — the ``_pending_*`` → widget-key apply loop (+ the
     ``n_names`` clamp). MUST run at the TOP of the script, BEFORE any widget that
     owns one of those keys is created (i.e. before the sidebar).
  2. :func:`handle_interpret` — the Interpret button handler. Parses the query and
     STAGES it (via :func:`stage_nl_request`), then reruns. The engine is NOT
     called here — the scan fires in :func:`screener.ui.scan.run_scan_if_requested`.
  3. :func:`stage_nl_request` — writes the ``_pending_*`` keys consumed by
     :func:`apply_pending` on the next run, and sets ``_nl_run_after_apply`` (which
     the single scan block pops to fire the engine exactly once after an Interpret).
"""

from __future__ import annotations

import streamlit as st

from screener import agent


# --- natural-language agent: staging + pending-apply ---------------------
# THE CRUX of wiring the NL output into the existing keyed widgets WITHOUT the
# Streamlit "created with a default value but also had its value set via Session
# State API" warning, and WITHOUT breaking the cold-scan guard. Two pieces:
#
# (1) _stage_nl_request writes _pending_* keys (consumed at the top of the NEXT
#     run, before the widgets are created) and sets _nl_run_after_apply, which the
#     single scan block pops to fire the engine exactly once after an Interpret.
# (2) The pending-apply loop below pops each _pending_* into its widget key BEFORE
#     any widget with that key is instantiated this run, so the widgets render with
#     the NL-chosen values and Streamlit reads them from session_state (no warning).
def stage_nl_request(req: agent.ScreenRequest) -> None:
    """Stage an interpreted request into _pending_* keys for the next run."""
    st.session_state["_pending_profile_name"] = req.profile
    st.session_state["_pending_n_names"] = req.n_names
    st.session_state["_pending_f_text"] = req.text
    st.session_state["_pending_f_sectors"] = list(req.sectors)
    st.session_state["_pending_f_min_score"] = float(req.min_score)
    st.session_state["_pending_f_earnings_only"] = bool(req.earnings_only)
    st.session_state["nl_last_req"] = req
    st.session_state["_nl_run_after_apply"] = True


_PENDING = {
    "_pending_profile_name": "profile_name",
    "_pending_n_names": "n_names",
    "_pending_f_text": "f_text",
    "_pending_f_sectors": "f_sectors",
    "_pending_f_min_score": "f_min_score",
    "_pending_f_earnings_only": "f_earnings_only",
}


def apply_pending(universe) -> None:
    """Apply staged _pending_* values into their widget keys (before the sidebar).

    Pops each _pending_* into its widget key BEFORE any widget with that key is
    instantiated this run, then clamps a staged ``n_names`` to the live slider's
    range so rendering can't raise on a degenerate small universe.
    """
    for _pend_key, _widget_key in _PENDING.items():
        if _pend_key in st.session_state:
            st.session_state[_widget_key] = st.session_state.pop(_pend_key)
    # Keep a staged n_names within the live slider's range so rendering can't raise on
    # a degenerate small universe (in production at 503 names this never triggers).
    if "n_names" in st.session_state:
        _smin = min(5, len(universe))
        _smax = max(_smin, len(universe))
        st.session_state["n_names"] = max(_smin, min(int(st.session_state["n_names"]), _smax))


# --- natural-language Interpret handler ----------------------------------
# Phase 1 of the two-phase NL flow: parse the query, STAGE the result into
# _pending_* keys (+ _nl_run_after_apply), then rerun so the top-of-script
# pending-apply seeds the widget keys BEFORE the widgets render next pass. The
# engine is NOT called here — the scan fires in phase 2 below.
def handle_interpret(interpret_clicked: bool, universe_sectors: list) -> None:
    """If Interpret was clicked, parse + stage the request, then rerun."""
    if interpret_clicked:
        req = agent.parse_query(
            st.session_state.get("nl_query", ""),
            universe_sectors=universe_sectors,
            provider=st.session_state.get("nl_provider"),
        )
        # If the query named a ticker that ISN'T in the universe yet, fetch it via the
        # provider and persist it to data/universe.csv so the scan can include it.
        # ensure_symbol self-guards: a non-ticker or already-present symbol makes NO
        # network call and returns False. Imported lazily so this module — and the
        # Interpret handler's hot path — stay import-cheap. agent.parse_query stays
        # network-free; the only network touch lives here, after parsing.
        from screener.universe import ensure_symbol

        added = ensure_symbol((req.text or "").strip())
        if added:
            # A brand-new row landed in the universe. Drop the engine memo so a
            # full-universe scan computed earlier today is recomputed to include it
            # (cache_day alone wouldn't invalidate it). Lazy import avoids pulling the
            # streamlit-cached module unless we actually added a symbol.
            from screener.ui.caching import run_cached

            run_cached.clear()
        stage_nl_request(req)
        st.rerun()
