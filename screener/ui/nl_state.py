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
def _bare_ticker_candidate(query: str) -> str:
    """Upper-cased symbol when the WHOLE query is ONE ticker-shaped token, else ``""``.

    The agent's symbol matcher is deliberately case-SENSITIVE (so prose like "tech
    stocks" can't be hijacked by a stray cap), which means a lowercase bare ticker —
    "pltr", "dkng" — is left as a non-symbol and was silently never offered for add.
    This recovers exactly that case: a single token, validated against the universe's
    canonical ticker shape (:data:`screener.universe._TICKER_RE`) and the agent's
    stopword list, so it stays blind to multi-word / prose queries (which keep ``""``).
    """
    from screener.agent import _TICKER_STOPWORDS
    from screener.universe import _TICKER_RE

    toks = (query or "").strip().split()
    if len(toks) != 1:
        return ""
    sym = toks[0].upper()
    return sym if _TICKER_RE.match(sym) and sym not in _TICKER_STOPWORDS else ""


def handle_interpret(interpret_clicked: bool, universe_sectors: list) -> None:
    """If Interpret was clicked, parse + (maybe) add the named ticker, stage, rerun.

    The auto-add is no longer silent and no longer upper-case-only:

    * **Lowercase bare tickers work** — :func:`_bare_ticker_candidate` recovers a
      single-token symbol the case-sensitive parser skipped, and when it is present
      or freshly added we normalize ``req.text`` to its canonical form so the symbol
      filter + scan ``include_symbol`` actually surface it.
    * **The outcome is reported** via ``nl_add_msg`` (rendered in the transparency
      banner): "✓ Added X" on a new fetch, and — only for an UNAMBIGUOUS upper-case
      ticker the user typed — a "couldn't add" error on a failed fetch. A lowercase
      token that fetches nothing stays quiet (it may simply not be a ticker), and an
      already-present symbol stays a quiet normal filter, exactly as before.
    """
    if not interpret_clicked:
        return
    import dataclasses

    from screener.universe import ensure_symbol, load_universe

    # Clear any prior add-result so the banner reflects only THIS interpretation.
    st.session_state.pop("nl_add_msg", None)

    req = agent.parse_query(
        st.session_state.get("nl_query", ""),
        universe_sectors=universe_sectors,
        provider=st.session_state.get("nl_provider"),
    )
    # The symbol to (maybe) add: the parser's upper-case match, else a lowercase bare
    # token it left as a non-symbol. ensure_symbol self-guards (no network for a
    # non-ticker or already-present symbol); imported lazily so the hot path stays cheap.
    parser_symbol = (req.text or "").strip().upper()
    fallback_symbol = "" if parser_symbol else _bare_ticker_candidate(
        st.session_state.get("nl_query", "")
    )
    candidate = parser_symbol or fallback_symbol

    if candidate:
        was_present = candidate in set(load_universe()["symbol"])
        added = ensure_symbol(candidate)
        if added:
            # A brand-new row landed. Drop the engine memo so a full-universe scan
            # computed earlier today is recomputed to include it (cache_day alone
            # wouldn't invalidate it). Lazy import keeps this off the cold path.
            from screener.ui.caching import run_cached

            run_cached.clear()
            st.session_state["nl_add_msg"] = (
                "success",
                f"✓ Added {candidate} to the universe ({len(load_universe())} names).",
            )
        elif not was_present and parser_symbol:
            # Unambiguous: the user typed an upper-case ticker that fetched nothing.
            st.session_state["nl_add_msg"] = (
                "error",
                f"Couldn’t add {candidate}: Yahoo returned no data for that symbol.",
            )
        # Point the filter / include_symbol at the canonical form for a lowercase
        # token that is present or was just added (so it's actually surfaced); a
        # failed lowercase fallback stays unfiltered — it may not be a ticker at all.
        if fallback_symbol and (was_present or added):
            req = dataclasses.replace(req, text=candidate)

    stage_nl_request(req)
    st.rerun()
