"""The run handler: the ONE engine call site (relocated verbatim from app.py).

:func:`run_scan_if_requested` is the ONLY caller of :func:`screener.ui.caching.run_cached`
(the engine memo). It fires on (Run scan) OR (Clear cache & rescan) OR (a freshly-
applied NL request) — never on a plain rerun — and always ends in ``st.rerun()`` so
the just-stored scan paints the filters + RESULTS/EMPTY state in one consistent pass.
Do NOT add another ``run_cached`` call anywhere; this site is the cold-scan guard.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from screener.cache import Cache
from screener.ui.caching import run_cached


# --- run handler: the ONLY engine call site ------------------------------
# Fires on (Run scan) OR (a freshly-applied NL request) — never on a plain rerun.
# pop() consumes the NL flag so the scan runs exactly once after an Interpret;
# this IS the cold-scan guard, generalized to two explicit actions.
def run_scan_if_requested(run_clicked: bool, clear_clicked: bool) -> None:
    """Run the engine ONCE if requested, store the result, and rerun."""
    do_scan = run_clicked or clear_clicked or st.session_state.pop("_nl_run_after_apply", False)
    if do_scan:
        if clear_clicked:
            # "Clear cache & rescan" must force FRESH data. st.cache_data.clear() alone
            # only drops this process's in-memory memo and leaves cache.py's date-keyed
            # on-disk parquet/JSON in place, so a same-day rescan would re-read identical
            # files and look like a no-op. Wipe BOTH: the on-disk cache (so the provider
            # re-hits Yahoo) and the memo (so run_cached recomputes), then fall through
            # to actually rescan below — the old button did neither.
            Cache().clear()
            st.cache_data.clear()
        if run_clicked or clear_clicked:
            # A manual Run scan / clear-rescan supersedes any prior NL interpretation —
            # drop the stale banner so it can't describe a scan the user didn't ask for
            # in natural language.
            st.session_state.pop("nl_last_req", None)
        # Read from the widget keys, already reconciled with any staged NL values.
        profile_name = st.session_state["profile_name"]
        n_names = st.session_state["n_names"]
        cache_day = dt.date.today().isoformat()
        spinner_msg = (
            f"Cache cleared — re-fetching {n_names} names from Yahoo. This can take a while…"
            if clear_clicked
            else f"Scanning {n_names} names — the first run of the day hits Yahoo "
            "and can take a while…"
        )
        with st.spinner(spinner_msg):
            df = run_cached(profile_name, n_names, cache_day)
        st.session_state["scan"] = {
            "profile_name": profile_name,
            "n_names": n_names,
            "cache_day": cache_day,
            "df": df,
        }
        st.session_state["selected_symbol"] = None
        # The sidebar (filters) and the main area both read st.session_state["scan"],
        # but the sidebar already rendered ABOVE this handler in top-to-bottom order.
        # Rerun once so the just-stored scan paints the filters + RESULTS/EMPTY state
        # in a single, consistent pass (the spec's "natural rerun") rather than one
        # interaction late.
        st.rerun()
