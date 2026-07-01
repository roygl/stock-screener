"""Main results view + the four-state switch (relocated verbatim from app.py).

Holds:

- :func:`render_pre_scan` — the PRE_SCAN state (the cold-scan guard: no engine
  call, no table).
- :func:`render_results` — the RESULTS state: context line, swing banner, the
  AgGrid table, the CSV download, and the selection reconciliation. This keeps the
  ``grid render → resolve_selection → keyed selectbox → detail`` sequence in ONE
  pass (the keyed selectbox raises if its stored value is absent from options), so
  only the post-symbol detail is delegated to :func:`screener.ui.detail_panel.render_detail`.
- :func:`render_state_switch` — the four mutually-exclusive main states
  (PRE_SCAN / ENGINE_EMPTY / FILTERED_EMPTY / RESULTS).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from screener import display
from screener.profiles import PROFILES, get_profile
from screener.ui.detail_panel import render_detail
from screener.ui.grid import render_results_grid


# --- main: four mutually-exclusive states --------------------------------
def render_pre_scan() -> None:
    """PRE_SCAN: no scan yet. The cold-scan guard — no engine call, no table."""
    st.info("Pick a profile up top (universe size lives in ⚙ Settings), then press Run ▶.")
    st.markdown("**Profiles**")
    st.markdown(
        "\n".join(f"- **{PROFILES[k].label}**" for k in PROFILES)
    )


def render_results(
    df: pd.DataFrame, view: pd.DataFrame, profile, n_names: int, cache_day: str
) -> None:
    """RESULTS: context line, swing banner, table, selectbox, why-it-ranks panel.

    ``df`` is the full cached result (the reasons/score/rank source of truth);
    ``view`` is the SAME filtered + index-reset frame passed to ``st.dataframe``,
    so a positional table-click maps straight back to ``view.iloc[pos]``.
    """
    st.caption(display.scan_context_line(profile.label, n_names, cache_day, len(df)))
    # Name the hard filters doing the narrowing, so a small match count (e.g. swing's
    # ~35 of 500) reads as intended selectivity, not a failed/partial fetch.
    hint = display.selectivity_hint(profile, len(df), n_names)
    if hint:
        st.caption(hint)

    # Swing-only earnings warning banner (gated on the flag inside the helper).
    if "earnings_in_window" in profile.flags:
        banner = display.earnings_summary(df)
        if banner:
            st.warning(banner)

    # Density (Compact ↔ Detailed) is a display preference owned by the sidebar.
    density = "detailed" if st.session_state.get("table_density") == "Detailed" else "compact"
    table_df = display.table_view(view, profile, density=density)
    col_order = display.column_order(profile, view, density=density)
    # The per-ticker external links are buttons in the detail panel; this AgGrid
    # build can't render link cells, so drop them from the grid.
    col_order = [c for c in col_order if c not in ("tv_url", "yf_url")]
    # column_order now owns the two universe-wide TA columns; we only re-render the
    # Extension cell as badge TEXT (emoji = the only per-cell color cue). In buy
    # zone stays a raw bool the grid formats Yes/No.
    if "extension_state" in table_df.columns:
        table_df["extension_state"] = table_df["extension_state"].map(display.extension_badge)
    table_df = table_df[[c for c in col_order if c in table_df.columns]]

    st.caption("Double-click a row to inspect it — or use the selector below the table.")
    # Selection input 1: a DOUBLE-CLICK on a grid row -> that row's symbol.
    # AgGrid's `selected_rows` PERSISTS across reruns: once a row is selected it is
    # returned on EVERY rerun, not just the one the user clicked. Since
    # resolve_selection gives a table click top precedence ("a fresh table click
    # wins"), a persisted grid selection would outrank the dropdown forever — so
    # after clicking a row, switching the "Inspect a row" selector below never took
    # effect. Treat the grid as a selection input ONLY when its symbol CHANGES
    # (a genuinely fresh click); on an unchanged grid selection pass None so the
    # dropdown (selectbox_symbol) governs.
    grid_symbol = render_results_grid(table_df, profile)
    table_click_symbol = (
        grid_symbol if grid_symbol != st.session_state.get("_last_grid_symbol") else None
    )
    st.session_state["_last_grid_symbol"] = grid_symbol
    st.caption(display.filter_summary(len(view), len(df)))
    st.download_button(
        "⬇ Download CSV",
        display.export_frame(view, profile).to_csv(index=False).encode("utf-8"),
        file_name=f"screen_{profile.name}_{cache_day}.csv",
        mime="text/csv",
        help="Download the filtered results you see (fit, signals, and the 'why' summary).",
    )

    # Selection input 2: the deterministic selectbox (works before any click).
    # It carries a stable key, but filtering can remove the previously-selected
    # value from the options — Streamlit raises if a keyed selectbox's stored
    # value is absent from `options`. So we reconcile the stored value against the
    # CURRENT view (table click > stored value > prev session symbol > rank-1)
    # and write that valid symbol back to the widget's key BEFORE rendering.
    options = list(view["symbol"])
    stored = st.session_state.get("inspect_select")
    desired = display.resolve_selection(
        view,
        table_click_symbol,
        stored if stored in options else None,
        st.session_state.get("selected_symbol"),
    )
    st.session_state["inspect_select"] = desired
    sel_symbol = st.selectbox(
        "Inspect a row",
        options=options,
        format_func=lambda s: display.row_option_label(view, s),
        key="inspect_select",
    )

    # Reconcile to one symbol (a fresh table click wins over the selectbox value).
    symbol = display.resolve_selection(
        view, table_click_symbol, sel_symbol, st.session_state.get("selected_symbol")
    )
    st.session_state["selected_symbol"] = symbol

    # The why-it-ranks panel + the four TA detail sections for the resolved symbol.
    render_detail(df, symbol, profile, cache_day)


# State switch.
def render_state_switch() -> None:
    """Render exactly one of PRE_SCAN / ENGINE_EMPTY / FILTERED_EMPTY / RESULTS."""
    if "scan" not in st.session_state:
        render_pre_scan()
    else:
        scan = st.session_state["scan"]
        df = scan["df"]
        profile = get_profile(scan["profile_name"])
        n_names = scan["n_names"]
        cache_day = scan["cache_day"]

        if display.is_empty_result(df):
            # ENGINE_EMPTY: the engine returned zero rows.
            st.caption(
                display.scan_context_line(profile.label, n_names, cache_day, 0)
            )
            st.warning(display.empty_message(profile.label, n_names))
        else:
            view = display.apply_filters(
                df,
                text=st.session_state.get("f_text", ""),
                ticker=st.session_state.get("f_ticker", ""),
                sectors=st.session_state.get("f_sectors", []),
                min_score=st.session_state.get("f_min_score", 0.0),
                earnings_only=st.session_state.get("f_earnings_only", False),
                profile=profile,
                extended_hidden=st.session_state.get("f_extended_hidden", False),
                in_buy_zone_only=st.session_state.get("f_in_buy_zone_only", False),
                in_watchlist_only=st.session_state.get("f_in_watchlist_only", False),
                watchlist=st.session_state.get("watchlist", set()),
            )
            if len(view) == 0:
                # FILTERED_EMPTY: filters hid every row (the text filter is the header
                # search box; sector / min-score / checkboxes live in the sidebar).
                st.caption(
                    display.scan_context_line(profile.label, n_names, cache_day, len(df))
                )
                st.info(display.filtered_empty_message(len(df)))
            else:
                render_results(df, view, profile, n_names, cache_day)
