"""Stock screener — Streamlit dashboard (Milestone 5).

Thin Streamlit wiring ONLY. Every pure, testable piece (filtering, formatting,
the reasons-table builder, badge/label/caption helpers, the column-config
descriptor, selection reconciliation, and all the messages) lives in
:mod:`screener.display`, which never imports streamlit. This file owns the
widgets, the ``st.session_state`` orchestration, the ``@st.cache_data`` memo, and
the four mutually-exclusive main states.

The engine is invoked at EXACTLY one site — inside ``if run_clicked:`` — so the
page never auto-runs a cold full-universe scan on a rerun. Sorting (native table
header-click) and filtering (sidebar widgets) operate purely on the cached result
in ``st.session_state["scan"]["df"]`` and never re-run the engine.

Run locally:
    .venv/bin/python -m streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from screener import agent, display
from screener.cache import Cache
from screener.profiles import PROFILES, get_profile
from screener.universe import load_universe

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")


# --- memoized scan -------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_cached(profile_name: str, n_names: int, cache_day: str) -> pd.DataFrame:
    """Run the engine once per (profile, size, day) and memoize the result.

    The key is the hashable tuple ``(profile_name, n_names, cache_day)`` — never
    the DataFrame or provider, which are rebuilt inside. ``cache_day`` is
    ``date.today().isoformat()``: it aligns this in-memory memo with cache.py's
    DATE-KEYED on-disk cache and ``run_screen``'s ``as_of = today`` default, so a
    same-day re-pick of the same (profile, size) is an instant hit and can never
    diverge from the on-disk price/fundamentals/earnings files. (Do NOT drop the
    day from the key — that would let a stale cross-day result be served.) The
    returned frame's ``reasons`` OrderedDict column pickles cleanly through
    ``st.cache_data`` (verified).
    """
    from screener.engine import run_screen

    return run_screen(profile_name, load_universe().head(n_names))


@st.cache_data(show_spinner=False)
def patterns_for_symbol(symbol: str, cache_day: str) -> dict:
    """Detect chart patterns for ONE inspected symbol across 1w/1d/1mo.

    Keyed on (symbol, cache_day) so it aligns with the date-keyed on-disk cache:
    re-fetching an already-scanned symbol's prices the same day is instant. Pure
    output (dict[str, list[Pattern]]) pickles through st.cache_data.
    """
    from screener.provider import YFinanceProvider   # import INSIDE the function
    from screener import patterns
    prices = YFinanceProvider().price_history(symbol)
    return patterns.detect_all_timeframes(prices)


@st.cache_data(show_spinner=False)
def levels_for_symbol(symbol: str, cache_day: str) -> dict:
    """Support/resistance LevelSets for ONE inspected symbol across 1w/1d/1mo.

    Same date-keyed warm-read model as :func:`patterns_for_symbol`. Returns
    ``dict[str, levels.LevelSet]`` (always keys ``1w``/``1d``/``1mo``); the
    frozen ``LevelSet``/``Level`` dataclasses are picklable through
    ``st.cache_data``. Fail-soft: a bad/short frame yields empty LevelSets.
    """
    from screener.provider import YFinanceProvider   # import INSIDE the function
    from screener import levels
    prices = YFinanceProvider().price_history(symbol)
    return levels.levels_all_timeframes(prices)


@st.cache_data(show_spinner=False)
def extension_for_symbol(symbol: str, cache_day: str) -> dict:
    """Overextension/parabolic readout for ONE inspected symbol (daily).

    Returns ``ExtensionState.to_dict()`` (a plain dict) so the richer detail
    fields (pct_above_ema20/50, rsi, up_run, atr_pct) survive ``st.cache_data``
    pickling. Fail-soft: a bad/short frame yields the neutral default
    (state ``"normal"``, score ``0.0``, NaN detail fields).
    """
    from screener.provider import YFinanceProvider   # import INSIDE the function
    from screener import indicators
    prices = YFinanceProvider().price_history(symbol)
    return indicators.extension_state(prices).to_dict()


@st.cache_data(show_spinner=False)
def buy_zone_for_symbol(symbol: str, cache_day: str):
    """Daily buy zone (entry band) for ONE inspected symbol — or ``None``.

    Returns a frozen ``levels.BuyZone`` (picklable) or ``None`` when no support
    sits below the close and the 20-EMA is not rising. Date-keyed warm read,
    same model as :func:`patterns_for_symbol`.
    """
    from screener.provider import YFinanceProvider   # import INSIDE the function
    from screener import levels
    prices = YFinanceProvider().price_history(symbol)
    return levels.buy_zone(prices, timeframe="1d")


# --- title + global disclaimer caption -----------------------------------
st.title("📈 Stock Screener")
st.caption("US large-cap · end-of-day · ranks and describes · educational buy zone, not advice")

# --- load the universe (surface failures, don't crash) -------------------
try:
    universe = load_universe()
except Exception as exc:  # noqa: BLE001 - show the failure in the UI, not a traceback
    st.error(f"Could not load the ticker universe: {exc}")
    st.stop()

# Engine-independent full-universe sectors — computed ONCE so the natural-language
# agent can canonicalize sector names before any scan exists.
universe_sectors = sorted(universe["sector"].dropna().unique().tolist())


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
def _stage_nl_request(req: agent.ScreenRequest) -> None:
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
for _pend_key, _widget_key in _PENDING.items():
    if _pend_key in st.session_state:
        st.session_state[_widget_key] = st.session_state.pop(_pend_key)
# Keep a staged n_names within the live slider's range so rendering can't raise on
# a degenerate small universe (in production at 503 names this never triggers).
if "n_names" in st.session_state:
    _smin = min(5, len(universe))
    _smax = max(_smin, len(universe))
    st.session_state["n_names"] = max(_smin, min(int(st.session_state["n_names"]), _smax))


# --- builder: pure column-config descriptor -> real st.column_config ------
def _build_column_config(profile) -> dict:
    """Turn :func:`display.column_config_spec` (a pure dict) into st.column_config.

    This is the single purity boundary: ``st.column_config.*`` objects can only
    be built where streamlit is imported, so the descriptor is produced purely in
    ``display`` and realised here.
    """
    cfg: dict = {}
    for col, desc in display.column_config_spec(profile).items():
        kind = desc.get("kind")
        label = desc.get("label", col)
        help_text = desc.get("help")  # header "?" tooltip (None -> no tooltip)
        if kind == "progress":
            cfg[col] = st.column_config.ProgressColumn(
                label, help=help_text, format=desc.get("format"),
                min_value=desc.get("min", 0.0), max_value=desc.get("max", 1.0),
            )
        elif kind == "percent":
            # Percent-style features are FRACTIONS in the engine (0.12 == 12%).
            # The "percent" format preset (NOT a printf "%%" pattern) multiplies
            # by 100 for display, so 0.12 renders as "12.00%".
            cfg[col] = st.column_config.NumberColumn(label, help=help_text, format=desc.get("format", "percent"))
        elif kind == "number":
            cfg[col] = st.column_config.NumberColumn(label, help=help_text, format=desc.get("format"))
        elif kind == "checkbox":
            cfg[col] = st.column_config.CheckboxColumn(label, help=help_text)
        elif kind == "link":
            # Per-ticker external link: `display_text` is the in-cell glyph; the
            # column holds a real URL per row (built in display.table_view).
            cfg[col] = st.column_config.LinkColumn(
                label, help=help_text, display_text=desc.get("display_text"),
            )
        else:  # text
            cfg[col] = st.column_config.TextColumn(label, help=help_text)
    return cfg


# --- sidebar: controls, filters, disclaimer ------------------------------
with st.sidebar:
    # Natural-language box FIRST (above Controls). It is a SECOND trigger into the
    # single engine call site — never a separate scan path.
    st.subheader("Ask in plain English")
    st.text_input(
        "Describe what to screen for",
        key="nl_query",
        placeholder="e.g. top 20 momentum tech names, high conviction",
    )
    if agent.agent_available():
        st.caption("LLM-backed (claude) — interprets your request, then runs the scan.")
    else:
        st.caption(
            "Offline rule-based parser (set ANTHROPIC_API_KEY + install anthropic "
            "for LLM mode)."
        )
    interpret_clicked = st.button("Interpret & run", key="nl_btn")
    st.divider()

    st.header("Controls")

    profile_name = st.radio(
        "Profile",
        options=list(PROFILES),
        format_func=lambda k: PROFILES[k].label,
        captions=[display.profile_description(k) for k in PROFILES],
        key="profile_name",
    )
    profile = get_profile(profile_name)

    # Asset-class toggle: a disabled no-op stub (crypto is v2).
    st.radio("Asset class", ["US equities"], disabled=True, help="Crypto arrives in v2.")

    # The production S&P 500 universe is 503 names, so this is min 5 / max 503 /
    # default 25 as specced. The bounds are clamped only so a degenerate tiny
    # universe (fewer than 5 names) can't make min_value exceed max_value.
    universe_len = len(universe)
    slider_min = min(5, universe_len)
    slider_max = max(slider_min, universe_len)
    if slider_max > slider_min:
        # Initialize-then-no-default: seed the key once (so a staged NL value or a
        # prior selection survives) and pass NO value= arg, the canonical pattern
        # that avoids Streamlit's default-vs-session_state warning.
        if "n_names" not in st.session_state:
            st.session_state["n_names"] = min(25, slider_max)
        n_names = st.slider(
            "Universe size (names to scan)",
            min_value=slider_min,
            max_value=slider_max,
            step=5,
            key="n_names",
            help="A cold scan hits Yahoo once per name; start small.",
        )
    else:
        # Universe too small for a range slider — scan all of it. Mirror into the
        # session key so the scan block (which reads st.session_state["n_names"])
        # has a value even though no keyed slider rendered.
        n_names = universe_len
        st.session_state["n_names"] = universe_len
        st.caption(f"Universe has {universe_len} name(s); scanning all of them.")
    st.caption(display.universe_size_hint(n_names))

    run_clicked = st.button("Run scan", type="primary", key="run_btn")
    st.caption("Sorting and filtering act on the last scan — no refetch.")
    clear_clicked = st.button(
        "Clear cache & rescan",
        key="clear_btn",
        help="Delete today's cached prices/fundamentals and re-fetch from Yahoo, then rescan.",
    )

    # Filters appear only after a scan exists.
    if "scan" in st.session_state:
        st.divider()
        st.subheader("Filters")
        scanned = st.session_state["scan"]["df"]
        st.text_input("Filter symbol / name", key="f_text")
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

    st.divider()
    st.caption(display.DISCLAIMER_TEXT)
    with st.expander("Disclaimer — not financial advice"):
        st.write(display.DISCLAIMER_DETAIL)


# --- natural-language Interpret handler ----------------------------------
# Phase 1 of the two-phase NL flow: parse the query, STAGE the result into
# _pending_* keys (+ _nl_run_after_apply), then rerun so the top-of-script
# pending-apply seeds the widget keys BEFORE the widgets render next pass. The
# engine is NOT called here — the scan fires in phase 2 below.
if interpret_clicked:
    req = agent.parse_query(
        st.session_state.get("nl_query", ""),
        universe_sectors=universe_sectors,
    )
    _stage_nl_request(req)
    st.rerun()


# --- run handler: the ONLY engine call site ------------------------------
# Fires on (Run scan) OR (a freshly-applied NL request) — never on a plain rerun.
# pop() consumes the NL flag so the scan runs exactly once after an Interpret;
# this IS the cold-scan guard, generalized to two explicit actions.
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


# --- main: four mutually-exclusive states --------------------------------
def _render_pre_scan() -> None:
    """PRE_SCAN: no scan yet. The cold-scan guard — no engine call, no table."""
    st.info("Pick a profile and universe size in the sidebar, then press Run scan.")
    st.markdown("**Profiles**")
    st.markdown(
        "\n".join(f"- **{PROFILES[k].label}**" for k in PROFILES)
    )


def _render_results(
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

    table_df = display.table_view(view, profile)
    col_order = display.column_order(profile, view)
    # Append the two universe-wide TA columns (owned here, not by column_order):
    # Extension renders as badge TEXT (emoji = the only per-cell color cue a
    # TextColumn allows); In buy zone stays a raw bool for the CheckboxColumn.
    # table_view copies only column_order cols, so add them from `view` directly.
    # Display-only augmentation of table_df is safe: the positional click below
    # still indexes `view` (unmodified), not table_df.
    if "extension_state" in view.columns:
        table_df["extension_state"] = view["extension_state"].map(display.extension_badge).values
        if "extension_state" not in col_order:
            col_order.append("extension_state")
    if "in_buy_zone" in view.columns:
        table_df["in_buy_zone"] = view["in_buy_zone"].values
        if "in_buy_zone" not in col_order:
            col_order.append("in_buy_zone")
    cfg = _build_column_config(profile)
    event = st.dataframe(
        table_df,
        hide_index=True,
        width="stretch",
        column_order=col_order,
        column_config=cfg,
        on_select="rerun",
        selection_mode="single-row",
        key="results_table",
    )
    st.caption(display.filter_summary(len(view), len(df)))

    # Selection input 1: native table click -> POSITIONAL index into `view`.
    table_click_symbol = None
    try:
        selected_rows = event.selection["rows"]
    except (AttributeError, KeyError, TypeError):
        selected_rows = getattr(getattr(event, "selection", None), "rows", []) or []
    if selected_rows:
        pos = selected_rows[0]
        if 0 <= pos < len(view):
            table_click_symbol = view.iloc[pos]["symbol"]

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

    # Look the row up by symbol VALUE in the cached frame (never a stored int).
    row = df.loc[df["symbol"] == symbol].iloc[0]
    reasons = row["reasons"]

    st.subheader(f"Why {symbol} ranks #{int(row['rank'])}")
    st.metric("Score", f"{float(row['score']):.3f}", help=display.SCORE_HELP)

    # Jump straight out to an external chart / quote for the inspected ticker.
    _tv_col, _yf_col = st.columns(2)
    _tv_col.link_button(
        "TradingView", display.tradingview_url(symbol),
        icon=":material/show_chart:", use_container_width=True,
    )
    _yf_col.link_button(
        "Yahoo", display.yahoo_url(symbol),
        icon=":material/open_in_new:", use_container_width=True,
    )

    # Swing-only earnings badge for the inspected row.
    if "earnings_in_window" in profile.flags:
        badge = display.earnings_badge(
            row.get("earnings_in_window"), row.get("days_to_earnings")
        )
        if badge:
            st.badge(badge, color="orange")

    # Plain-English headline before the numbers: where the row is strong / weak.
    summary = display.explain_rank(row, reasons, profile, total=len(df))
    if summary:
        st.markdown(summary)

    reasons_df = display.reasons_to_frame(reasons, profile)
    st.dataframe(
        reasons_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Signal": st.column_config.TextColumn("Signal", width="small"),
            "What it measures": st.column_config.TextColumn(
                "What it measures", help=display.WHAT_HELP, width="large",
            ),
            "Value": st.column_config.TextColumn("Value", width="small"),
            "Percentile": st.column_config.ProgressColumn(
                "Percentile", help=display.PERCENTILE_HELP,
                format="%.2f", min_value=0.0, max_value=1.0, width="medium",
            ),
            "Contribution": st.column_config.ProgressColumn(
                "Contribution", help=display.CONTRIBUTION_HELP,
                format="%.3f", min_value=0.0,
                max_value=display.max_contribution(reasons), width="medium",
            ),
        },
    )
    st.caption(display.contribution_caption(reasons, float(row["score"])))

    # Definitions for the columns + every signal in this profile (kept out of the
    # table so the bars stay readable; the inline "What it measures" column carries
    # the short version).
    with st.expander("ℹ️ How to read this"):
        intro = display.profile_description(profile.name)
        if intro:
            st.caption(intro)
        st.markdown(
            f"- **Score** — {display.SCORE_HELP}\n"
            f"- **Percentile** — {display.PERCENTILE_HELP}\n"
            f"- **Contribution** — {display.CONTRIBUTION_HELP}"
        )
        st.markdown("**Signals in this profile**")
        for label, desc in display.signal_glossary(profile):
            st.markdown(f"- **{label}** — {desc}")

    # --- Chart patterns (descriptive shapes for the inspected symbol) -----
    # On-demand for the ONE inspected symbol only — NOT part of the universe scan.
    # The fetch is cached + date-keyed (a warm read for an already-scanned name),
    # so this adds no cost to the cold-scan path and lives only in the RESULTS state.
    st.divider()
    st.subheader("Chart patterns")
    st.caption(
        "Mechanically-detected price shapes across timeframes — descriptive "
        "geometry from end-of-day bars, not signals to act on."
    )
    detected = patterns_for_symbol(symbol, cache_day)
    TF_LABELS = {"1w": "Weekly", "1d": "Daily", "1mo": "Monthly"}
    any_found = any(detected.get(tf) for tf in ("1w", "1d", "1mo"))
    if not any_found:
        st.write("No notable patterns detected on the 1w / 1d / 1mo timeframes.")
    else:
        for tf in ("1w", "1d", "1mo"):
            pats = detected.get(tf, [])
            st.markdown(f"**{TF_LABELS[tf]}**")
            if not pats:
                st.caption("No notable patterns.")
                continue
            rows = [
                {
                    "Pattern": p.label(),
                    "Direction": p.direction,
                    "Confidence": float(p.confidence),
                    "Span": f"{p.start.date()} → {p.end.date()}",
                }
                for p in pats
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", format="%.2f", min_value=0.0, max_value=1.0
                    )
                },
            )

    # --- Overextension (how stretched the inspected symbol is) ------------
    # Warm per-symbol read (date-keyed cache), same model as Chart patterns.
    # extension_for_symbol returns ExtensionState.to_dict(); detail % fields can
    # be NaN on a short frame -> format_signed_pct renders "—" rather than raise.
    st.divider()
    st.subheader("Overextension")
    ext = extension_for_symbol(symbol, cache_day)
    ext_state = ext.get("state", "normal")
    st.badge(
        display.extension_badge(ext_state),
        color=display.extension_state_color(ext_state),
    )
    _e20, _e50, _ersi, _erun = st.columns(4)
    _e20.metric("% above 20-EMA", display.format_signed_pct(ext.get("pct_above_ema20")))
    _e50.metric("% above 50-EMA", display.format_signed_pct(ext.get("pct_above_ema50")))
    _rsi_val = ext.get("rsi")
    _ersi.metric(
        "RSI(14)",
        "—" if _rsi_val != _rsi_val or _rsi_val is None else f"{float(_rsi_val):.0f}",
    )
    _erun.metric("Up-day run", f"{int(ext.get('up_run', 0))}")
    st.caption(display.EXTENSION_HELP)

    # --- Support & resistance (תמיכה והתנגדות) per timeframe --------------
    # levels_for_symbol returns dict[tf, LevelSet]; levels_to_frame yields a
    # frame with a numeric 0..1 Strength (ProgressColumn, like the Confidence
    # bar above) and a pre-formatted signed-pct Distance string (TextColumn).
    st.divider()
    st.subheader("Support & resistance")
    level_sets = levels_for_symbol(symbol, cache_day)
    for tf in ("1w", "1d", "1mo"):
        st.markdown(f"**{TF_LABELS[tf]}**")
        frame = display.levels_to_frame(level_sets.get(tf))
        if frame.empty:
            st.caption("No notable levels.")
            continue
        st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_config={
                "Strength": st.column_config.ProgressColumn(
                    "Strength", format="%.2f", min_value=0.0, max_value=1.0
                ),
            },
        )
    st.caption(display.LEVELS_HELP)

    # --- Buy zone (explicit entry band — descriptive, with disclaimer) ----
    # The disclaimer travels inside buy_zone_caption in BOTH the present-zone
    # and None branches (a guardrail for the relaxed "no advice" stance).
    st.divider()
    st.subheader("Buy zone")
    zone = buy_zone_for_symbol(symbol, cache_day)
    st.metric("Buy zone", display.format_buy_zone(zone))
    if zone is None:
        st.caption("No buy zone below the current price.")
    st.caption(display.buy_zone_caption(zone))
    with st.expander("ℹ️ About the buy zone"):
        st.caption(display.BUY_ZONE_HELP)


# --- NL transparency: show how the last request was interpreted ----------
# Above the four-state switch so the user always sees the interpretation that
# drove the current scan (the explanation + the resolved knobs).
last = st.session_state.get("nl_last_req")
if last is not None:
    st.info(f"Interpreted your request — {last.explanation}")
    st.caption(
        f"profile={last.profile} · names={last.n_names} · min_score={last.min_score:g}"
        + (f" · sectors={', '.join(last.sectors)}" if last.sectors else "")
        + (f" · symbol={last.text}" if last.text else "")
        + (" · earnings-only" if last.earnings_only else "")
    )


# State switch.
if "scan" not in st.session_state:
    _render_pre_scan()
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
            sectors=st.session_state.get("f_sectors", []),
            min_score=st.session_state.get("f_min_score", 0.0),
            earnings_only=st.session_state.get("f_earnings_only", False),
            profile=profile,
            extended_hidden=st.session_state.get("f_extended_hidden", False),
            in_buy_zone_only=st.session_state.get("f_in_buy_zone_only", False),
        )
        if len(view) == 0:
            # FILTERED_EMPTY: filters hid every row (filters stay in the sidebar).
            st.caption(
                display.scan_context_line(profile.label, n_names, cache_day, len(df))
            )
            st.info(display.filtered_empty_message(len(df)))
        else:
            _render_results(df, view, profile, n_names, cache_day)
