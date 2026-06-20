"""Post-symbol detail rendering (relocated verbatim from app.py).

:func:`render_detail` is the back half of the old ``_render_results``: everything
from the symbol-keyed row lookup onward — the "Why X ranks #N" header, the company
header card, fit score, external links, the swing earnings badge, the plain-English
rank explanation, the signal radar, the reasons table, the "How to read this"
glossary, and the four TA detail sections (Chart patterns, Overextension, Support &
resistance, Buy zone). The per-symbol TA fetches use the date-keyed ``@st.cache_data``
memos in :mod:`screener.ui.caching` (warm reads for an already-scanned name).

The selection reconciliation that yields ``symbol`` stays in
:mod:`screener.ui.results_view` (one pass: grid render → resolve → keyed selectbox →
this detail), so the keyed selectbox can never see a stored value absent from its
options. ``render_detail`` only consumes the already-resolved ``symbol``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from screener import calendar, display
from screener.ui.caching import (
    buy_zone_for_symbol,
    extension_for_symbol,
    levels_for_symbol,
    patterns_for_symbol,
)


def _event_risk_for_row(row: pd.Series, cache_day: str) -> dict | None:
    """Event-risk dict for the inspected row, or ``None`` — no refetch, no streamlit.

    Reconstructs the next earnings date from the engine's signed ``days_to_earnings``
    (set on EVERY row, not just swing profiles) relative to the scan's ``cache_day``,
    then folds it through :func:`screener.calendar.next_event_for_symbol` to get the
    same ``{event, date, days_until, impact, within_warning}`` shape used for macro
    events. Returns ``None`` when the earnings date is missing or already past.
    """
    days = row.get("days_to_earnings")
    try:
        days = float(days)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(days):
        return None
    as_of = dt.date.fromisoformat(cache_day)
    earnings_date = as_of + dt.timedelta(days=int(days))
    return calendar.next_event_for_symbol(str(row["symbol"]), earnings_date, as_of)


def render_detail(df: pd.DataFrame, symbol: str, profile, cache_day: str) -> None:
    """Render the why-it-ranks panel + TA detail sections for one inspected symbol.

    ``df`` is the full cached result (the reasons/score/rank source of truth);
    ``symbol`` is the already-reconciled selection from the results view.
    """
    # Look the row up by symbol VALUE in the cached frame (never a stored int).
    row = df.loc[df["symbol"] == symbol].iloc[0]
    reasons = row["reasons"]

    st.subheader(f"Why {symbol} ranks #{int(row['rank'])}")

    # Company header card: the headline price/volatility/size numbers a mainstream
    # finance site leads with, read straight off the in-hand row (all fail-soft to
    # "—"). Daily change uses st.metric's delta for native green/red.
    _price_col, _chg_col, _atr_col, _cap_col = st.columns(4)
    _price_col.metric("Price", display.format_price(row.get("price")))
    _chg_str = display.format_signed_pct(row.get("change_pct"))
    _chg_col.metric(
        "Daily change", _chg_str,
        delta=_chg_str if _chg_str != "—" else None,
    )
    _atr_str = display.format_price(row.get("atr"))
    _atr_pct_str = display.format_signed_pct(row.get("atr_pct"))
    if _atr_str != "—" and _atr_pct_str != "—":
        _atr_str = f"{_atr_str} ({_atr_pct_str})"
    _atr_col.metric("ATR (14)", _atr_str)
    _cap_col.metric("Market cap", display.format_market_cap(row.get("market_cap")))

    # Industry (finer than sector) + the long "what the company does" prose, each
    # shown only when present so a thin-data ticker doesn't render empty chrome.
    _industry = row.get("industry")
    if isinstance(_industry, str) and _industry.strip():
        st.caption(f"Industry: {_industry.strip()}")
    _summary = row.get("business_summary")
    if isinstance(_summary, str) and _summary.strip():
        with st.expander("What the company does"):
            st.write(_summary.strip())

    st.metric("Fit score", f"{display.fit_score(row['score'])} / 100", help=display.SCORE_HELP)

    # Jump straight out to an external chart / quote for the inspected ticker, plus a
    # ★ toggle that adds/removes this symbol from the watchlist (a plain set in
    # session_state; the header's "★ Watchlist only" pill filters on it). The toggle
    # mutates the set then reruns so the new state paints everywhere in one pass.
    _wl = st.session_state.setdefault("watchlist", set())
    _starred = symbol in _wl
    _tv_col, _yf_col, _wl_col = st.columns(3)
    _tv_col.link_button(
        "TradingView", display.tradingview_url(symbol),
        icon=":material/show_chart:", use_container_width=True,
    )
    _yf_col.link_button(
        "Yahoo", display.yahoo_url(symbol),
        icon=":material/open_in_new:", use_container_width=True,
    )
    if _wl_col.button(
        "★ Watchlisted" if _starred else "☆ Watchlist",
        key=f"wl_toggle_{symbol}",
        use_container_width=True,
        help="Star this ticker so you can filter the table to your watchlist (★ Watchlist only, up top).",
    ):
        (_wl.discard if _starred else _wl.add)(symbol)
        st.rerun()

    # Event-risk badge for the inspected row — generalized from the old swing-only
    # earnings badge to ANY profile. The engine sets days_to_earnings on every row
    # (signed days vs the scan's as_of), so we reconstruct the next earnings date
    # WITHOUT a refetch and fold it through screener.calendar.next_event_for_symbol
    # (the same shape used for macro events). Imminent earnings show a ⚠ badge.
    event = _event_risk_for_row(row, cache_day)
    if event is not None and event["within_warning"]:
        st.badge(f"⚠ {event['event']} in {event['days_until']}d", color="orange")

    # Plain-English headline before the numbers: where the row is strong / weak.
    summary = display.explain_rank(row, reasons, profile, total=len(df))
    if summary:
        st.markdown(summary)

    # Signal radar (snowflake): a visual TL;DR of the reasons table — one axis per
    # signal, further from the centre = a higher percentile vs the scan. Rendered in
    # an isolated iframe (components.html); the SVG carries its own theme-robust
    # colours since that iframe doesn't inherit Streamlit's theme.
    radar = display.radar_spec(reasons, profile)
    if len(radar.get("values", [])) >= 3:
        st.caption("Signal radar — percentile on each of this profile's signals.")
        components.html(
            f'<div style="display:flex;justify-content:center">{display.radar_svg(radar)}</div>',
            height=360,
        )

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
