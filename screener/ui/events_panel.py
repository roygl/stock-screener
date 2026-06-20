"""The "Upcoming events" macro-event surface (rendering only).

A pull-based, in-app economic-event panel rendered when the user opens the app —
NOT a server-side alerting service (Streamlit Community Cloud sleeps when idle and
has no cron, so a date "alert" cannot fire server-side; see DEPLOY.md). It lists
the next high-impact US macro releases (FOMC rate decisions, CPI, the monthly jobs
report) with a "days until" countdown, an impact tag, and an advance-warning flag
for events inside N days, framed as windows of *heightened expected volatility* —
descriptive, never buy/sell guidance.

All logic lives in :mod:`screener.calendar` (pure, offline, no streamlit); this
module only RENDERS its output and reuses the app's existing not-advice disclaimer
copy from :mod:`screener.display`. The bundled CSV read + ``days_until`` recompute
are memoized per ``cache_day`` via :func:`screener.ui.caching.events_upcoming`, the
same date-keyed warm-read model as the per-symbol detail memos.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from screener import calendar, display
from screener.ui.caching import events_upcoming

# Forex Factory / Investing.com convention: a colored dot per impact tier.
_IMPACT_TAG = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟡 Low"}


def _impact_tag(impact: str) -> str:
    """Human label for an impact tier ('high'/'medium'/'low'), fail-soft to title-case."""
    return _IMPACT_TAG.get(str(impact).strip().lower(), str(impact).strip().title())


def render_events_panel() -> None:
    """Render the on-open "Upcoming events" expander near the top of the main area.

    Memoized per ``cache_day`` (the events CSV never changes within a session-day),
    with a high-impact-only toggle, a 'days until' countdown column, an impact tag,
    and a ⚠ warning flag for events inside the advance-warning window. Fail-soft: an
    empty/absent calendar renders a single caption rather than crashing.
    """
    cache_day = dt.date.today().isoformat()
    events = events_upcoming(cache_day, calendar.DEFAULT_HORIZON_DAYS)

    label = (
        "📅 Upcoming events — heightened expected volatility"
        if not events.empty
        else "📅 Upcoming events"
    )
    with st.expander(label, expanded=False):
        if events.empty:
            st.caption(
                "No scheduled macro events within the next "
                f"{calendar.DEFAULT_HORIZON_DAYS} days (or the calendar is unavailable)."
            )
            st.caption(display.DISCLAIMER_TEXT)
            return

        # High-impact-only toggle (Forex Factory / Investing.com filter convention):
        # default ON so the panel leads with the market-moving FOMC/CPI/jobs tier.
        high_only = st.checkbox(
            "High-impact only", value=True, key="events_high_only",
            help="Show only the red-tier US macro releases (FOMC, CPI, jobs report).",
        )
        view = events[events["impact"] == "high"].copy() if high_only else events.copy()
        if view.empty:
            st.caption("No high-impact macro events in the window — untick to see all.")
            st.caption(display.DISCLAIMER_TEXT)
            return

        # Build the display frame from the pure calendar output: a countdown string,
        # the impact tag, and an advance-warning flag for imminent events. All the
        # date/tier math came from screener.calendar; this is presentation only.
        rows = []
        for _, ev in view.iterrows():
            d = int(ev["days_until"])
            countdown = "Today" if d == 0 else ("Tomorrow" if d == 1 else f"in {d} days")
            warn = "⚠" if calendar.advance_warning(d) else ""
            rows.append(
                {
                    "When": f"{ev['date']} ({countdown})",
                    "Time (ET)": str(ev["time_et"]),
                    "Event": str(ev["event"]),
                    "Impact": _impact_tag(ev["impact"]),
                    "Warning": warn,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "When": st.column_config.TextColumn("When", width="medium"),
                "Time (ET)": st.column_config.TextColumn("Time (ET)", width="small"),
                "Event": st.column_config.TextColumn("Event", width="large"),
                "Impact": st.column_config.TextColumn("Impact", width="small"),
                "Warning": st.column_config.TextColumn(
                    "Warning",
                    help=f"⚠ marks events within {calendar.DEFAULT_WARN_WITHIN_DAYS} days.",
                    width="small",
                ),
            },
        )
        # The Fed publishes FOMC dates but qualifies them as tentative until confirmed.
        if bool(view["tentative"].any()):
            st.caption("FOMC dates are tentative until confirmed by the Federal Reserve.")
        st.caption(
            "Scheduled events mark windows of heightened expected volatility — "
            "descriptive context, not a signal to act."
        )
        st.caption(display.DISCLAIMER_TEXT)
