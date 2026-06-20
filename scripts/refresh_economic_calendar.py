#!/usr/bin/env python
"""Refresh ``data/economic_events.csv`` — the bundled macro-event table.

The screener surfaces upcoming high-impact US macro events (FOMC rate decisions,
CPI, Employment Situation/jobs) as *educational event risk*. Those release dates
are public-domain government facts, so we BUNDLE them in ``data/economic_events.csv``
and read that file at runtime — there is **no live fetch in the app**. (Streamlit
Community Cloud sleeps when idle and has no cron / background jobs, and bls.gov
bot-blocks datacenter IPs with an Akamai 403, so a runtime fetch would fail there.)

This script is the **manual, run-locally** refresh step. Run it **about once a
year** — after the Fed publishes the next year's FOMC calendar, and whenever you
want to roll the BLS feed forward (BLS publishes ~1 year ahead). It is NOT invoked
by the app and NOT on a schedule; there is nowhere to cron it.

What it does
------------
1. Downloads the BLS news-release iCalendar feed and parses the CPI and Employment
   Situation release dates from it (an inline ``.ics`` line parser; no extra deps).
2. Reads the FOMC dates from :data:`FOMC_EVENTS` below — a hand-maintained constant
   transcribed from the Federal Reserve FOMC calendar page, which publishes no
   machine-readable export. **Edit that constant by hand** when the Fed posts a new
   year.
3. Merges, sorts ascending by date, and rewrites ``data/economic_events.csv`` with
   the exact columns the app expects: ``date,time_et,event,category,impact,tentative``.

If the BLS download fails (e.g. the 403 above, or no network), the script keeps the
CPI/JOBS rows already in the existing CSV so a refresh of just the FOMC constant
still works offline. As a documented fallback when bls.gov 403s, point
:data:`BLS_ICS_URL` at a Wayback Machine snapshot, e.g.
``http://web.archive.org/web/20260612174939id_/https://www.bls.gov/schedule/news_release/bls.ics``.

Usage
-----
    python scripts/refresh_economic_calendar.py

Stdlib only (``urllib`` for the fetch, an inline ``.ics`` parser); ``pandas`` is
used for the CSV write since it is already a project dependency. Import-clean: it
does no work at import time — only ``main()`` touches the network or the filesystem.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

# data/economic_events.csv lives one level up from this script's directory.
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "economic_events.csv"

# Exact column order the app's loader expects.
COLUMNS = ["date", "time_et", "event", "category", "impact", "tentative"]

# BLS news-release iCalendar feed (CPI + Employment Situation, ~8:30 AM ET).
# NOTE: bls.gov bot-blocks datacenter IPs (Akamai 403). If a direct fetch 403s,
# replace this with a Wayback Machine snapshot of the same .ics (see module docstring).
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"

# A browser-ish User-Agent — bls.gov rejects the default urllib agent outright.
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Map a BLS .ics SUMMARY to our (event label, category). Only these two BLS
# releases are high-impact for a US large-cap screener; everything else is ignored.
_BLS_SUMMARY_MAP = {
    "Consumer Price Index": ("CPI release", "CPI"),
    "Employment Situation": ("Employment Situation (jobs report)", "JOBS"),
}
_BLS_TIME_ET = "08:30"

# ---------------------------------------------------------------------------
# FOMC — hand-maintained. The Federal Reserve publishes no machine-readable
# export of the FOMC calendar (federalreserve.gov/monetarypolicy/fomccalendars.htm),
# so transcribe the rate-decision dates (the SECOND day of each two-day meeting,
# ~2:00 PM ET) here. The Fed marks future dates "tentative until confirmed", hence
# tentative=true. Drop dates as they pass and ADD the new year when the Fed posts it.
# (date, time_et) pairs:
# ---------------------------------------------------------------------------
FOMC_EVENTS = [
    ("2026-07-29", "14:00"),
    ("2026-09-16", "14:00"),
    ("2026-10-28", "14:00"),
    ("2026-12-09", "14:00"),
    ("2027-01-27", "14:00"),
    ("2027-03-17", "14:00"),
    ("2027-04-28", "14:00"),
    ("2027-06-09", "14:00"),
    ("2027-07-28", "14:00"),
    ("2027-09-15", "14:00"),
    ("2027-10-27", "14:00"),
    ("2027-12-08", "14:00"),
]


def _unfold_ics(text: str) -> list[str]:
    """Unfold iCalendar content lines.

    Per RFC 5545, a long line may be split across multiple physical lines, each
    continuation beginning with a single space or tab. Join those back so each
    logical property (e.g. ``SUMMARY:...``) is one line.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _parse_bls_ics(text: str) -> list[dict]:
    """Parse CPI + Employment Situation release dates from a BLS ``.ics`` string.

    A tiny stdlib parser: walk VEVENT blocks, read ``SUMMARY`` and ``DTSTART``,
    keep only the two summaries in :data:`_BLS_SUMMARY_MAP`. Returns event-row
    dicts (unsorted, dates as ``YYYY-MM-DD``). ``DTSTART`` values may carry params
    (e.g. ``DTSTART;VALUE=DATE:20260714``) and a date-only or datetime value; we
    take the leading 8 digits ``YYYYMMDD``.
    """
    rows: list[dict] = []
    summary: str | None = None
    dtstart: str | None = None
    in_event = False

    for line in _unfold_ics(text):
        if line.startswith("BEGIN:VEVENT"):
            in_event, summary, dtstart = True, None, None
        elif line.startswith("END:VEVENT"):
            if summary in _BLS_SUMMARY_MAP and dtstart:
                event, category = _BLS_SUMMARY_MAP[summary]
                rows.append(
                    {
                        "date": dtstart,
                        "time_et": _BLS_TIME_ET,
                        "event": event,
                        "category": category,
                        "impact": "high",
                        "tentative": "false",
                    }
                )
            in_event, summary, dtstart = False, None, None
        elif in_event:
            # Property name is everything before the first ':' (params use ';').
            name, _, value = line.partition(":")
            key = name.split(";", 1)[0].strip().upper()
            if key == "SUMMARY":
                summary = value.strip()
            elif key == "DTSTART":
                digits = "".join(ch for ch in value if ch.isdigit())[:8]
                if len(digits) == 8:
                    dtstart = f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return rows


def _fetch_bls_ics(url: str = BLS_ICS_URL) -> str:
    """Download the BLS ``.ics`` feed as text (raises on any HTTP/network error)."""
    req = Request(url, headers=_HTTP_HEADERS)
    with urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed, trusted gov/archive URL
        return resp.read().decode("utf-8", errors="replace")


def _fomc_rows() -> list[dict]:
    """Build FOMC event-row dicts from the hand-maintained :data:`FOMC_EVENTS`."""
    return [
        {
            "date": date,
            "time_et": time_et,
            "event": "FOMC rate decision",
            "category": "FOMC",
            "impact": "high",
            "tentative": "true",
        }
        for date, time_et in FOMC_EVENTS
    ]


def _existing_bls_rows(csv_path: Path) -> list[dict]:
    """Fallback CPI/JOBS rows from the current CSV (used when the fetch fails)."""
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "category" not in df.columns:
        return []
    keep = df[df["category"].isin(["CPI", "JOBS"])]
    return keep[COLUMNS].to_dict("records") if not keep.empty else []


def build_events(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Assemble the full event table: live BLS CPI/JOBS + the FOMC constant.

    Tries the live BLS fetch; on any failure, falls back to the CPI/JOBS rows
    already in ``csv_path`` so a FOMC-only refresh still works offline. Returns a
    DataFrame with :data:`COLUMNS`, sorted ascending by date.
    """
    try:
        bls_rows = _parse_bls_ics(_fetch_bls_ics())
        if not bls_rows:
            raise ValueError("BLS feed parsed to zero CPI/JOBS rows")
        print(f"Fetched {len(bls_rows)} CPI/JOBS rows from BLS.")
    except Exception as exc:  # noqa: BLE001 - degrade to the bundled rows
        bls_rows = _existing_bls_rows(csv_path)
        print(
            f"WARNING: BLS fetch/parse failed ({exc}); "
            f"keeping {len(bls_rows)} CPI/JOBS rows from the existing CSV.",
            file=sys.stderr,
        )

    rows = bls_rows + _fomc_rows()
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = (
        df.drop_duplicates(subset=["date", "category"])
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    return df


def main() -> None:
    """Rebuild and rewrite ``data/economic_events.csv``."""
    df = build_events(CSV_PATH)
    df.to_csv(CSV_PATH, index=False)
    counts = df["category"].value_counts().to_dict()
    print(
        f"Wrote {len(df)} events to {CSV_PATH} "
        f"(FOMC: {counts.get('FOMC', 0)}, "
        f"CPI: {counts.get('CPI', 0)}, "
        f"JOBS: {counts.get('JOBS', 0)})."
    )


if __name__ == "__main__":
    main()
