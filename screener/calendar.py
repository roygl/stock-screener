"""Economic-event calendar (pure, offline).

A bundled, public-domain CSV of US macro release dates (FOMC rate decisions, CPI
prints, the monthly jobs report) plus a fold-in for the per-ticker earnings date
the app already fetches via :mod:`screener.provider`. The calendar exists to flag
windows of *heightened expected volatility* around scheduled events — it is
descriptive, never buy/sell guidance.

The loader mirrors :func:`screener.universe.load_universe`'s ``@lru_cache`` CSV
pattern but is deliberately FAIL-SOFT: a missing, empty, or malformed CSV (or any
read error) yields an empty canonical frame rather than raising, so the optional
calendar never crashes the app. No streamlit, no network, nothing fetched at
import time; ``as_of`` is always supplied by the caller so results stay
deterministic and testable.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import pandas as pd

# data/economic_events.csv lives one level up from this file's package.
EVENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "economic_events.csv"

# Canonical CSV schema (column order preserved on the returned frame).
EVENT_COLUMNS = ("date", "time_et", "event", "category", "impact", "tentative")

# Categories that always read red-tier (US FOMC rate decision / CPI / jobs report).
_HIGH_IMPACT = frozenset({"FOMC", "CPI", "JOBS"})

DEFAULT_HORIZON_DAYS = 45
DEFAULT_WARN_WITHIN_DAYS = 7


def _empty_events() -> pd.DataFrame:
    """An empty events frame with the canonical columns and a 'date' object dtype."""
    df = pd.DataFrame({c: [] for c in EVENT_COLUMNS})
    df["date"] = pd.to_datetime(df["date"]).dt.date  # empty object col of dt.date
    return df


@lru_cache(maxsize=1)
def load_events(path: str | Path | None = None) -> pd.DataFrame:
    """Load the bundled macro-event table from the static CSV.

    Returns a DataFrame with columns date, time_et, event, category, impact,
    tentative — 'date' parsed to ``datetime.date`` objects, 'category' upper-cased,
    'impact' lower-cased, 'tentative' coerced to ``bool``, sorted ascending by date.
    Cached so repeated calls in a single session don't re-read the file. FAIL-SOFT:
    a missing, empty, or malformed CSV (or any read error) yields an empty canonical
    frame rather than raising, so the optional calendar never crashes the app.
    """
    csv_path = Path(path) if path is not None else EVENTS_PATH
    if not csv_path.exists():
        return _empty_events()
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except Exception:  # noqa: BLE001 - unreadable/garbage CSV -> empty, never raise
        return _empty_events()
    df.columns = [c.strip().lower() for c in df.columns]
    if any(c not in df.columns for c in EVENT_COLUMNS):
        return _empty_events()  # missing a required column -> degrade, don't raise
    df = df[list(EVENT_COLUMNS)].copy()
    parsed = pd.to_datetime(df["date"], errors="coerce")
    df = df[parsed.notna()].copy()
    df["date"] = parsed[parsed.notna()].dt.date
    df["category"] = df["category"].str.strip().str.upper()
    df["impact"] = df["impact"].str.strip().str.lower()
    df["tentative"] = (
        df["tentative"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )
    return df.sort_values("date").reset_index(drop=True)


def impact_for(category: str) -> str:
    """Impact tier for an event category: 'high' for FOMC/CPI/JOBS, else 'medium'.

    Pure and case-insensitive. An unknown/empty category defaults to 'medium' (a
    single-name earnings report is material but not a market-wide macro event).
    """
    return "high" if str(category).strip().upper() in _HIGH_IMPACT else "medium"


def advance_warning(days_until: int, *, within_days: int = DEFAULT_WARN_WITHIN_DAYS) -> bool:
    """True when an event is imminent: ``0 <= days_until <= within_days``.

    A past event (negative ``days_until``) is NOT a warning; the event-day itself
    (``0``) is.
    """
    return 0 <= int(days_until) <= int(within_days)


def upcoming_events(as_of: dt.date, *, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    """Future events within ``horizon_days`` of ``as_of``, with a 'days_until' column.

    ``as_of`` is supplied by the caller (never ``date.today()`` here) so the result
    is deterministic/testable. Keeps rows with ``as_of <= date <= as_of + horizon``,
    adds an integer ``days_until`` (``(date - as_of).days``), sorts ascending by date
    and resets the index. An empty source frame yields an empty frame with the same
    columns plus ``days_until``.
    """
    df = load_events()
    out_cols = list(EVENT_COLUMNS) + ["days_until"]
    if df.empty:
        empty = _empty_events()
        empty["days_until"] = pd.Series([], dtype="int64")
        return empty[out_cols]
    horizon_end = as_of + dt.timedelta(days=int(horizon_days))
    mask = df["date"].map(lambda d: as_of <= d <= horizon_end)
    win = df[mask].copy()
    win["days_until"] = win["date"].map(lambda d: (d - as_of).days).astype("int64")
    return win.sort_values("date").reset_index(drop=True)[out_cols]


def next_event_for_symbol(
    symbol: str,
    earnings_date: dt.date | None,
    as_of: dt.date,
    *,
    warn_within_days: int = DEFAULT_WARN_WITHIN_DAYS,
) -> dict | None:
    """Fold a per-ticker earnings date into the same event-risk shape.

    Returns ``{event, date, days_until, impact, within_warning}`` for the symbol's
    next scheduled earnings report, or ``None`` when ``earnings_date`` is ``None`` or
    already past (``< as_of``). ``impact`` is 'medium' (a single-name report, not a
    macro event); ``within_warning`` flags it inside ``warn_within_days``. Network-
    free: the caller (UI layer) supplies ``earnings_date`` from the provider.
    """
    if earnings_date is None:
        return None
    days_until = (earnings_date - as_of).days
    if days_until < 0:
        return None
    return {
        "event": f"{str(symbol).strip().upper()} earnings",
        "date": earnings_date,
        "days_until": int(days_until),
        "impact": "medium",
        "within_warning": advance_warning(days_until, within_days=warn_within_days),
    }


if __name__ == "__main__":
    ev = load_events()
    print(f"Loaded {len(ev)} macro events.")
    print(upcoming_events(dt.date.today()).to_string(index=False))
