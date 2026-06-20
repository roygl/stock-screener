"""Deterministic, network-free tests for the calendar engine (screener/calendar.py).

No ``pytest`` import and no ``yfinance``: every test is a plain ``test_*`` function
using ``assert`` so the suite runs BOTH under
``python -m pytest tests/test_calendar.py`` AND standalone as
``python tests/test_calendar.py`` (the ``__main__`` runner counts pass/fail,
prints a summary, and exits non-zero on any failure).

All dates are checked against a FIXED ``AS_OF`` so nothing depends on the wall
clock; the loader is fail-soft so the missing/malformed-file cases assert no raise.
The bundled CSV (``data/economic_events.csv``) is exercised directly, while the
edge cases write temp CSVs and point the loader at them via a monkeypatched
``EVENTS_PATH`` (clearing the ``@lru_cache`` around each redirect).
"""

import datetime as dt
import os
import sys
import tempfile

# Put the repo root on sys.path so "import screener" resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from screener import calendar as cal  # noqa: E402

# Fixed reference date so days_until / horizon results are deterministic.
AS_OF = dt.date(2026, 6, 20)

# The real bundled path, captured once so redirect tests can restore it.
_BUNDLED_PATH = cal.EVENTS_PATH


# --- helpers -------------------------------------------------------------
def _load_bundled() -> pd.DataFrame:
    """The real bundled CSV, with a fresh cache so prior redirects don't leak."""
    cal.EVENTS_PATH = _BUNDLED_PATH
    cal.load_events.cache_clear()
    return cal.load_events()


def _redirect_to(tmp_csv: str) -> pd.DataFrame:
    """Point the loader at ``tmp_csv`` by monkeypatching EVENTS_PATH + clearing cache.

    Returns the loaded frame. Callers that exercise ``upcoming_events`` rely on this
    redirect since those helpers read the cached default ``load_events()``.
    """
    cal.EVENTS_PATH = type(cal.EVENTS_PATH)(tmp_csv)
    cal.load_events.cache_clear()
    return cal.load_events()


# --- 1. loading the bundled CSV ------------------------------------------
def test_load_events_parses_bundled_csv():
    df = _load_bundled()
    assert len(df) == 24, f"expected 24 bundled events, got {len(df)}"
    assert list(df.columns) == list(cal.EVENT_COLUMNS)
    # 'date' holds python date objects, not Timestamps.
    assert all(isinstance(d, dt.date) and not isinstance(d, dt.datetime) for d in df["date"])
    # categories upper-cased.
    assert set(df["category"]) <= {"FOMC", "CPI", "JOBS"}
    assert df["category"].str.isupper().all()
    # tentative coerced to bool: FOMC rows are tentative, CPI/JOBS rows are not.
    assert df["tentative"].dtype == bool
    fomc = df[df["category"] == "FOMC"]
    cpi_jobs = df[df["category"].isin({"CPI", "JOBS"})]
    assert fomc["tentative"].all()
    assert not cpi_jobs["tentative"].any()
    # sorted ascending by date.
    assert list(df["date"]) == sorted(df["date"])


# --- 2 & 3. fail-soft loader ---------------------------------------------
def test_load_events_missing_file_is_empty():
    with tempfile.TemporaryDirectory() as d:
        df = _redirect_to(os.path.join(d, "nope.csv"))
    assert df.empty
    assert list(df.columns) == list(cal.EVENT_COLUMNS)


def test_load_events_malformed_is_empty():
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "bad.csv")
        # Missing the required 'category' (and others) column.
        with open(bad, "w") as fh:
            fh.write("date,event\n2026-07-02,jobs report\n")
        df = _redirect_to(bad)
    assert df.empty
    assert list(df.columns) == list(cal.EVENT_COLUMNS)


# --- 4. days_until computation -------------------------------------------
def test_days_until_computation():
    _load_bundled()
    up = cal.upcoming_events(AS_OF, horizon_days=60)
    by_date = {d: n for d, n in zip(up["date"], up["days_until"])}
    assert by_date[dt.date(2026, 7, 2)] == 12   # jobs report
    assert by_date[dt.date(2026, 7, 14)] == 24  # CPI release
    assert up["days_until"].dtype == "int64"


# --- 5. horizon filtering ------------------------------------------------
def test_horizon_filtering():
    _load_bundled()
    up30 = cal.upcoming_events(AS_OF, horizon_days=30)
    dates = set(up30["date"])
    # 07-20 cutoff: keeps 07-02 JOBS + 07-14 CPI, drops 07-29 FOMC (day 39).
    assert dt.date(2026, 7, 2) in dates
    assert dt.date(2026, 7, 14) in dates
    assert dt.date(2026, 7, 29) not in dates
    # A 1-day window keeps nothing (nearest event is 12 days out).
    assert cal.upcoming_events(AS_OF, horizon_days=1).empty


# --- 6. past events excluded ---------------------------------------------
def test_upcoming_excludes_past():
    _load_bundled()
    later = dt.date(2026, 8, 1)  # after the first two bundled rows
    up = cal.upcoming_events(later, horizon_days=60)
    assert dt.date(2026, 7, 2) not in set(up["date"])
    assert dt.date(2026, 7, 14) not in set(up["date"])
    assert (up["days_until"] >= 0).all()
    assert list(up["date"]) == sorted(up["date"])
    assert up["days_until"].dtype == "int64"


# --- 7. impact tagging ---------------------------------------------------
def test_impact_tagging():
    assert cal.impact_for("FOMC") == "high"
    assert cal.impact_for("cpi") == "high"
    assert cal.impact_for(" JOBS ") == "high"
    assert cal.impact_for("EARNINGS") == "medium"
    assert cal.impact_for("") == "medium"


# --- 8. advance warning --------------------------------------------------
def test_advance_warning():
    assert cal.advance_warning(0) is True
    assert cal.advance_warning(cal.DEFAULT_WARN_WITHIN_DAYS) is True
    assert cal.advance_warning(cal.DEFAULT_WARN_WITHIN_DAYS + 1) is False
    assert cal.advance_warning(-1) is False
    # custom window honored.
    assert cal.advance_warning(10, within_days=14) is True
    assert cal.advance_warning(15, within_days=14) is False


# --- 9. next_event_for_symbol with earnings ------------------------------
def test_next_event_for_symbol_with_earnings():
    near = cal.next_event_for_symbol("aapl", dt.date(2026, 6, 25), AS_OF)
    assert near == {
        "event": "AAPL earnings",
        "date": dt.date(2026, 6, 25),
        "days_until": 5,
        "impact": "medium",
        "within_warning": True,
    }
    far = cal.next_event_for_symbol("AAPL", dt.date(2026, 9, 1), AS_OF)
    assert far is not None
    assert far["within_warning"] is False
    assert far["days_until"] == (dt.date(2026, 9, 1) - AS_OF).days


# --- 10. next_event_for_symbol without earnings --------------------------
def test_next_event_for_symbol_without_earnings():
    assert cal.next_event_for_symbol("AAPL", None, AS_OF) is None
    # A past earnings date is dropped (no negative days_until surfaced).
    assert cal.next_event_for_symbol("AAPL", dt.date(2026, 6, 1), AS_OF) is None


def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report any unexpected error
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
    total = passed + failed
    print(f"\n{passed}/{total} passed" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
