"""Data-provider interface and the v1 ``yfinance`` implementation.

The rest of the app talks to a :class:`DataProvider`, never to ``yfinance``
directly, so a paid API (e.g. FMP) can be swapped in later without touching the
indicator or ranking engines (see DECISIONS.md).

Three pieces of per-ticker data feed the screener:
- :meth:`price_history`  — daily OHLCV bars (momentum, MAs, RSI, MACD, rel-vol).
- :meth:`fundamentals`   — valuation / growth / classification snapshot.
- :meth:`earnings_date`  — next scheduled report, for the swing earnings-window flag.

``yfinance`` is unofficial and scraping-based, so every fetch is wrapped: a
failure for one ticker yields an empty frame / ``None`` rather than aborting a
whole-universe scan. Results are cached locally (see :mod:`screener.cache`) so a
second run the same day never re-hits Yahoo.
"""

from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Iterable, Mapping

import pandas as pd

from .cache import Cache

log = logging.getLogger(__name__)

# OHLCV schema for every price frame, lowercased to match the universe columns.
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

# ~2y of daily bars: enough for 12-mo momentum, the 150-day SMA, 52-wk high, and
# MACD/EMA warm-up, with room for the historical series M3 needs.
DEFAULT_LOOKBACK_DAYS = 730


@dataclass(frozen=True)
class Fundamentals:
    """Valuation / growth / classification snapshot for one ticker.

    Every field except ``symbol`` is optional — ``yfinance`` routinely omits some
    keys — so consumers must tolerate ``None``.
    """

    symbol: str
    name: str | None = None
    sector: str | None = None
    market_cap: float | None = None
    forward_pe: float | None = None
    trailing_pe: float | None = None
    revenue_growth: float | None = None   # latest reported YoY, as a fraction (0.12 = 12%)
    earnings_growth: float | None = None  # latest reported YoY, as a fraction

    @property
    def is_empty(self) -> bool:
        """True when nothing beyond the symbol came back (don't cache these)."""
        return all(getattr(self, f.name) is None for f in fields(self) if f.name != "symbol")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "Fundamentals":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def empty_price_frame() -> pd.DataFrame:
    """An empty OHLCV frame with the canonical schema and a named date index."""
    return pd.DataFrame(columns=PRICE_COLUMNS, index=pd.DatetimeIndex([], name="date"))


class DataProvider(ABC):
    """Swappable source of price history, fundamentals, and earnings dates."""

    @abstractmethod
    def price_history(self, symbol: str, *, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
        """Daily OHLCV bars indexed by tz-naive ``date`` (oldest first).

        Returns an empty frame (canonical columns) when data is unavailable.
        """

    @abstractmethod
    def fundamentals(self, symbol: str) -> Fundamentals:
        """Fundamentals snapshot; fields are ``None`` when missing."""

    @abstractmethod
    def earnings_date(self, symbol: str) -> dt.date | None:
        """Next scheduled earnings date, or ``None`` if unknown."""

    # Convenience fan-outs over the universe. Concrete here (a simple loop) so any
    # provider gets them free; per-symbol caching keeps repeat runs cheap.
    def bulk_price_history(
        self, symbols: Iterable[str], *, lookback_days: int = DEFAULT_LOOKBACK_DAYS
    ) -> dict[str, pd.DataFrame]:
        return {s: self.price_history(s, lookback_days=lookback_days) for s in symbols}

    def bulk_fundamentals(self, symbols: Iterable[str]) -> dict[str, Fundamentals]:
        return {s: self.fundamentals(s) for s in symbols}


class YFinanceProvider(DataProvider):
    """``yfinance``-backed provider with a local, date-keyed cache."""

    def __init__(self, cache: Cache | None = None, *, use_cache: bool = True) -> None:
        self._cache = cache if cache is not None else Cache()
        self._use_cache = use_cache

    # --- public API ------------------------------------------------------
    def price_history(self, symbol: str, *, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
        sym = _canonical(symbol)
        if self._use_cache:
            cached = self._cache.get_frame("prices", sym)
            if cached is not None:
                return cached
        df = self._fetch_prices(sym, lookback_days)
        if self._use_cache and not df.empty:
            self._cache.put_frame("prices", sym, df)
        return df

    def fundamentals(self, symbol: str) -> Fundamentals:
        sym = _canonical(symbol)
        if self._use_cache:
            cached = self._cache.get_json("fundamentals", sym)
            if cached is not None:
                return Fundamentals.from_dict(cached)
        snap = self._fetch_fundamentals(sym)
        if self._use_cache and not snap.is_empty:
            self._cache.put_json("fundamentals", sym, snap.to_dict())
        return snap

    def earnings_date(self, symbol: str) -> dt.date | None:
        sym = _canonical(symbol)
        if self._use_cache:
            cached = self._cache.get_json("earnings", sym)
            if cached is not None:
                raw = cached.get("earnings_date")
                return dt.date.fromisoformat(raw) if raw else None
        date = self._fetch_earnings_date(sym)
        # Cache only positive hits: a missing date may just mean "not announced yet".
        if self._use_cache and date is not None:
            self._cache.put_json("earnings", sym, {"earnings_date": date.isoformat()})
        return date

    # --- fetch helpers (network) ----------------------------------------
    def _fetch_prices(self, sym: str, lookback_days: int) -> pd.DataFrame:
        end = dt.date.today() + dt.timedelta(days=1)            # end is exclusive
        start = end - dt.timedelta(days=lookback_days + 1)
        try:
            import yfinance as yf

            raw = yf.Ticker(_yahoo(sym)).history(
                start=start, end=end, interval="1d", auto_adjust=True, actions=False,
            )
        except Exception as exc:
            log.warning("price_history(%s) failed: %s", sym, exc)
            return empty_price_frame()
        return _normalize_prices(raw)

    def _fetch_fundamentals(self, sym: str) -> Fundamentals:
        info = self._info(sym)
        return Fundamentals(
            symbol=sym,
            name=_text(info.get("longName") or info.get("shortName")),
            sector=_text(info.get("sector")),
            market_cap=_num(info.get("marketCap")),
            forward_pe=_num(info.get("forwardPE")),
            trailing_pe=_num(info.get("trailingPE")),
            revenue_growth=_first_num(info, "revenueGrowth"),
            earnings_growth=_first_num(info, "earningsGrowth", "earningsQuarterlyGrowth"),
        )

    def _info(self, sym: str) -> Mapping:
        try:
            import yfinance as yf

            ticker = yf.Ticker(_yahoo(sym))
            getter = getattr(ticker, "get_info", None)
            info = getter() if callable(getter) else ticker.info
            return info if isinstance(info, Mapping) else {}
        except Exception as exc:
            log.warning("fundamentals(%s) failed: %s", sym, exc)
            return {}

    def _fetch_earnings_date(self, sym: str) -> dt.date | None:
        try:
            import yfinance as yf

            ticker = yf.Ticker(_yahoo(sym))
        except Exception as exc:
            log.warning("earnings_date(%s) failed: %s", sym, exc)
            return None

        date = _earnings_from_calendar(_safe_attr(ticker, "calendar"))
        if date is None:
            try:
                date = _earnings_from_dates(ticker.get_earnings_dates(limit=12))
            except Exception:
                date = None
        return date


# --- module-level helpers (pure; easy to unit-test) ----------------------
def _canonical(symbol: str) -> str:
    """Our internal symbol form: upper-case, trimmed (matches universe.csv)."""
    return symbol.strip().upper()


def _yahoo(symbol: str) -> str:
    """Yahoo's form: class shares use a dash, e.g. ``BRK.B`` -> ``BRK-B``."""
    return _canonical(symbol).replace(".", "-")


def _text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _num(value) -> float | None:
    """Coerce to float, rejecting bools, NaN, and non-numerics."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN (NaN != NaN)


def _first_num(info: Mapping, *keys: str) -> float | None:
    """First key that yields a real number — so a valid 0.0 isn't skipped."""
    for key in keys:
        val = _num(info.get(key))
        if val is not None:
            return val
    return None


def _normalize_prices(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return empty_price_frame()
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[PRICE_COLUMNS]
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close"])


def _safe_attr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _coerce_date(value) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    return None if pd.isna(ts) else ts.date()


def _next_on_or_after(dates: Iterable, today: dt.date) -> dt.date | None:
    """Soonest date that is today or later; else the latest known date."""
    valid = sorted(d for d in (_coerce_date(x) for x in dates) if d is not None)
    if not valid:
        return None
    upcoming = [d for d in valid if d >= today]
    return upcoming[0] if upcoming else valid[-1]


def _earnings_from_calendar(calendar) -> dt.date | None:
    """Handle both yfinance shapes: a dict, or a DataFrame indexed by field."""
    if calendar is None:
        return None
    today = dt.date.today()
    if isinstance(calendar, Mapping):
        value = calendar.get("Earnings Date")
        candidates = value if isinstance(value, (list, tuple)) else [value]
        return _next_on_or_after(candidates, today)
    if isinstance(calendar, pd.DataFrame) and "Earnings Date" in getattr(calendar, "index", []):
        return _next_on_or_after(calendar.loc["Earnings Date"].tolist(), today)
    return None


def _earnings_from_dates(frame: pd.DataFrame) -> dt.date | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    return _next_on_or_after(list(frame.index), dt.date.today())


if __name__ == "__main__":  # smoke test: prove the cache serves the 2nd run
    import sys
    import time

    logging.basicConfig(level=logging.WARNING)
    symbols = sys.argv[1:] or ["AAPL", "MSFT", "NVDA"]
    provider = YFinanceProvider()

    for label in ("cold", "warm"):
        t0 = time.time()
        prices = provider.bulk_price_history(symbols)
        funds = provider.bulk_fundamentals(symbols)
        elapsed = time.time() - t0
        rows = {s: len(df) for s, df in prices.items()}
        print(f"[{label}] {elapsed:5.2f}s  rows={rows}")
        for s in symbols:
            f = funds[s]
            print(f"    {s:6} sector={f.sector!r} fwdPE={f.forward_pe} "
                  f"mktcap={f.market_cap} next_earnings={provider.earnings_date(s)}")
