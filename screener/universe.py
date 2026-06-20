"""Universe loading.

The *universe* is the set of tickers the screener ranks. For v1 it is a static
S&P 500 list shipped in `data/universe.csv`. Later milestones may swap this for a
live constituent feed, but the rest of the app only depends on `load_universe()`
returning a DataFrame with `symbol`, `name`, and `sector` columns.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# data/universe.csv lives one level up from this file's package.
UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.csv"

REQUIRED_COLUMNS = ("symbol", "name", "sector")

# yfinance reports a coarser, differently-spelled sector taxonomy than the GICS
# strings stored in universe.csv. This maps each yfinance sector to the CSV's 11
# canonical GICS sectors so an auto-added ticker lands in the right bucket; an
# unmapped/None yfinance sector falls back to "" (load_universe tolerates blanks).
_YF_TO_GICS: dict[str, str] = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}

# A plausible US ticker: 1-5 letters, with an optional dot/dash class suffix
# (e.g. BRK.B, BF-B). Guards ensure_symbol against fetching prose tokens.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,4})?$")


@lru_cache(maxsize=1)
def load_universe(path: str | Path | None = None) -> pd.DataFrame:
    """Load the ticker universe from the static CSV.

    Returns a DataFrame with columns: symbol, name, sector.
    Cached so repeated calls in a single session don't re-read the file.
    """
    csv_path = Path(path) if path is not None else UNIVERSE_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Universe file not found at {csv_path}. "
            "Expected a static CSV with columns: symbol, name, sector."
        )

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Universe file {csv_path} is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df[list(REQUIRED_COLUMNS)].copy()
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df = df[df["symbol"] != ""].drop_duplicates(subset="symbol").reset_index(drop=True)
    return df


def tickers(path: str | Path | None = None) -> list[str]:
    """Convenience: just the list of ticker symbols."""
    return load_universe(path)["symbol"].tolist()


def universe_slice(
    universe_df: pd.DataFrame, n_names: int, include_symbol: str = ""
) -> pd.DataFrame:
    """The first ``n_names`` rows, guaranteeing ``include_symbol`` is in the slice.

    PURE — no I/O, no network. Returns ``universe_df.head(n_names)`` unchanged,
    except: when ``include_symbol`` (stripped/upper) names a row that EXISTS in
    ``universe_df`` but falls OUTSIDE the head slice, that one row is appended so a
    natural-language query for a specific ticker can scan it even when it ranks past
    the chosen size. No-op when ``include_symbol`` is empty, already in the slice, or
    not in ``universe_df`` — so a normal scan (``include_symbol=""``) gets the plain
    head and shares the cache.
    """
    head = universe_df.head(n_names)
    sym = str(include_symbol).strip().upper()
    if not sym:
        return head
    if sym in set(head["symbol"]):
        return head
    extra = universe_df[universe_df["symbol"] == sym]
    if extra.empty:
        return head
    return pd.concat([head, extra], ignore_index=True)


def append_symbol(
    symbol, name, sector, *, path: str | Path | None = None
) -> bool:
    """Append one ``{symbol, name, sector}`` row to the universe CSV.

    PURE write — no network. ``path`` defaults to :data:`UNIVERSE_PATH`, RESOLVED
    at call time (the ``None`` sentinel idiom :func:`load_universe` uses) so a test
    that monkeypatches ``UNIVERSE_PATH`` is honored — a function-default bound at
    import time would not be.

    Canonicalizes ``symbol`` (strip/upper) and returns ``False`` for an empty symbol
    or one already present in the universe (the cached :func:`load_universe`, read
    from ``path`` so a temp file is respected), leaving the file untouched. Otherwise
    it re-reads the existing CSV, appends the row (column order ``symbol, name,
    sector``), drops duplicates keeping the FIRST occurrence (so existing rows + their
    order win), writes back with ``to_csv(index=False)`` — which quotes any name
    containing a comma — then clears the :func:`load_universe` cache so the new symbol
    becomes visible. Returns ``True`` on a successful append.
    """
    csv_path = Path(path) if path is not None else UNIVERSE_PATH
    sym = str(symbol).strip().upper()
    if not sym:
        return False
    if sym in set(load_universe(csv_path)["symbol"]):
        return False

    existing = pd.read_csv(csv_path, dtype=str).fillna("")
    new_row = pd.DataFrame(
        [{"symbol": sym, "name": str(name), "sector": str(sector)}]
    )
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined = combined.drop_duplicates(subset="symbol", keep="first")
    combined.to_csv(csv_path, index=False)
    load_universe.cache_clear()
    return True


def ensure_symbol(symbol, *, path: str | Path | None = None) -> bool:
    """Make ``symbol`` part of the universe, fetching it from the provider if new.

    NEVER raises — every failure path returns ``False``. ``path`` defaults to
    :data:`UNIVERSE_PATH`, resolved at call time (so a monkeypatched ``UNIVERSE_PATH``
    is honored). The flow:

    1. Canonicalize (strip/upper) and validate the shape with :data:`_TICKER_RE`
       (1-5 letters + optional class suffix); a non-ticker token (prose, "PE", a
       sentence) returns ``False`` with NO network call.
    2. If it is ALREADY in the universe, return ``False`` immediately — again with
       no network call (the common case for an in-universe NL query).
    3. Otherwise fetch fundamentals via the data provider (imported LAZILY here so
       this module stays import-cheap). If the snapshot ``is_empty`` or has no
       ``name``, return ``False`` — we refuse to poison the universe with a bogus
       ticker that yielded nothing.
    4. Map the yfinance sector to a GICS string via :data:`_YF_TO_GICS` (``""`` when
       unmapped/None) and delegate to :func:`append_symbol`, returning its result.

    Any unexpected exception is swallowed and reported as ``False``.
    """
    csv_path = Path(path) if path is not None else UNIVERSE_PATH
    sym = str(symbol).strip().upper()
    if not _TICKER_RE.match(sym):
        return False
    try:
        # Membership check is inside the guard too, so even a bad/missing CSV path
        # degrades to False rather than raising. Still runs before the provider
        # import, so an already-present symbol never triggers a network call.
        if sym in set(load_universe(csv_path)["symbol"]):
            return False
        from screener.provider import YFinanceProvider

        f = YFinanceProvider().fundamentals(sym)
        if f.is_empty or not f.name:
            return False
        sector = _YF_TO_GICS.get(f.sector, "")
        return append_symbol(sym, f.name, sector, path=csv_path)
    except Exception:  # noqa: BLE001 - ensure_symbol must never raise
        return False


if __name__ == "__main__":
    u = load_universe()
    print(f"Loaded {len(u)} tickers across {u['sector'].nunique()} sectors.")
    print(u.head(10).to_string(index=False))
