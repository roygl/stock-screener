"""Universe loading.

The *universe* is the set of tickers the screener ranks. For v1 it is a static
S&P 500 list shipped in `data/universe.csv`. Later milestones may swap this for a
live constituent feed, but the rest of the app only depends on `load_universe()`
returning a DataFrame with `symbol`, `name`, and `sector` columns.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

# data/universe.csv lives one level up from this file's package.
UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.csv"

REQUIRED_COLUMNS = ("symbol", "name", "sector")


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


if __name__ == "__main__":
    u = load_universe()
    print(f"Loaded {len(u)} tickers across {u['sector'].nunique()} sectors.")
    print(u.head(10).to_string(index=False))
