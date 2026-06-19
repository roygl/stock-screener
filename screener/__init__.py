"""Stock screener package.

- M1: universe loader.
- M2: data layer — a swappable ``DataProvider`` (``YFinanceProvider``) behind a
  local, date-keyed ``Cache``.

Later milestones add the indicator engine (M3) and profile/ranking engine (M4).
"""

from .cache import Cache
from .provider import DataProvider, Fundamentals, YFinanceProvider
from .universe import load_universe, tickers

__all__ = [
    "load_universe",
    "tickers",
    "DataProvider",
    "YFinanceProvider",
    "Fundamentals",
    "Cache",
]
