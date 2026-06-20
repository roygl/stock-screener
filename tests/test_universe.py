"""Pure, OFFLINE tests for the universe auto-add machinery (Stage 9).

Covers :mod:`screener.universe`'s three new helpers without the network and
WITHOUT ever mutating the real ``data/universe.csv``:

- :func:`append_symbol` — adds a new row, refuses a duplicate, preserves the
  ``symbol,name,sector`` schema + existing row order (existing wins on dedupe),
  and makes the symbol visible through :func:`load_universe` after the cache clear.
- :func:`universe_slice` — head-only by default; unions an out-of-slice universe
  symbol; no-ops for an in-slice / unknown / empty / case-folded ``include_symbol``.
- :func:`ensure_symbol` — driven by a FAKE provider (so no yfinance / network):
  a valid fundamentals snapshot appends with the GICS-mapped sector; an empty
  snapshot does not append; an already-present symbol returns False and NEVER
  touches the provider; a non-ticker string returns False (also no provider).
- The import-cheapness invariant: :mod:`screener.agent`'s OWN module-level
  imports pull in NEITHER ``yfinance`` NOR ``screener.provider``.

Every test isolates the CSV: each writes a small fixture into ``tmp_path`` and
monkeypatches :data:`screener.universe.UNIVERSE_PATH` to it (and clears the
:func:`load_universe` lru_cache before and after), so the real universe file is
never read or written. Run with ``pytest tests/test_universe.py -q``.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from screener import universe


# --- fixtures ------------------------------------------------------------
# A tiny, ordered universe with one comma-bearing name (to prove CSV quoting)
# and a "deep" symbol that ranks past a small head slice.
_SEED_ROWS = [
    ("MMM", "3M", "Industrials"),
    ("AAPL", "Apple Inc.", "Information Technology"),
    ("BRKB", "Berkshire Hathaway", "Financials"),
    ("CHWY", "Chewy, Inc.", "Consumer Discretionary"),  # comma in the name
    ("ZTS", "Zoetis", "Health Care"),
]


@pytest.fixture()
def universe_csv(tmp_path, monkeypatch):
    """A temp universe.csv wired in via UNIVERSE_PATH, with the load cache reset.

    Yields the temp path. Points :data:`screener.universe.UNIVERSE_PATH` at it so
    the no-``path`` code paths (e.g. :func:`load_universe`'s default) hit the temp
    file too, and clears the lru_cache before and after so neither this test nor a
    neighbour sees a stale frame. The real ``data/universe.csv`` is never touched.
    """
    path = tmp_path / "universe.csv"
    pd.DataFrame(_SEED_ROWS, columns=["symbol", "name", "sector"]).to_csv(
        path, index=False
    )
    monkeypatch.setattr(universe, "UNIVERSE_PATH", path)
    universe.load_universe.cache_clear()
    try:
        yield path
    finally:
        universe.load_universe.cache_clear()


def _make_stub(snapshot):
    """Build a stub YFinanceProvider class whose ``fundamentals`` returns ``snapshot``.

    The returned class records every symbol passed to :meth:`fundamentals` on its
    ``.calls`` list, so a test can assert the provider was (or was never) consulted.
    No network — this is what ``ensure_symbol`` calls once monkeypatched in.
    """
    calls: "list[str]" = []

    class _P:
        def __init__(self, *args, **kwargs):
            pass

        def fundamentals(self, symbol):
            calls.append(symbol)
            return snapshot

    _P.calls = calls
    return _P


# --- append_symbol -------------------------------------------------------
def test_append_symbol_adds_new_row(universe_csv):
    assert universe.append_symbol("NVDA", "NVIDIA Corp.", "Information Technology") is True
    df = pd.read_csv(universe_csv, dtype=str)
    assert "NVDA" in set(df["symbol"])
    row = df[df["symbol"] == "NVDA"].iloc[0]
    assert row["name"] == "NVIDIA Corp."
    assert row["sector"] == "Information Technology"


def test_append_symbol_canonicalizes_symbol(universe_csv):
    # Lower-case / padded input is stored upper-cased + trimmed.
    assert universe.append_symbol("  nvda ", "NVIDIA Corp.", "Information Technology") is True
    df = pd.read_csv(universe_csv, dtype=str)
    assert "NVDA" in set(df["symbol"])
    assert "nvda" not in set(df["symbol"])


def test_append_symbol_empty_symbol_refused(universe_csv):
    before = pd.read_csv(universe_csv, dtype=str)
    assert universe.append_symbol("   ", "Whatever", "Energy") is False
    after = pd.read_csv(universe_csv, dtype=str)
    assert after.equals(before)  # file untouched


def test_append_symbol_refuses_duplicate(universe_csv):
    # AAPL is already seeded; a re-add is a no-op and must not duplicate the row.
    assert universe.append_symbol("AAPL", "Apple DIFFERENT", "Energy") is False
    df = pd.read_csv(universe_csv, dtype=str)
    assert list(df["symbol"]).count("AAPL") == 1
    # Existing name/sector win — the bogus re-add values are NOT written.
    row = df[df["symbol"] == "AAPL"].iloc[0]
    assert row["name"] == "Apple Inc."
    assert row["sector"] == "Information Technology"


def test_append_symbol_preserves_schema_and_order(universe_csv):
    universe.append_symbol("NVDA", "NVIDIA Corp.", "Information Technology")
    df = pd.read_csv(universe_csv, dtype=str)
    # Columns unchanged + in order.
    assert list(df.columns) == ["symbol", "name", "sector"]
    # All seed symbols stay, in their original order, with the new one appended last.
    assert list(df["symbol"]) == ["MMM", "AAPL", "BRKB", "CHWY", "ZTS", "NVDA"]
    # The comma-bearing name round-trips intact (pandas quoted it on write).
    chwy = df[df["symbol"] == "CHWY"].iloc[0]
    assert chwy["name"] == "Chewy, Inc."


def test_append_symbol_visible_after_cache_clear(universe_csv):
    # load_universe is lru_cached; populate the cache, then append, then re-read.
    assert "NVDA" not in set(universe.load_universe()["symbol"])
    universe.append_symbol("NVDA", "NVIDIA Corp.", "Information Technology")
    # append_symbol clears the cache itself, so the new row is visible immediately.
    assert "NVDA" in set(universe.load_universe()["symbol"])


def test_append_symbol_respects_path_arg(tmp_path):
    # Explicit path= argument is honored even with UNIVERSE_PATH pointing elsewhere.
    path = tmp_path / "alt.csv"
    pd.DataFrame(_SEED_ROWS, columns=["symbol", "name", "sector"]).to_csv(path, index=False)
    universe.load_universe.cache_clear()
    try:
        assert universe.append_symbol("PLTR", "Palantir", "Information Technology", path=path) is True
        df = pd.read_csv(path, dtype=str)
        assert "PLTR" in set(df["symbol"])
    finally:
        universe.load_universe.cache_clear()


# --- universe_slice ------------------------------------------------------
def _seed_df():
    return pd.DataFrame(_SEED_ROWS, columns=["symbol", "name", "sector"])


def test_universe_slice_head_only_no_include():
    df = _seed_df()
    out = universe.universe_slice(df, 2)
    assert list(out["symbol"]) == ["MMM", "AAPL"]


def test_universe_slice_unions_out_of_slice_symbol():
    df = _seed_df()
    # ZTS ranks 5th — outside head(2) — so it gets unioned onto the slice.
    out = universe.universe_slice(df, 2, "ZTS")
    assert list(out["symbol"]) == ["MMM", "AAPL", "ZTS"]


def test_universe_slice_noop_when_in_slice():
    df = _seed_df()
    # AAPL is already in head(2); no duplicate row is added.
    out = universe.universe_slice(df, 2, "AAPL")
    assert list(out["symbol"]) == ["MMM", "AAPL"]


def test_universe_slice_noop_when_unknown():
    df = _seed_df()
    out = universe.universe_slice(df, 2, "ZZZZ")
    assert list(out["symbol"]) == ["MMM", "AAPL"]


def test_universe_slice_noop_when_empty():
    df = _seed_df()
    out = universe.universe_slice(df, 2, "")
    assert list(out["symbol"]) == ["MMM", "AAPL"]


def test_universe_slice_case_insensitive_include():
    df = _seed_df()
    # Lower-case + padded include_symbol still matches and unions the row.
    out = universe.universe_slice(df, 2, "  zts ")
    assert list(out["symbol"]) == ["MMM", "AAPL", "ZTS"]


# --- ensure_symbol (fake provider; no network) ---------------------------
def test_ensure_symbol_appends_with_gics_mapped_sector(universe_csv, monkeypatch):
    # yfinance sector "Technology" must map to the CSV's "Information Technology".
    from screener.provider import Fundamentals

    snap = Fundamentals(symbol="NVDA", name="NVIDIA Corp.", sector="Technology")
    stub = _make_stub(snap)
    monkeypatch.setattr("screener.provider.YFinanceProvider", stub)

    assert universe.ensure_symbol("NVDA") is True
    assert stub.calls == ["NVDA"]
    df = pd.read_csv(universe_csv, dtype=str)
    row = df[df["symbol"] == "NVDA"].iloc[0]
    assert row["name"] == "NVIDIA Corp."
    assert row["sector"] == "Information Technology"  # GICS-mapped, not "Technology"


def test_ensure_symbol_unmapped_sector_becomes_blank(universe_csv, monkeypatch):
    from screener.provider import Fundamentals

    # A sector with no GICS mapping (and None) -> stored as "".
    snap = Fundamentals(symbol="NVDA", name="NVIDIA Corp.", sector="Made Up Sector")
    monkeypatch.setattr("screener.provider.YFinanceProvider", _make_stub(snap))

    assert universe.ensure_symbol("NVDA") is True
    df = pd.read_csv(universe_csv, dtype=str).fillna("")
    assert df[df["symbol"] == "NVDA"].iloc[0]["sector"] == ""


def test_ensure_symbol_empty_snapshot_no_append(universe_csv, monkeypatch):
    from screener.provider import Fundamentals

    # is_empty snapshot (only symbol) -> refuse, file untouched.
    snap = Fundamentals(symbol="NVDA")
    assert snap.is_empty
    monkeypatch.setattr("screener.provider.YFinanceProvider", _make_stub(snap))

    before = pd.read_csv(universe_csv, dtype=str)
    assert universe.ensure_symbol("NVDA") is False
    after = pd.read_csv(universe_csv, dtype=str)
    assert after.equals(before)


def test_ensure_symbol_no_name_no_append(universe_csv, monkeypatch):
    from screener.provider import Fundamentals

    # Has a sector but NO name -> still refused (don't poison with a nameless row).
    snap = Fundamentals(symbol="NVDA", sector="Technology")
    monkeypatch.setattr("screener.provider.YFinanceProvider", _make_stub(snap))

    assert universe.ensure_symbol("NVDA") is False
    assert "NVDA" not in set(pd.read_csv(universe_csv, dtype=str)["symbol"])


def test_ensure_symbol_already_present_skips_provider(universe_csv, monkeypatch):
    # AAPL is seeded; ensure_symbol must return False WITHOUT ever calling the provider.
    stub = _make_stub(object())  # if this is consulted, the snapshot would explode later
    monkeypatch.setattr("screener.provider.YFinanceProvider", stub)

    assert universe.ensure_symbol("aapl") is False  # case-folded, still matches
    assert stub.calls == []  # provider NEVER consulted for an in-universe symbol


def test_ensure_symbol_non_ticker_skips_provider(universe_csv, monkeypatch):
    stub = _make_stub(object())
    monkeypatch.setattr("screener.provider.YFinanceProvider", stub)

    # A prose token / sentence is not a valid ticker -> False, no provider call.
    assert universe.ensure_symbol("show me cheap tech") is False
    assert universe.ensure_symbol("") is False
    assert universe.ensure_symbol("TOOLONG") is False  # >5 letters
    assert stub.calls == []


def test_ensure_symbol_allows_class_share_suffix(universe_csv, monkeypatch):
    from screener.provider import Fundamentals

    # BRK.B-style class-share tickers (dot/dash suffix) are valid shapes.
    snap = Fundamentals(symbol="BF.B", name="Brown-Forman", sector="Consumer Defensive")
    monkeypatch.setattr("screener.provider.YFinanceProvider", _make_stub(snap))

    assert universe.ensure_symbol("BF.B") is True
    df = pd.read_csv(universe_csv, dtype=str)
    assert "BF.B" in set(df["symbol"])
    assert df[df["symbol"] == "BF.B"].iloc[0]["sector"] == "Consumer Staples"


# --- import-cheapness invariant ------------------------------------------
def test_agent_module_has_no_heavy_imports():
    """screener.agent's OWN top-level imports pull in neither yfinance nor provider.

    Parses agent.py's source and inspects only MODULE-LEVEL import statements
    (imports nested inside functions — the lazy SDK / provider imports — are
    intentionally ignored). The package ``__init__`` may eagerly import provider,
    but agent.py itself must stay import-cheap so ``from screener import agent``
    is network-free.
    """
    import screener.agent as agent_mod

    src = open(agent_mod.__file__).read()
    tree = ast.parse(src)
    offenders: "list[str]" = []
    for node in tree.body:  # top-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "yfinance" or alias.name.startswith("screener.provider"):
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "yfinance" or mod == "screener.provider" or mod.endswith(".provider"):
                offenders.append(mod)
    assert offenders == [], f"agent.py has heavy top-level imports: {offenders}"
