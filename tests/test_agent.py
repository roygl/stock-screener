"""Synthetic, framework-free, OFFLINE tests for the natural-language agent layer.

Covers :mod:`screener.agent` end to end without a browser or the network:

- :func:`rule_based_parse` — profile synonyms (+ the swing-vs-momentum ordering
  subtlety), "top N"/"scan N" counts (+ clamping via the validate pass), sector
  detection/canonicalization, the score-floor regex + conviction phrases, the
  swing-only earnings gate, and single-unambiguous-ticker extraction.
- :func:`validate_request` — the safety layer: bad/missing/uppercase profile,
  n_names range + type coercion, unknown-sector dropping + canonicalization,
  min_score clamp/NaN, earnings forced off for non-swing, explanation preserved,
  and "never raises" on hostile inputs.
- :func:`parse_query` — the no-key fallback equals the rule-based parser, plus a
  MONKEYPATCHED LLM path (fake ``_llm_extract``) proving the LLM dict ALSO flows
  through ``validate_request`` and that an LLM error falls back cleanly.
- :class:`ScreenRequest` shape (frozen + hashable tuple sectors) and the
  VALID_PROFILES cross-check.

Like ``tests/test_display.py`` / ``tests/test_engine.py``: NO ``pytest``, NO
``yfinance``, NO ``streamlit``, NO ``anthropic`` import — every test is a plain
``test_*`` function using ``assert`` so the suite runs standalone as
``python tests/test_agent.py`` (the ``__main__`` runner counts pass/fail, prints
a summary, and exits non-zero on any failure).
"""

import dataclasses
import os
import sys

# Put the repo root on sys.path so "import screener" resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screener import agent  # noqa: E402


# --- fixtures ------------------------------------------------------------
# A fixed universe-sectors list reused across tests. Uses the production GICS
# spelling ("Information Technology", "Health Care") so the canonicalization
# tests exercise the real-data shape.
SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]


# =========================================================================
# rule_based_parse — profiles
# =========================================================================
def test_rule_based_profiles():
    assert agent.rule_based_parse("cheap value names to buy and hold", SECTORS).profile == "long_term"
    assert agent.rule_based_parse("show me swing breakout setups", SECTORS).profile == "swing"
    assert agent.rule_based_parse("high growth momentum leaders", SECTORS).profile == "momentum"
    # Default when nothing matches.
    assert agent.rule_based_parse("show me stocks", SECTORS).profile == "momentum"


def test_rule_based_profile_ordering_subtlety():
    # "momentum trade" embeds the bare word "momentum"; swing phrases win.
    assert agent.rule_based_parse("momentum trade pullback", SECTORS).profile == "swing"


# =========================================================================
# rule_based_parse — n_names (+ clamp via the validate pass)
# =========================================================================
def test_rule_based_n_names():
    assert agent.rule_based_parse("top 20 momentum names", SECTORS).n_names == 20
    assert agent.rule_based_parse("scan 50 names", SECTORS).n_names == 50
    assert agent.rule_based_parse("first 8 tickers", SECTORS).n_names == 8
    # No count -> default 25.
    assert agent.rule_based_parse("momentum names", SECTORS).n_names == 25


def test_rule_based_n_names_clamped():
    # The parser routes through validate_request, so out-of-range counts clamp.
    assert agent.rule_based_parse("top 1000 names", SECTORS).n_names == 503
    assert agent.rule_based_parse("top 2 names", SECTORS).n_names == 5


# =========================================================================
# rule_based_parse — sectors
# =========================================================================
def test_rule_based_sectors():
    req = agent.rule_based_parse("energy and technology momentum", SECTORS)
    assert "Energy" in req.sectors
    assert "Information Technology" in req.sectors  # the tech sector present in SECTORS
    # No sector named -> empty.
    assert agent.rule_based_parse("momentum names", SECTORS).sectors == ()
    # An unknown alias is dropped.
    assert agent.rule_based_parse("crypto sector momentum", SECTORS).sectors == ()


# =========================================================================
# rule_based_parse — min_score
# =========================================================================
def test_rule_based_min_score():
    assert agent.rule_based_parse("momentum, score above 0.6", SECTORS).min_score == 0.6
    # A percentage is normalized to a fraction.
    assert agent.rule_based_parse("momentum, score above 80", SECTORS).min_score == 0.8
    # A conviction phrase implies 0.7.
    assert agent.rule_based_parse("high conviction momentum", SECTORS).min_score == 0.7
    # None -> 0.0.
    assert agent.rule_based_parse("momentum names", SECTORS).min_score == 0.0


# =========================================================================
# rule_based_parse — earnings gating
# =========================================================================
def test_rule_based_earnings_gating():
    swing = agent.rule_based_parse("swing names with earnings this week", SECTORS)
    assert swing.profile == "swing"
    assert swing.earnings_only is True
    # "earnings" present but profile is momentum -> forced off.
    mom = agent.rule_based_parse("momentum names with earnings", SECTORS)
    assert mom.profile == "momentum"
    assert mom.earnings_only is False


# =========================================================================
# rule_based_parse — ticker text
# =========================================================================
def test_rule_based_ticker_text():
    assert agent.rule_based_parse("how does NVDA look", SECTORS).text == "NVDA"
    # No single unambiguous ticker (stopwords + lowercase words excluded).
    assert agent.rule_based_parse("tech and energy stocks", SECTORS).text == ""
    # Two survivors -> ambiguous -> empty.
    assert agent.rule_based_parse("AAPL vs MSFT", SECTORS).text == ""


# =========================================================================
# validate_request — profile
# =========================================================================
def test_validate_profile():
    assert agent.validate_request({"profile": "banana"}, SECTORS).profile == "momentum"
    assert agent.validate_request({}, SECTORS).profile == "momentum"
    # Case-insensitive.
    assert agent.validate_request({"profile": "SWING"}, SECTORS).profile == "swing"


# =========================================================================
# validate_request — n_names range + coercion
# =========================================================================
def test_validate_n_names_range():
    assert agent.validate_request({"n_names": 1}, SECTORS).n_names == 5
    assert agent.validate_request({"n_names": 99999}, SECTORS).n_names == 503
    assert agent.validate_request({"n_names": "30"}, SECTORS).n_names == 30  # string coerced
    assert agent.validate_request({"n_names": None}, SECTORS).n_names == 25
    assert agent.validate_request({}, SECTORS).n_names == 25
    assert agent.validate_request({"n_names": "abc"}, SECTORS).n_names == 25  # non-numeric -> default


# =========================================================================
# validate_request — sectors dropped + canonicalized
# =========================================================================
def test_validate_sectors():
    req = agent.validate_request({"sectors": ["Energy", "Atlantis", "health care"]}, SECTORS)
    assert req.sectors == ("Energy", "Health Care")  # Atlantis dropped, "health care" canonicalized
    # A single string is accepted.
    assert agent.validate_request({"sectors": "Energy"}, SECTORS).sectors == ("Energy",)
    # None -> empty.
    assert agent.validate_request({"sectors": None}, SECTORS).sectors == ()


# =========================================================================
# validate_request — min_score clamp
# =========================================================================
def test_validate_min_score():
    assert agent.validate_request({"min_score": 1.5}, SECTORS).min_score == 1.0
    assert agent.validate_request({"min_score": -0.2}, SECTORS).min_score == 0.0
    assert agent.validate_request({"min_score": float("nan")}, SECTORS).min_score == 0.0
    assert agent.validate_request({"min_score": "x"}, SECTORS).min_score == 0.0
    assert agent.validate_request({"min_score": 0.42}, SECTORS).min_score == 0.42


# =========================================================================
# validate_request — earnings forced off for non-swing
# =========================================================================
def test_validate_earnings_swing_gate():
    assert agent.validate_request({"profile": "momentum", "earnings_only": True}, SECTORS).earnings_only is False
    assert agent.validate_request({"profile": "swing", "earnings_only": True}, SECTORS).earnings_only is True
    assert agent.validate_request({"profile": "long_term", "earnings_only": True}, SECTORS).earnings_only is False


# =========================================================================
# validate_request — explanation preserved
# =========================================================================
def test_validate_explanation_preserved():
    assert agent.validate_request({"explanation": "  hi  "}, SECTORS).explanation == "hi"  # trimmed, not clobbered
    assert agent.validate_request({}, SECTORS).explanation == ""


# =========================================================================
# validate_request — never raises
# =========================================================================
def test_validate_never_raises():
    # Empty dict + None universe.
    req = agent.validate_request({}, None)
    assert req.profile == "momentum" and req.n_names == 25 and req.min_score == 0.0 and req.sectors == ()
    # Hostile types must not raise and must fall back to safe defaults.
    hostile = agent.validate_request(
        {"sectors": 123, "n_names": object(), "min_score": []}, SECTORS
    )
    assert hostile.profile == "momentum"
    assert hostile.n_names == 25
    assert hostile.min_score == 0.0
    assert hostile.sectors == ()


# =========================================================================
# agent_available + parse_query fallback WITHOUT a key
# =========================================================================
def test_parse_query_fallback_without_key():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        assert agent.agent_available() is False
        got = agent.parse_query("top 15 swing names", universe_sectors=SECTORS)
        expected = agent.rule_based_parse("top 15 swing names", SECTORS)
        # Same result as the rule-based parser, field for field.
        assert got == expected
        assert got.profile == "swing"
        assert got.n_names == 15
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


# =========================================================================
# parse_query LLM path via MONKEYPATCH (no network, no anthropic needed)
# =========================================================================
def test_parse_query_llm_path_monkeypatched():
    saved_available = agent.agent_available
    saved_extract = agent._llm_extract
    try:
        agent.agent_available = lambda: True
        agent._llm_extract = lambda query, universe_sectors, model: {
            "profile": "long_term",
            "n_names": 12,
            "min_score": 0.9,
            "sectors": ["Energy", "Bogus"],
            "text": "",
            "earnings_only": True,
            "explanation": "value names",
        }
        req = agent.parse_query("whatever", universe_sectors=SECTORS)
        assert req.profile == "long_term"
        assert req.n_names == 12  # in range, unchanged
        assert req.min_score == 0.9
        assert req.sectors == ("Energy",)  # Bogus dropped by validate_request
        assert req.earnings_only is False  # forced off (profile != swing)
        assert req.explanation.startswith("LLM")  # parse_query prefixes the LLM note
    finally:
        agent.agent_available = saved_available
        agent._llm_extract = saved_extract


def test_parse_query_llm_error_falls_back():
    saved_available = agent.agent_available
    saved_extract = agent._llm_extract

    def _boom(query, universe_sectors, model):
        raise RuntimeError("boom")

    try:
        agent.agent_available = lambda: True
        agent._llm_extract = _boom
        req = agent.parse_query("top 7 momentum names", universe_sectors=SECTORS)
        # Falls back to the rule-based parser; never raises.
        assert req.profile == "momentum"
        assert req.n_names == 7
    finally:
        agent.agent_available = saved_available
        agent._llm_extract = saved_extract


# =========================================================================
# cross-checks + dataclass shape
# =========================================================================
def test_valid_profiles_in_sync():
    assert set(agent.VALID_PROFILES) == {"long_term", "swing", "momentum"}
    # Optional, import-guarded cross-check against the real registry.
    try:
        from screener.profiles import PROFILES
    except Exception:  # pragma: no cover - profiles always present in this repo
        return
    assert set(agent.VALID_PROFILES) == set(PROFILES)


def test_screen_request_is_frozen_and_hashable():
    req = agent.ScreenRequest()
    # Frozen: attribute assignment raises.
    raised = False
    try:
        req.profile = "swing"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised
    # sectors is a tuple -> the instance is hashable.
    assert isinstance(req.sectors, tuple)
    assert isinstance(hash(req), int)


# --- standalone runner ---------------------------------------------------
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
