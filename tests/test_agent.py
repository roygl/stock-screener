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
import importlib.util
import os
import sys
import types

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
        # Identical to the rule-based parser on every field EXCEPT the explanation,
        # which now carries a visible fallback note (Stage 4 transparency).
        assert dataclasses.replace(got, explanation="") == dataclasses.replace(expected, explanation="")
        assert got.profile == "swing"
        assert got.n_names == 15
        # The explanation says it fell back and WHY (no key), folding the reason
        # into the rule parser's own "Rule-based: <summary>" prefix.
        assert got.explanation.startswith("Rule-based (")
        assert "unavailable" in got.explanation
        summary = expected.explanation.split(":", 1)[1].strip()  # "profile=swing, top 15"
        assert summary in got.explanation
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
        # agent_available now takes an optional provider arg; _llm_extract is now
        # (query, universe_sectors, provider, model). Patch both variadically so
        # this test stays signature-agnostic.
        agent.agent_available = lambda *a, **k: True
        agent._llm_extract = lambda *a, **k: {
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

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    try:
        agent.agent_available = lambda *a, **k: True
        agent._llm_extract = _boom
        req = agent.parse_query("top 7 momentum names", universe_sectors=SECTORS)
        # Falls back to the rule-based parser; never raises.
        assert req.profile == "momentum"
        assert req.n_names == 7
    finally:
        agent.agent_available = saved_available
        agent._llm_extract = saved_extract


# =========================================================================
# provider registry — env snapshot helper
# =========================================================================
# Every env var the agent's resolution helpers read, so a test can save the full
# set, mutate freely, and restore byte-for-byte in a finally (mirrors the
# pop/restore the existing fallback test does for ANTHROPIC_API_KEY).
_AGENT_ENV_KEYS = (
    "SCREENER_AGENT_PROVIDER",
    "SCREENER_AGENT_MODEL",
    "SCREENER_OLLAMA_BASE_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",  # Gemini key alias (Stage 4)
    "MISTRAL_API_KEY",
    # the per-provider model overrides
    "SCREENER_ANTHROPIC_MODEL",
    "SCREENER_OPENAI_MODEL",
    "SCREENER_XAI_MODEL",
    "SCREENER_GEMINI_MODEL",
    "SCREENER_MISTRAL_MODEL",
    "SCREENER_OLLAMA_MODEL",
)


def _snapshot_env() -> dict:
    """Snapshot the agent-relevant env vars (value or None if unset)."""
    return {k: os.environ.get(k) for k in _AGENT_ENV_KEYS}


def _restore_env(saved: dict) -> None:
    """Restore env vars to a prior :func:`_snapshot_env` exactly (set or unset)."""
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _clear_agent_env() -> None:
    """Unset every agent env var so a test starts from a known-empty slate."""
    for k in _AGENT_ENV_KEYS:
        os.environ.pop(k, None)


# The 6 backend ids, pinned. This is the stability guard the plan calls for: a new
# id must be added here deliberately (mirrors the VALID_PROFILES cross-check).
_EXPECTED_PROVIDER_IDS = ("anthropic", "openai", "xai", "gemini", "mistral", "ollama")


# =========================================================================
# provider registry — shape + stability guard
# =========================================================================
def test_provider_registry_shape():
    # Exactly the expected ids, and the dict key matches each Provider.id.
    assert set(agent.PROVIDERS) == set(_EXPECTED_PROVIDER_IDS)
    assert len(agent.PROVIDERS) == len(_EXPECTED_PROVIDER_IDS)
    for pid, p in agent.PROVIDERS.items():
        assert isinstance(p, agent.Provider)
        assert p.id == pid  # dict key is the provider's own id
        # Non-empty id/label/kind/default_model (env_key may be "" for keyless).
        assert isinstance(p.id, str) and p.id
        assert isinstance(p.label, str) and p.label
        assert isinstance(p.kind, str) and p.kind
        assert isinstance(p.default_model, str) and p.default_model
        assert isinstance(p.model_env, str) and p.model_env
        # kind drives the SDK/call path — only two are known.
        assert p.kind in {"anthropic", "openai"}
        # base_url is the SDK default (None) or a string endpoint.
        assert p.base_url is None or isinstance(p.base_url, str)
    # ids are unique (dict keys guarantee this, but assert the count explicitly).
    assert len({p.id for p in agent.PROVIDERS.values()}) == len(agent.PROVIDERS)


def test_provider_registry_ids_pinned():
    # Stability guard: pin the exact roster so adding/removing a backend is a
    # deliberate edit that updates this test too.
    assert tuple(sorted(agent.PROVIDERS)) == tuple(sorted(_EXPECTED_PROVIDER_IDS))
    # Each expected id is present individually (clear failure message if one drops).
    for pid in _EXPECTED_PROVIDER_IDS:
        assert pid in agent.PROVIDERS


def test_provider_registry_kinds():
    # anthropic is the only native path; everyone else rides the openai SDK.
    assert agent.PROVIDERS["anthropic"].kind == "anthropic"
    for pid in ("openai", "xai", "gemini", "mistral", "ollama"):
        assert agent.PROVIDERS[pid].kind == "openai"
    # Ollama is the keyless one; all others carry a key env var.
    assert agent.PROVIDERS["ollama"].env_key == ""
    for pid in ("anthropic", "openai", "xai", "gemini", "mistral"):
        assert agent.PROVIDERS[pid].env_key != ""


def test_default_provider():
    assert agent.DEFAULT_PROVIDER == "gemini"
    assert agent.DEFAULT_PROVIDER in agent.PROVIDERS


# =========================================================================
# _resolve_provider — explicit > env > default; unknown -> default
# =========================================================================
def test_resolve_provider_precedence():
    saved = _snapshot_env()
    try:
        _clear_agent_env()
        # No arg, no env -> DEFAULT_PROVIDER.
        assert agent._resolve_provider(None) is agent.PROVIDERS[agent.DEFAULT_PROVIDER]
        # Explicit arg wins.
        assert agent._resolve_provider("openai") is agent.PROVIDERS["openai"]
        # Env var used when no explicit arg.
        os.environ["SCREENER_AGENT_PROVIDER"] = "mistral"
        assert agent._resolve_provider(None) is agent.PROVIDERS["mistral"]
        # Explicit arg beats the env var.
        assert agent._resolve_provider("xai") is agent.PROVIDERS["xai"]
    finally:
        _restore_env(saved)


def test_resolve_provider_unknown_falls_back():
    saved = _snapshot_env()
    try:
        _clear_agent_env()
        # Unknown explicit id -> default (never raises, never KeyError).
        assert agent._resolve_provider("does-not-exist") is agent.PROVIDERS[agent.DEFAULT_PROVIDER]
        # Unknown env id -> default too.
        os.environ["SCREENER_AGENT_PROVIDER"] = "garbage"
        assert agent._resolve_provider(None) is agent.PROVIDERS[agent.DEFAULT_PROVIDER]
        # Empty-string arg is falsy -> resolves via env/default, not a KeyError.
        _clear_agent_env()
        assert agent._resolve_provider("") is agent.PROVIDERS[agent.DEFAULT_PROVIDER]
    finally:
        _restore_env(saved)


# =========================================================================
# _resolve_model — explicit > per-provider env > global env > default
# =========================================================================
def test_resolve_model_precedence():
    saved = _snapshot_env()
    try:
        _clear_agent_env()
        p = agent.PROVIDERS["openai"]
        # Nothing set -> the registry default.
        assert agent._resolve_model(p, None) == p.default_model
        # Global env beats the default.
        os.environ["SCREENER_AGENT_MODEL"] = "global-model"
        assert agent._resolve_model(p, None) == "global-model"
        # Per-provider model_env beats the global env.
        os.environ[p.model_env] = "per-provider-model"
        assert agent._resolve_model(p, None) == "per-provider-model"
        # An explicit model arg beats everything.
        assert agent._resolve_model(p, "explicit-model") == "explicit-model"
    finally:
        _restore_env(saved)


def test_resolve_model_per_provider_isolation():
    # The OpenAI model_env must not leak into another provider's resolution.
    saved = _snapshot_env()
    try:
        _clear_agent_env()
        os.environ["SCREENER_OPENAI_MODEL"] = "openai-only"
        xai = agent.PROVIDERS["xai"]
        # xai reads SCREENER_XAI_MODEL, not the OpenAI one -> its default stands.
        assert agent._resolve_model(xai, None) == xai.default_model
        # And the OpenAI provider does see it.
        assert agent._resolve_model(agent.PROVIDERS["openai"], None) == "openai-only"
    finally:
        _restore_env(saved)


# =========================================================================
# agent_available — key presence + SDK probe (monkeypatched find_spec)
# =========================================================================
def _patch_find_spec(present: bool):
    """Return a fake importlib.util.find_spec: truthy module if present else None."""
    def _fake(name, *args, **kwargs):
        return object() if present else None

    return _fake


def test_agent_available_no_key_is_false():
    saved = _snapshot_env()
    saved_find = importlib.util.find_spec
    try:
        _clear_agent_env()
        # Even with the SDK "present", a keyed provider with no key is unavailable.
        importlib.util.find_spec = _patch_find_spec(True)
        assert agent.agent_available("openai") is False
        assert agent.agent_available("anthropic") is False
    finally:
        importlib.util.find_spec = saved_find
        _restore_env(saved)


def test_agent_available_sdk_absent_is_false():
    saved = _snapshot_env()
    saved_find = importlib.util.find_spec
    try:
        _clear_agent_env()
        # Key present but the SDK is absent (find_spec -> None) -> unavailable.
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        importlib.util.find_spec = _patch_find_spec(False)
        assert agent.agent_available("openai") is False
        assert agent.agent_available("anthropic") is False
    finally:
        importlib.util.find_spec = saved_find
        _restore_env(saved)


def test_agent_available_true_when_key_and_sdk():
    saved = _snapshot_env()
    saved_find = importlib.util.find_spec
    try:
        _clear_agent_env()
        importlib.util.find_spec = _patch_find_spec(True)
        os.environ["OPENAI_API_KEY"] = "sk-test"
        assert agent.agent_available("openai") is True
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        os.environ["GEMINI_API_KEY"] = "sk-test"
        # No-arg call resolves to gemini (the default) -> True with the key + SDK present.
        assert agent.agent_available() is True
        assert agent.agent_available("anthropic") is True
    finally:
        importlib.util.find_spec = saved_find
        _restore_env(saved)


def test_agent_available_ollama_keyless():
    saved = _snapshot_env()
    saved_find = importlib.util.find_spec
    try:
        _clear_agent_env()
        # Ollama needs no key: available iff the openai SDK is importable, regardless
        # of any key being set.
        importlib.util.find_spec = _patch_find_spec(True)
        assert agent.agent_available("ollama") is True  # no key set, still available
        importlib.util.find_spec = _patch_find_spec(False)
        assert agent.agent_available("ollama") is False  # SDK gone -> unavailable
    finally:
        importlib.util.find_spec = saved_find
        _restore_env(saved)


def test_agent_available_never_raises():
    saved = _snapshot_env()
    saved_find = importlib.util.find_spec
    try:
        _clear_agent_env()

        # A find_spec that explodes must be swallowed -> False, not a raise.
        def _boom(name, *args, **kwargs):
            raise RuntimeError("boom")

        importlib.util.find_spec = _boom
        os.environ["OPENAI_API_KEY"] = "sk-test"
        assert agent.agent_available("openai") is False
        # An unknown provider id resolves to the default and still never raises.
        assert agent.agent_available("totally-unknown") in (True, False)
    finally:
        importlib.util.find_spec = saved_find
        _restore_env(saved)


# =========================================================================
# Stage 4 — GOOGLE_API_KEY alias + availability_status + visible fallback reason
# =========================================================================
def test_gemini_google_api_key_alias():
    saved = _snapshot_env()
    saved_find = importlib.util.find_spec
    try:
        _clear_agent_env()
        importlib.util.find_spec = _patch_find_spec(True)
        # No key at all -> unavailable.
        assert agent.agent_available("gemini") is False
        # The official GOOGLE_API_KEY alias counts as the Gemini key.
        os.environ["GOOGLE_API_KEY"] = "g-test"
        assert agent.agent_available("gemini") is True
        assert agent._provider_api_key(agent.PROVIDERS["gemini"]) == "g-test"
        # The primary GEMINI_API_KEY takes precedence when both are set.
        os.environ["GEMINI_API_KEY"] = "primary"
        assert agent._provider_api_key(agent.PROVIDERS["gemini"]) == "primary"
        # The alias is Gemini-only: it does NOT satisfy another provider's key.
        assert agent._provider_api_key(agent.PROVIDERS["openai"]) is None
    finally:
        importlib.util.find_spec = saved_find
        _restore_env(saved)


def test_availability_status_reasons():
    saved = _snapshot_env()
    saved_find = importlib.util.find_spec
    try:
        _clear_agent_env()
        importlib.util.find_spec = _patch_find_spec(True)
        # Missing key -> (False, reason naming the env var(s)).
        ok, reason = agent.availability_status("openai")
        assert ok is False and "OPENAI_API_KEY" in reason
        # Gemini's reason names BOTH the primary key and the GOOGLE_API_KEY alias.
        ok, reason = agent.availability_status("gemini")
        assert ok is False
        assert "GEMINI_API_KEY" in reason and "GOOGLE_API_KEY" in reason
        # Key present + SDK present -> ready.
        os.environ["OPENAI_API_KEY"] = "sk-test"
        assert agent.availability_status("openai") == (True, "ready")
        # Key present but SDK absent -> a "not importable" reason, not "no key".
        importlib.util.find_spec = _patch_find_spec(False)
        ok, reason = agent.availability_status("openai")
        assert ok is False and "openai" in reason and "importable" in reason
    finally:
        importlib.util.find_spec = saved_find
        _restore_env(saved)


def test_availability_status_ollama_keyless_ready():
    saved = _snapshot_env()
    saved_find = importlib.util.find_spec
    try:
        _clear_agent_env()
        importlib.util.find_spec = _patch_find_spec(True)
        # Keyless Ollama: ready on SDK alone (no key reason ever).
        assert agent.availability_status("ollama") == (True, "ready")
    finally:
        importlib.util.find_spec = saved_find
        _restore_env(saved)


def test_parse_query_llm_error_note_in_explanation():
    # The exception fallback path annotates the explanation with the error reason.
    saved_available = agent.agent_available
    saved_extract = agent._llm_extract

    def _boom(*args, **kwargs):
        raise RuntimeError("kaboom-detail")

    try:
        agent.agent_available = lambda *a, **k: True
        agent._llm_extract = _boom
        req = agent.parse_query("top 7 momentum names", universe_sectors=SECTORS)
        assert req.profile == "momentum" and req.n_names == 7
        assert req.explanation.startswith("Rule-based (")
        assert "error" in req.explanation
        assert "RuntimeError" in req.explanation  # the exception type is surfaced
    finally:
        agent.agent_available = saved_available
        agent._llm_extract = saved_extract


# =========================================================================
# _set_screen_schema — strict-safe object schema
# =========================================================================
def test_set_screen_schema_sectors_enum():
    schema = agent._set_screen_schema(list(SECTORS))
    # The sectors enum reflects exactly the passed list.
    assert schema["properties"]["sectors"]["items"]["enum"] == list(SECTORS)


def test_set_screen_schema_strict_shape():
    schema = agent._set_screen_schema(list(SECTORS))
    assert schema["type"] == "object"
    # Strict mode requires additionalProperties False and ALL props required.
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())
    assert set(schema["required"]) == {
        "profile", "n_names", "min_score", "sectors",
        "text", "earnings_only", "explanation",
    }


def test_set_screen_schema_no_numeric_bounds():
    schema = agent._set_screen_schema(list(SECTORS))
    # Strict function-calling forbids numeric min/max; clamping lives in
    # validate_request, so the numeric props must NOT carry minimum/maximum.
    for prop in ("n_names", "min_score"):
        assert "minimum" not in schema["properties"][prop]
        assert "maximum" not in schema["properties"][prop]


def test_set_screen_schema_empty_sectors():
    # With no universe sectors, the array still validates (no empty/illegal enum).
    schema = agent._set_screen_schema([])
    items = schema["properties"]["sectors"]["items"]
    assert items["type"] == "string"
    assert "enum" not in items  # no enum rather than an empty (invalid) one


# =========================================================================
# _parse_openai_toolcall — JSON-string arguments off a fake response
# =========================================================================
def _fake_openai_response(arguments):
    """Build a SimpleNamespace mimicking resp.choices[0].message.tool_calls[0]...."""
    function = types.SimpleNamespace(name="set_screen", arguments=arguments)
    tool_call = types.SimpleNamespace(function=function)
    message = types.SimpleNamespace(tool_calls=[tool_call])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def test_parse_openai_toolcall_ok():
    resp = _fake_openai_response(
        '{"profile": "swing", "n_names": 7, "sectors": ["Energy"]}'
    )
    out = agent._parse_openai_toolcall(resp)
    assert out == {"profile": "swing", "n_names": 7, "sectors": ["Energy"]}


def test_parse_openai_toolcall_empty_raises():
    # Empty tool_calls -> ValueError so parse_query falls back.
    message = types.SimpleNamespace(tool_calls=[])
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
    raised = False
    try:
        agent._parse_openai_toolcall(resp)
    except ValueError:
        raised = True
    assert raised


def test_parse_openai_toolcall_missing_raises():
    # tool_calls = None (model returned plain text) -> raises (caught upstream).
    message = types.SimpleNamespace(tool_calls=None)
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
    raised = False
    try:
        agent._parse_openai_toolcall(resp)
    except Exception:  # ValueError for None, but any raise satisfies the fallback contract
        raised = True
    assert raised


# =========================================================================
# parse_query — fallback for EVERY provider with no key/SDK present
# =========================================================================
def test_parse_query_fallback_all_providers():
    saved = _snapshot_env()
    try:
        _clear_agent_env()  # no keys, and no SCREENER_AGENT_PROVIDER override
        # Neither SDK is installed in the test env, so every provider degrades to
        # the offline rule parser, deterministically and without raising.
        for pid in _EXPECTED_PROVIDER_IDS:
            got = agent.parse_query(
                "top 12 swing names", universe_sectors=SECTORS, provider=pid
            )
            expected = agent.rule_based_parse("top 12 swing names", SECTORS)
            # Identical to the rule parser EXCEPT the explanation, which now carries
            # the visible fallback reason (no key -> "unavailable"; a keyless Ollama
            # with no local server -> "error" from the refused connection).
            assert dataclasses.replace(got, explanation="") == dataclasses.replace(expected, explanation=""), pid
            assert got.profile == "swing"
            assert got.n_names == 12
            # The offline path is labelled rule-based, never LLM, and states WHY.
            assert not got.explanation.startswith("LLM"), pid
            assert got.explanation.startswith("Rule-based"), pid
            assert ("unavailable" in got.explanation) or ("error" in got.explanation), pid
    finally:
        _restore_env(saved)


def test_parse_query_unknown_provider_falls_back():
    # An unknown provider id resolves to the default and still degrades cleanly
    # (no key/SDK) — never raises.
    saved = _snapshot_env()
    try:
        _clear_agent_env()
        got = agent.parse_query(
            "top 6 momentum names", universe_sectors=SECTORS, provider="not-a-provider"
        )
        assert got.profile == "momentum"
        assert got.n_names == 6
        assert not got.explanation.startswith("LLM")
    finally:
        _restore_env(saved)


# =========================================================================
# parse_query — provider-aware LLM prefix (monkeypatched, no SDK/network)
# =========================================================================
def test_parse_query_llm_prefix_is_provider_aware():
    # When the model returns an explanation, parse_query prefixes it with the
    # resolved provider id: "LLM (<id>): <explanation>".
    saved = _snapshot_env()
    saved_available = agent.agent_available
    saved_extract = agent._llm_extract
    try:
        _clear_agent_env()
        agent.agent_available = lambda *a, **k: True
        agent._llm_extract = lambda *a, **k: {
            "profile": "momentum",
            "n_names": 10,
            "explanation": "ten momentum names",
        }
        req = agent.parse_query("q", universe_sectors=SECTORS, provider="openai")
        assert req.explanation == "LLM (openai): ten momentum names"
        # A different provider id flows through to the prefix.
        req2 = agent.parse_query("q", universe_sectors=SECTORS, provider="xai")
        assert req2.explanation == "LLM (xai): ten momentum names"
    finally:
        agent.agent_available = saved_available
        agent._llm_extract = saved_extract
        _restore_env(saved)


def test_parse_query_llm_synthesized_prefix_embeds_model():
    # When the model OMITS an explanation, parse_query synthesizes one that embeds
    # the resolved provider id AND model: "LLM (<id>, <model>): profile=..., top ...".
    saved = _snapshot_env()
    saved_available = agent.agent_available
    saved_extract = agent._llm_extract
    try:
        _clear_agent_env()
        agent.agent_available = lambda *a, **k: True
        agent._llm_extract = lambda *a, **k: {"profile": "momentum", "n_names": 10}
        req = agent.parse_query("q", universe_sectors=SECTORS, provider="openai")
        model = agent.PROVIDERS["openai"].default_model
        assert req.explanation == f"LLM (openai, {model}): profile=momentum, top 10"
    finally:
        agent.agent_available = saved_available
        agent._llm_extract = saved_extract
        _restore_env(saved)


# =========================================================================
# cross-checks + dataclass shape
# =========================================================================
def test_valid_profiles_in_sync():
    assert set(agent.VALID_PROFILES) == {"long_term", "swing", "momentum", "all"}
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
