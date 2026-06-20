"""Natural-language agent layer — map a plain-English query to screen knobs.

This module sits ON TOP of the deterministic engine; the engine stays the source
of truth. The agent only translates a free-text request into the same knobs the
sidebar already exposes (profile, universe size, sectors, score floor, a symbol
filter, the swing-only earnings toggle) and a one-line ``explanation`` of how it
read the request. It NEVER gives buy/sell/hold advice — both the rule-based path
and the LLM system prompt are constrained to parameter extraction only.

Design constraints (house style + the build spec):

- **Pure and cheap at import time.** This module imports only the stdlib at
  module scope. It NEVER imports streamlit, and it NEVER imports anthropic at
  module top — the SDK is *lazy-imported inside* :func:`_llm_extract`, so
  ``from screener import agent`` succeeds with anthropic absent (it is today) and
  with no API key. :func:`agent_available` probes for the SDK via
  ``importlib.util.find_spec`` without importing it.
- **One safety layer.** :func:`validate_request` coerces and clamps EVERY field
  and never raises; BOTH the rule-based dict and the LLM's tool-call dict are
  routed through it, so clamping (n_names 5..503, min_score [0,1]), sector
  canonicalization against the live universe, the bad-profile fallback, and the
  swing-only earnings gate live in exactly one place.
- **Graceful degradation.** :func:`parse_query` uses the LLM only when a key is
  present AND the SDK imports; on ANY failure (or no key) it falls back to the
  offline :func:`rule_based_parse`. The UI never crashes and never requires the
  optional dependency.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Iterable, Optional

# --- module constants: one source of truth shared by both parse paths ----
# These three profile names are the literal set the screener supports. They are
# hardcoded here (rather than importing screener.profiles.PROFILES) so this
# module stays import-cheap and free of heavy deps. They MUST stay in sync with
# screener.profiles.PROFILES — tests/test_agent.py cross-checks them.
VALID_PROFILES: "tuple[str, ...]" = ("long_term", "swing", "momentum")
DEFAULT_PROFILE = "momentum"

N_NAMES_MIN = 5
N_NAMES_MAX = 503
N_NAMES_DEFAULT = 25

# The LLM model. Overridable via env so ops can pin/upgrade without a code change;
# defaults to opus-4-8 (do NOT downgrade for cost — this is a one-shot extraction).
DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You convert a user's natural-language request into parameters for a US "
    "large-cap stock SCREENER. The screener only describes and ranks stocks by "
    "mechanical signals; it never gives financial advice. Map the request to the "
    "set_screen tool: pick the closest profile (long_term = value/cheap/"
    "buy-and-hold; swing = breakout/short-term/pullback; momentum = growth/"
    "trending leaders); set n_names if a count is requested; set sectors only to "
    "sectors the user names; set min_score in [0,1] if they ask for a quality/"
    "conviction floor (treat a percentage like 80 as 0.80); set earnings_only "
    "only for swing requests that mention earnings; set text to a single ticker "
    "symbol only if the user clearly names one. Write a one-line `explanation` of "
    "how you interpreted the request. NEVER recommend buying, selling, or "
    "holding; NEVER predict prices; only fill in the parameters."
)


@dataclass(frozen=True)
class ScreenRequest:
    """The controllable screen knobs parsed from a natural-language query.

    Frozen (immutable + hashable) so it can be cached / compared safely. ``sectors``
    is a TUPLE (not a list) precisely because the dataclass is frozen — a mutable
    list field would break hashing/equality; callers convert at the boundary with
    ``list(req.sectors)``.

    Defaults reproduce today's neutral dashboard state so an under-specified query
    behaves like the unfiltered app: ``profile`` defaults to ``"momentum"`` (matches
    :data:`DEFAULT_PROFILE`), ``n_names`` to 25 (the slider default), ``min_score``
    0.0 and ``earnings_only`` False (the filter-widget defaults).
    """

    profile: str = "momentum"            # one of "long_term" | "swing" | "momentum"
    n_names: int = 25                    # clamped 5..503 by validate_request
    min_score: float = 0.0               # clamped [0.0, 1.0]
    sectors: "tuple[str, ...]" = ()      # subset of universe sectors; tuple => hashable/frozen
    text: str = ""                       # trimmed free-text symbol / name filter
    earnings_only: bool = False          # forced False unless profile == "swing"
    explanation: str = ""                # one-line human note of how the query was read


# Ordered (keyword -> profile) table. First keyword found in the lowercased query
# wins, so SWING's multi-word phrases are checked BEFORE bare "momentum" (the
# substring "momentum" lives inside "momentum trade"). Then momentum/growth, then
# long-term/value. This ordering is load-bearing — see rule_based_parse step 2.
_PROFILE_KEYWORDS: "tuple[tuple[str, str], ...]" = (
    # swing first (incl. multi-word phrases that embed other profile words)
    ("swing", "swing"),
    ("breakout", "swing"),
    ("break out", "swing"),
    ("pullback", "swing"),
    ("momentum trade", "swing"),
    ("short-term", "swing"),
    ("short term", "swing"),
    # momentum / growth
    ("momentum", "momentum"),
    ("growth", "momentum"),
    ("trending", "momentum"),
    ("high flyer", "momentum"),
    ("high-flyer", "momentum"),
    ("leaders", "momentum"),
    # long-term / value
    ("long-term", "long_term"),
    ("long term", "long_term"),
    ("value", "long_term"),
    ("cheap", "long_term"),
    ("undervalued", "long_term"),
    ("buy and hold", "long_term"),
    ("buy-and-hold", "long_term"),
    ("fundamental", "long_term"),
)

# "top N" / "scan N" / "first N"  OR  "N names/stocks/tickers/companies".
_N_NAMES_RE = re.compile(
    r"(?:top|scan|first)\s+(\d{1,4})|(\d{1,4})\s+(?:names|stocks|tickers|companies)"
)

# "score above/over/>=/greater than/at least X" -> a number (int or decimal).
_MIN_SCORE_RE = re.compile(
    r"score\s*(?:above|over|>=?|greater than|at least)\s*(\d*\.?\d+)"
)

# Conviction phrases that imply a 0.7 score floor when no explicit number is given.
_HIGH_CONVICTION = (
    "high conviction",
    "high-conviction",
    "strong",
    "best",
    "top quality",
)

# Sector aliases -> a substring to look for in the ACTUAL universe sectors. The
# canonical spelling is resolved from universe_sectors at runtime (the S&P CSV may
# use "Information Technology" rather than "Technology"), so we never hardcode the
# stored value here — only what to scan for.
_SECTOR_ALIASES: "tuple[tuple[str, str], ...]" = (
    ("tech", "tech"),
    ("healthcare", "health"),
    ("health", "health"),
    ("energy", "energy"),
    ("financial", "financ"),
    ("finance", "financ"),
    ("banks", "financ"),
    ("industrial", "industrial"),
    ("consumer", "consumer"),
    ("utilities", "utilit"),
    ("materials", "material"),
    ("real estate", "real estate"),
    ("communication", "communication"),
)

# All-caps English words that are NOT tickers (so "tech stocks" can't be hijacked
# by a stray "T", and "PE"/"RSI" etc. are never mistaken for a symbol).
_TICKER_STOPWORDS = frozenset(
    {
        "A", "I", "P", "E", "PE", "RSI", "MACD", "SMA", "EMA", "US", "ETF",
        "AND", "OR", "THE", "TOP", "ALL", "NEW", "BUY", "CEO",
    }
)

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")


def _resolve_model(model: Optional[str]) -> str:
    """Resolve the model id: explicit arg > ``SCREENER_AGENT_MODEL`` env > default."""
    return model or os.environ.get("SCREENER_AGENT_MODEL") or DEFAULT_MODEL


# --- the safety layer: coerce + clamp ANY input, never raise -------------
def validate_request(raw_params: dict, universe_sectors: Iterable[str]) -> ScreenRequest:
    """Coerce/clamp an arbitrary params dict into a valid :class:`ScreenRequest`.

    The SINGLE safety layer applied to BOTH the LLM's tool-call dict and the
    rule-based raw dict, so all clamping/canonicalization lives in one place.
    Tolerates missing keys, wrong types, ``None``, and ``NaN``; it NEVER raises.

    Coercions (each individually guarded):

    - ``profile`` -> ``str.strip().lower()``; if not in :data:`VALID_PROFILES`,
      falls back to :data:`DEFAULT_PROFILE`.
    - ``n_names`` -> ``int`` then clamped to ``N_NAMES_MIN..N_NAMES_MAX`` (5..503).
      Clamping lives HERE (not in the strict tool schema, which forbids numeric
      min/max, nor in the rule parser).
    - ``min_score`` -> ``float`` then clamped to ``[0, 1]``; ``NaN`` -> 0.0.
    - ``sectors`` -> each value looked up case-insensitively against the live
      universe and stored in its CANONICAL spelling; unknown sectors are silently
      DROPPED (the safety contract); deduped preserving first-seen order; tuple.
    - ``text`` -> trimmed string.
    - ``earnings_only`` -> ``bool`` AND forced False unless the resolved profile is
      ``"swing"`` (mirrors display.apply_filters' swing gating).
    - ``explanation`` -> trimmed string, PRESERVED as-is (rule parser / LLM own it).
    """
    raw = raw_params if isinstance(raw_params, dict) else {}

    # Known universe sectors + a case-insensitive lower -> canonical lookup.
    known = {s for s in (universe_sectors or []) if isinstance(s, str)}
    lower_to_canonical = {s.lower(): s for s in known}

    # profile -------------------------------------------------------------
    try:
        profile = str(raw.get("profile")).strip().lower()
    except Exception:  # noqa: BLE001 - any coercion failure falls back
        profile = DEFAULT_PROFILE
    if profile not in VALID_PROFILES:
        profile = DEFAULT_PROFILE

    # n_names (coerce then clamp) ----------------------------------------
    # int(float(...)) so a float-style string ("30.0") coerces too; OverflowError
    # is caught so a non-finite value (inf) falls back rather than raising — this
    # function MUST NOT raise for any input (its whole job is to be the safety net).
    try:
        n_names = int(float(raw.get("n_names")))
    except (TypeError, ValueError, OverflowError):
        n_names = N_NAMES_DEFAULT
    n_names = max(N_NAMES_MIN, min(n_names, N_NAMES_MAX))

    # min_score (coerce, NaN-guard, clamp) -------------------------------
    try:
        min_score = float(raw.get("min_score"))
    except (TypeError, ValueError):
        min_score = 0.0
    if min_score != min_score:  # NaN
        min_score = 0.0
    min_score = max(0.0, min(min_score, 1.0))

    # sectors (canonicalize against the universe, drop unknowns, dedup) ---
    sectors: "list[str]" = []
    try:
        raw_sectors = raw.get("sectors")
        if isinstance(raw_sectors, str):
            candidates: "list" = [raw_sectors]
        elif isinstance(raw_sectors, (list, tuple)):
            candidates = list(raw_sectors)
        else:
            candidates = []
        seen = set()
        for cand in candidates:
            if not isinstance(cand, str):
                continue
            canonical = lower_to_canonical.get(cand.strip().lower())
            if canonical is not None and canonical not in seen:
                sectors.append(canonical)
                seen.add(canonical)
    except Exception:  # noqa: BLE001 - never let a bad sectors value raise
        sectors = []

    # text ----------------------------------------------------------------
    text = str(raw.get("text") or "").strip()

    # earnings_only (force off unless swing) ------------------------------
    earnings_only = bool(raw.get("earnings_only")) and (profile == "swing")

    # explanation (preserve; trim only) ----------------------------------
    explanation = str(raw.get("explanation") or "").strip()

    return ScreenRequest(
        profile=profile,
        n_names=n_names,
        min_score=min_score,
        sectors=tuple(sectors),
        text=text,
        earnings_only=earnings_only,
        explanation=explanation,
    )


# --- the deterministic offline parser ------------------------------------
def _match_profile(q: str) -> str:
    """First profile whose keyword appears in ``q`` (ordered table); else default."""
    for keyword, profile in _PROFILE_KEYWORDS:
        if keyword in q:
            return profile
    return DEFAULT_PROFILE


def _match_n_names(q: str) -> "Optional[int]":
    """First "top N" / "N names" integer in ``q``, or ``None`` (no clamping here)."""
    m = _N_NAMES_RE.search(q)
    if not m:
        return None
    digits = m.group(1) or m.group(2)
    try:
        return int(digits)
    except (TypeError, ValueError):
        return None


def _match_sectors(q: str, universe_sectors: Iterable[str]) -> "list[str]":
    """Sectors named in ``q``, in canonical universe spelling, deduped in order.

    Two passes, both case-insensitive against the ACTUAL universe sectors:
    (1) a direct substring hit on the universe sector's own name; (2) an alias
    (e.g. "tech" -> the sector containing "tech"). validate_request re-validates,
    so this is belt-and-suspenders.
    """
    known = [s for s in (universe_sectors or []) if isinstance(s, str)]
    out: "list[str]" = []
    seen = set()

    def _add(canonical: str) -> None:
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)

    # Pass 1: the sector's own name appears verbatim in the query.
    for sector in known:
        if sector.lower() in q:
            _add(sector)

    # Pass 2: aliases -> the universe sector that contains the alias substring.
    for alias, needle in _SECTOR_ALIASES:
        if alias in q:
            for sector in known:
                if needle in sector.lower():
                    _add(sector)
                    break
    return out


def _match_min_score(q: str) -> "Optional[float]":
    """Score floor from "score above X" (percent>1 normalized) or a conviction phrase."""
    m = _MIN_SCORE_RE.search(q)
    if m:
        try:
            value = float(m.group(1))
        except (TypeError, ValueError):
            value = None
        if value is not None:
            if value > 1:  # a percentage like "80" -> 0.80
                value = value / 100.0
            return value
    if any(phrase in q for phrase in _HIGH_CONVICTION):
        return 0.7
    return None


def _match_ticker(query: str) -> str:
    """A single unambiguous uppercase ticker in the ORIGINAL query, else ``""``.

    Scans the NON-lowercased query for ALL-CAPS tokens, drops stopwords, and
    returns the symbol only if EXACTLY one survives (zero or 2+ -> ambiguous ->
    empty, so "tech stocks" or "AAPL vs MSFT" never set a symbol filter).
    """
    candidates = [
        tok for tok in _TICKER_RE.findall(query or "") if tok not in _TICKER_STOPWORDS
    ]
    if len(candidates) == 1:
        return candidates[0]
    return ""


def _build_explanation(
    profile: str,
    n_names: int,
    sectors: "list[str]",
    min_score: "Optional[float]",
    earnings_only: bool,
    text: str,
) -> str:
    """Concise "Rule-based: ..." note of what was detected (kept under ~140 chars)."""
    parts = [f"profile={profile}", f"top {n_names}"]
    if sectors:
        parts.append(f"sectors=[{', '.join(sectors)}]")
    if min_score is not None:
        parts.append(f"min score {min_score:g}")
    if earnings_only:
        parts.append("earnings-only")
    if text:
        parts.append(f"symbol {text}")
    note = "Rule-based: " + ", ".join(parts)
    return note[:140]


def rule_based_parse(query: str, universe_sectors: Iterable[str]) -> ScreenRequest:
    """Deterministic, offline keyword parse of ``query`` into a :class:`ScreenRequest`.

    Lowercase-substring matching for everything except the ticker token (which
    scans the original-case query). Always returns a valid request: the raw dict
    is routed through :func:`validate_request` at the end, so clamping (n_names,
    min_score), sector canonicalization, and the swing-only earnings gate are
    applied centrally — this parser never clamps on its own.

    The ``explanation`` is synthesized from the VALIDATED (clamped) values so the
    transparency note always matches the knobs the scan actually uses (e.g. a
    "top 1000" request clamps to 503 — the note says 503, not 1000).
    """
    q = (query or "").lower()

    profile = _match_profile(q)
    n_names = _match_n_names(q)
    sectors = _match_sectors(q, universe_sectors)
    min_score = _match_min_score(q)
    # Set unconditionally; validate_request forces it False unless profile=="swing".
    earnings_only = ("earnings" in q) or ("reporting" in q) or ("report" in q)
    text = _match_ticker(query)

    raw: dict = {
        "profile": profile,
        "sectors": sectors,
        "earnings_only": earnings_only,
        "text": text,
    }
    if n_names is not None:
        raw["n_names"] = n_names
    if min_score is not None:
        raw["min_score"] = min_score

    # Validate/clamp FIRST, then build the explanation from the validated request
    # so the transparency caption can never contradict the knobs the engine runs.
    req = validate_request(raw, universe_sectors)
    shown_min = req.min_score if min_score is not None else None
    explanation = _build_explanation(
        req.profile, req.n_names, list(req.sectors), shown_min,
        req.earnings_only, req.text,
    )
    return replace(req, explanation=explanation)


# --- the optional LLM path -----------------------------------------------
def agent_available() -> bool:
    """True iff an API key is set AND the ``anthropic`` SDK is importable.

    Probes for the SDK with ``importlib.util.find_spec`` so it never actually
    imports anthropic at module scope (keeping this module import-cheap and safe
    when the optional dep is absent). Returns False on any error.
    """
    try:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        import importlib.util

        return importlib.util.find_spec("anthropic") is not None
    except Exception:  # noqa: BLE001 - availability probe must never raise
        return False


def _llm_extract(query: str, universe_sectors: Iterable[str], model: Optional[str]) -> dict:
    """Call Claude with STRICT forced tool use and return the raw params dict.

    anthropic is LAZY-imported here (never at module top) so the module loads
    without the optional dependency. The ``set_screen`` tool is ``strict=True``,
    so every property is ``required`` and ``additionalProperties`` is False; the
    numeric knobs carry NO ``minimum``/``maximum`` (strict forbids them — clamping
    happens in :func:`validate_request`). The ``sectors`` enum is built from the
    live universe so the model can only emit real sectors (still re-validated). May
    raise — :func:`parse_query`'s broad ``except`` handles the fallback.
    """
    import anthropic  # lazy, inside the function

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    sector_list = sorted({s for s in (universe_sectors or []) if isinstance(s, str)})
    tool = {
        "name": "set_screen",
        "description": (
            "Map the user's request into stock-screener parameters. Only "
            "translate the request into these knobs — never give buy/sell/hold "
            "advice or opinions."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "profile": {"type": "string", "enum": ["long_term", "swing", "momentum"]},
                "n_names": {"type": "integer"},  # NO min/max — strict forbids; clamp in validate_request
                "min_score": {"type": "number"},  # NO min/max
                "sectors": (
                    {"type": "array", "items": {"type": "string", "enum": sector_list}}
                    if sector_list
                    else {"type": "array", "items": {"type": "string"}}
                ),
                "text": {"type": "string"},
                "earnings_only": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": [
                "profile", "n_names", "min_score", "sectors", "text",
                "earnings_only", "explanation",
            ],
        },
    }
    resp = client.messages.create(
        model=_resolve_model(model),
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "set_screen"},
        messages=[{"role": "user", "content": query}],
    )
    return next(b.input for b in resp.content if b.type == "tool_use")


def parse_query(
    query: str, *, universe_sectors: Iterable[str], model: Optional[str] = None
) -> ScreenRequest:
    """Parse ``query`` via the LLM when available, else the rule-based parser.

    Uses the LLM only when :func:`agent_available` (a key is set AND the SDK
    imports); on ANY exception — typed anthropic errors, a malformed response, a
    missing tool block, even anthropic not actually importable despite find_spec —
    it falls back to :func:`rule_based_parse`. Both paths run through
    :func:`validate_request`, so the result is always a valid, clamped request.
    The LLM result's ``explanation`` is prefixed with ``"LLM"`` for transparency.
    """
    if not agent_available():
        return rule_based_parse(query, universe_sectors)
    try:
        raw = _llm_extract(query, universe_sectors, model)  # may raise
        req = validate_request(raw, universe_sectors)
        # Ensure a transparency note even if the model omitted one.
        if not req.explanation:
            req = replace(
                req,
                explanation=(
                    f"LLM ({_resolve_model(model)}): "
                    f"profile={req.profile}, top {req.n_names}"
                ),
            )
        else:
            req = replace(req, explanation="LLM: " + req.explanation)
        return req
    except Exception:  # noqa: BLE001 - any failure degrades to the offline parser
        return rule_based_parse(query, universe_sectors)
