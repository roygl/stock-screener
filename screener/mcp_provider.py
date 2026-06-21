"""Opt-in, OFF-by-default consumption of an EXTERNAL stock-info MCP server.

Milestone B. The screener can OPTIONALLY enrich the *inspected* ticker's detail
panel with a supplementary fundamentals + next-earnings snapshot pulled from a
third-party MCP server (a **local stdio** server, e.g. ``uvx yfmcp@latest``).

This is a CLIENT of an untrusted server, so the whole path is gated and contained
(see DECISIONS.md). The deterministic ``yfinance`` engine stays the backbone; this
is a thin overlay surfaced only when a user double-clicks a single ticker — it is
NEVER on the universe-scan path, so it cannot affect the cold-scan guard.

Containment, in one place:

- **OFF by default.** Nothing happens unless ``MCP_STOCK_DATA_ENABLED`` is truthy.
  Unset ⇒ zero ``mcp`` import, zero network, behavior identical to the pure app.
- **Default-deny tool allow-list.** Only the read-only tools in
  :data:`ALLOWED_TOOLS` are ever invoked; any other tool the server advertises is
  ignored, even if added in a later server update.
- **Untrusted output.** Every result is coerced + range-clamped by
  :func:`validate_mcp_payload` / :func:`validate_mcp_earnings`, which (like
  :func:`screener.agent.validate_request`) tolerate missing keys, wrong types,
  ``None``/``NaN``, and hostile values and **never raise**.
- **Rug-pull guard.** The server's advertised tool surface is fingerprinted on
  connect; if ``MCP_STOCK_DATA_TOOLS_SHA256`` is pinned and the fingerprint
  diverges, the client refuses to talk to it.
- **Fail-soft.** Missing SDK, dead/slow server (hard timeout), oversized payload,
  or bad data ⇒ empty/``None``; the panel simply omits the supplementary section.
- **No key in the UI.** The optional auth token is read from the environment only.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import shlex
from typing import Optional

from .provider import (
    DEFAULT_LOOKBACK_DAYS,
    DataProvider,
    Fundamentals,
    _canonical,
    _coerce_date,
    _first_num,
    _next_on_or_after,
    _num,
    _text,
    empty_price_frame,
)

log = logging.getLogger(__name__)

# --- env gate + config (the single source of truth) ----------------------
ENABLED_ENV = "MCP_STOCK_DATA_ENABLED"          # master gate; OFF unless truthy
CMD_ENV = "MCP_STOCK_DATA_CMD"                   # stdio launch command (override)
TOKEN_ENV = "MCP_STOCK_DATA_TOKEN"              # optional bearer (unused by yfmcp)
TOOLS_SHA_ENV = "MCP_STOCK_DATA_TOOLS_SHA256"   # optional pinned tool fingerprint

# Default: narumiruna/yfinance-mcp — no API key, local stdio, same Yahoo source as
# the engine, so the whole MCP code path is exercisable for free.
DEFAULT_CMD = "uvx yfmcp@latest"

# Default-deny: ONLY these reviewed, read-only tools may be invoked. yfmcp's
# ``yfinance_get_ticker_info`` returns the full Yahoo ``.info`` snapshot (company
# info + valuation + the next-earnings timestamps) — all the detail panel needs
# for supplementary fundamentals + earnings. Every other advertised tool (news,
# options, screeners, holders, price history) is intentionally NOT called.
INFO_TOOL = "yfinance_get_ticker_info"
ALLOWED_TOOLS = frozenset({INFO_TOOL})

# Containment limits.
READ_TIMEOUT_S = 30.0                 # hard per-call ceiling (connect + read + parse)
MAX_RESULT_BYTES = 256 * 1024         # reject oversized payloads (context-flood vector)

# Sane numeric bounds so a hostile/garbled server can't push absurd values into the
# panel. Out-of-range ⇒ dropped to None (not displayed) rather than shown.
_PE_MAX = 100_000.0
_MARKET_CAP_MAX = 1e15                # $1,000T — far above any real market cap
_GROWTH_ABS_MAX = 1000.0             # |YoY growth fraction| cap (1000 = 100,000%)

# Instruction-like / anomalous markers we refuse to silently trust in a
# server-supplied tool description (defense-in-depth: v1 never feeds descriptions
# to an LLM, but the audit flags + logs them so a poisoned server is visible).
_SUSPICIOUS_MARKERS = (
    "ignore previous", "ignore all", "system:", "disregard", "override",
    "jailbreak", "exfiltrat", "‮", "​",
)


# --- gate + availability (mirrors agent.availability_status) --------------
def _env_flag(name: str) -> bool:
    """True iff env var ``name`` is set to a truthy token (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def mcp_enabled() -> bool:
    """Master gate: is the supplementary-MCP overlay on? OFF unless explicitly set."""
    return _env_flag(ENABLED_ENV)


def mcp_command() -> str:
    """The configured stdio launch command (default: the no-key yfmcp server)."""
    return (os.environ.get(CMD_ENV) or DEFAULT_CMD).strip()


def sdk_available() -> bool:
    """True iff the optional ``mcp`` client SDK is importable (never imports it)."""
    import importlib.util

    return importlib.util.find_spec("mcp") is not None


def availability_status() -> "tuple[bool, str]":
    """``(usable, human reason)`` for the overlay — drives the ⚙ Settings status line.

    Mirrors :func:`screener.agent.availability_status`: gate-off / SDK-missing /
    ready, so the UI is self-diagnosing instead of silently doing nothing. Never
    raises.
    """
    try:
        if not mcp_enabled():
            return False, f"off (set {ENABLED_ENV}=true to enable)"
        if not sdk_available():
            return False, "mcp SDK not importable (pip install mcp)"
        return True, f"on ({mcp_command()})"
    except Exception:  # noqa: BLE001 - status probe must never raise
        return False, "unavailable"


# --- the safety layer: coerce + clamp UNTRUSTED server output, never raise ---
def _clamp_num(value, *, lo=None, hi=None, abs_max=None) -> Optional[float]:
    """:func:`screener.provider._num` then an optional range gate.

    Out-of-range, non-numeric, or ``NaN`` ⇒ ``None`` (so a hostile value is
    dropped, not displayed). Never raises.
    """
    f = _num(value)
    if f is None:
        return None
    if abs_max is not None and abs(f) > abs_max:
        return None
    if lo is not None and f < lo:
        return None
    if hi is not None and f > hi:
        return None
    return f


def validate_mcp_payload(symbol: str, raw: object) -> Fundamentals:
    """Coerce an UNTRUSTED ``yfinance_get_ticker_info`` payload into :class:`Fundamentals`.

    The single safety gate for MCP fundamentals, modeled on
    :func:`screener.agent.validate_request`: tolerates missing keys, wrong types,
    ``None``, ``NaN``, and hostile values, and **never raises**. Only the specific
    fields the detail panel consumes are extracted (everything else in the payload
    is ignored), and numerics are range-clamped so an absurd value is dropped to
    ``None`` rather than rendered.
    """
    info = raw if isinstance(raw, dict) else {}
    return Fundamentals(
        symbol=_canonical(symbol),
        name=_text(info.get("longName") or info.get("shortName")),
        sector=_text(info.get("sector")),
        market_cap=_clamp_num(info.get("marketCap"), lo=0.0, hi=_MARKET_CAP_MAX),
        forward_pe=_clamp_num(info.get("forwardPE"), abs_max=_PE_MAX),
        trailing_pe=_clamp_num(info.get("trailingPE"), abs_max=_PE_MAX),
        revenue_growth=_clamp_num(_first_num(info, "revenueGrowth"), abs_max=_GROWTH_ABS_MAX),
        earnings_growth=_clamp_num(
            _first_num(info, "earningsGrowth", "earningsQuarterlyGrowth"), abs_max=_GROWTH_ABS_MAX
        ),
        industry=_text(info.get("industry")),
        business_summary=_text(info.get("longBusinessSummary")),
    )


def validate_mcp_earnings(raw: object, *, today: Optional[dt.date] = None) -> Optional[dt.date]:
    """Next earnings date from an UNTRUSTED ticker-info payload, or ``None``. Never raises.

    Yahoo's ``.info`` carries the next report as unix-second timestamps
    (``earningsTimestamp`` / ``earningsTimestampStart`` / ``earningsTimestampEnd``)
    and/or an ``earningsDate`` list. We read whichever is present, coerce to a
    date, and pick the soonest on-or-after ``today`` (same rule as
    :func:`screener.provider._next_on_or_after`).
    """
    info = raw if isinstance(raw, dict) else {}
    today = today or dt.date.today()
    candidates: list = []
    for key in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
        ts = _num(info.get(key))
        if ts is None:
            continue
        try:
            candidates.append(dt.datetime.fromtimestamp(ts, dt.timezone.utc).date())
        except (OverflowError, OSError, ValueError):
            continue
    raw_dates = info.get("earningsDate")
    if isinstance(raw_dates, (list, tuple)):
        candidates.extend(d for d in (_coerce_date(x) for x in raw_dates) if d is not None)
    return _next_on_or_after(candidates, today)


# --- tool-surface fingerprint (rug-pull guard) ---------------------------
def _tool_attr(tool, name: str, key: str):
    """Read ``name``/``key`` off an SDK Tool object OR a plain dict (test fakes)."""
    if isinstance(tool, dict):
        return tool.get(key)
    return getattr(tool, name, None)


def tools_fingerprint(tools) -> str:
    """Stable SHA-256 over the server's advertised ``(name, description, schema)``.

    The pin point for rug-pull detection: store this as
    ``MCP_STOCK_DATA_TOOLS_SHA256`` and the client refuses any server whose tool
    surface later changes. Pure + order-independent (sorted), so two equal tool
    sets in different orders hash identically.
    """
    items = []
    for tool in tools or []:
        items.append(
            json.dumps(
                {
                    "name": _tool_attr(tool, "name", "name"),
                    "description": _tool_attr(tool, "description", "description"),
                    "schema": _tool_attr(tool, "inputSchema", "inputSchema"),
                },
                sort_keys=True,
                default=str,
            )
        )
    blob = "\n".join(sorted(items))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _suspicious_tools(tools) -> "list[str]":
    """Names of tools whose description carries an instruction-like / hidden marker."""
    flagged = []
    for tool in tools or []:
        desc = str(_tool_attr(tool, "description", "description") or "").lower()
        if any(marker in desc for marker in _SUSPICIOUS_MARKERS):
            flagged.append(str(_tool_attr(tool, "name", "name")))
    return flagged


def audit_tools(tools) -> None:
    """Fingerprint + screen the server's tool surface; raise to refuse a bad one.

    - Logs the fingerprint (so the user can pin ``MCP_STOCK_DATA_TOOLS_SHA256``).
    - Logs any tool with a suspicious description (poisoning attempt).
    - If a pin is set and the fingerprint diverges, raises (rug-pull guard).
    """
    fingerprint = tools_fingerprint(tools)
    for name in _suspicious_tools(tools):
        log.warning("mcp: tool %r has a suspicious description — not trusted", name)
    pin = os.environ.get(TOOLS_SHA_ENV, "").strip().lower()
    if pin and fingerprint != pin:
        raise ValueError(
            f"mcp: tool fingerprint {fingerprint} != pinned {pin} (server changed; refusing)"
        )
    log.info("mcp: server tool fingerprint=%s (%d tools)", fingerprint, len(tools or []))


# --- result decoding (pure; unit-testable without the SDK) ----------------
def _extract_text(result, *, max_bytes: int = MAX_RESULT_BYTES) -> str:
    """Concatenate the text content blocks of a tool result; enforce the size cap.

    Raises if the server flagged an error or the payload exceeds ``max_bytes``
    (oversized output is a context-flood / injection vector). Non-text blocks are
    ignored.
    """
    if _tool_attr(result, "isError", "isError"):
        raise ValueError("mcp: tool returned isError=True")
    content = _tool_attr(result, "content", "content") or []
    parts = [
        text
        for block in content
        if isinstance((text := _tool_attr(block, "text", "text")), str)
    ]
    joined = "".join(parts)
    if len(joined.encode("utf-8")) > max_bytes:
        raise ValueError(f"mcp: result exceeds {max_bytes} bytes")
    return joined


def _parse_info_json(text: str) -> dict:
    """Best-effort parse of a ticker-info result into a dict (else ``{}``).

    yfmcp returns the ``.info`` mapping as JSON text. A non-dict or unparseable
    payload yields ``{}`` (the validators then produce empty ``Fundamentals``).
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


# --- the contained client ------------------------------------------------
class MCPProvider(DataProvider):
    """A CONTAINED client of an external stdio stock-info MCP server.

    Implements the :class:`DataProvider` methods the detail panel uses
    (``fundamentals`` + ``earnings_date``). ``price_history`` is intentionally NOT
    sourced from MCP in v1 — the engine's prices stay on yfinance — so it returns
    an empty frame. Every method is fail-soft: any failure yields empty/``None``.
    """

    def __init__(
        self,
        command: Optional[str] = None,
        *,
        token: Optional[str] = None,
        read_timeout_s: float = READ_TIMEOUT_S,
    ) -> None:
        self._command = command or mcp_command()
        self._token = token if token is not None else (os.environ.get(TOKEN_ENV) or None)
        self._read_timeout_s = read_timeout_s

    # DataProvider API ----------------------------------------------------
    def price_history(self, symbol: str, *, lookback_days: int = DEFAULT_LOOKBACK_DAYS):
        # Prices stay on the yfinance backbone in v1 — MCP supplements fundamentals
        # + earnings only. An empty frame keeps the DataProvider contract intact.
        return empty_price_frame()

    def fundamentals(self, symbol: str) -> Fundamentals:
        return validate_mcp_payload(symbol, self._ticker_info(symbol))

    def earnings_date(self, symbol: str) -> Optional[dt.date]:
        return validate_mcp_earnings(self._ticker_info(symbol))

    def ticker_snapshot(self, symbol: str) -> "tuple[Fundamentals, Optional[dt.date]]":
        """Fundamentals + next-earnings from ONE tool call (the panel's entry point)."""
        raw = self._ticker_info(symbol)
        return validate_mcp_payload(symbol, raw), validate_mcp_earnings(raw)

    # network boundary (the ONLY impure part; monkeypatched in tests) -----
    def _ticker_info(self, symbol: str) -> dict:
        """Call the allow-listed ticker-info tool and return a parsed dict (or ``{}``).

        Fully contained: allow-list + hard timeout + size cap live below; ANY
        failure ⇒ ``{}`` (never raises, never partially crashes the panel).
        """
        if INFO_TOOL not in ALLOWED_TOOLS:  # defensive: tool must be allow-listed
            return {}
        try:
            text = self._call_tool(INFO_TOOL, {"symbol": _canonical(symbol)})
        except Exception as exc:  # noqa: BLE001 - fail soft to no-data
            log.warning("mcp: ticker_info(%s) failed: %s", symbol, exc)
            return {}
        return _parse_info_json(text)

    def _call_tool(self, name: str, arguments: dict) -> str:
        """Run ONE allow-listed tool over a fresh stdio session; return its text.

        Synchronous wrapper around the async ``mcp`` client (the SDK is async-only),
        which lazy-imports ``mcp`` so the app imports fine with the SDK absent and
        enforces the allow-list + a hard timeout.
        """
        if name not in ALLOWED_TOOLS:
            raise ValueError(f"mcp: tool {name!r} is not allow-listed")
        import asyncio

        return asyncio.run(asyncio.wait_for(self._call_tool_async(name, arguments), self._read_timeout_s))

    async def _call_tool_async(self, name: str, arguments: dict) -> str:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        argv = shlex.split(self._command)
        if not argv:
            raise ValueError("mcp: empty launch command")
        params = StdioServerParameters(command=argv[0], args=argv[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                audit_tools(getattr(await session.list_tools(), "tools", []))
                return _extract_text(await session.call_tool(name, arguments))


# --- detail-panel entry point --------------------------------------------
def supplementary_for_symbol(symbol: str) -> dict:
    """Supplementary fundamentals + next-earnings for ONE symbol, or ``{}`` when OFF.

    The detail-panel entry point (wrapped by a date-keyed ``st.cache_data`` memo in
    :mod:`screener.ui.caching`). Returns ``{}`` immediately — NO import, NO network
    — unless :func:`mcp_enabled`. Fail-soft: any error yields ``{}``. The result is
    a plain JSON-able dict so it pickles cleanly through ``st.cache_data``::

        {"fundamentals": {<Fundamentals.to_dict>},
         "earnings_date": "YYYY-MM-DD" | None,
         "command": "<launch command>"}     # or {} when disabled / nothing came back
    """
    if not mcp_enabled():
        return {}
    try:
        funds, earnings = MCPProvider().ticker_snapshot(symbol)
    except Exception as exc:  # noqa: BLE001 - the overlay must never crash the panel
        log.warning("mcp: supplementary_for_symbol(%s) failed: %s", symbol, exc)
        return {}
    if funds.is_empty and earnings is None:
        return {}
    return {
        "fundamentals": funds.to_dict(),
        "earnings_date": earnings.isoformat() if earnings else None,
        "command": mcp_command(),
    }


if __name__ == "__main__":  # smoke test + print the tool fingerprint to pin
    import sys

    logging.basicConfig(level=logging.INFO)
    if not mcp_enabled():
        print(f"MCP overlay is OFF. Set {ENABLED_ENV}=true (cmd: {mcp_command()}) to try it.")
        raise SystemExit(0)
    sym = (sys.argv[1:] or ["AAPL"])[0]
    funds, earnings = MCPProvider().ticker_snapshot(sym)
    print(f"[{sym}] sector={funds.sector!r} fwdPE={funds.forward_pe} "
          f"mktcap={funds.market_cap} next_earnings={earnings}")
    print("Pin the server's tool surface via "
          f"{TOOLS_SHA_ENV}=<the 'fingerprint=' value logged above>.")
