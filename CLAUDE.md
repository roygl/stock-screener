# CLAUDE.md — Project Brief

> Keep this at the repo root. It is the first thing Claude (or any contributor) reads.
> Update the **Active task** line whenever you start a new milestone.

## Purpose
A stock **screener** for US large-cap equities. The user picks a **profile**
(investing style) and gets a ranked, sortable, filterable table of tickers that
match that style's signals. It automates the *systematic scan / watchlist-building*
step — it does not replace charting, broker execution, or live alerting.

## Tech stack (v1)
- **Language:** Python
- **UI:** Streamlit (single-language, built-in table/filter/toggle widgets, fast deploy)
- **Equity data:** `yfinance` (free, end-of-day) behind a swappable `DataProvider` interface
- **Indicators:** pandas / numpy (compute momentum, RSI, MAs, MACD, etc. from price history)
- **Caching:** local cache layer so we don't hammer Yahoo

## Active task
> In-app economic-event calendar (Milestone A, DONE, branch `feat/economic-event-calendar`) — pure `screener/calendar.py` over a bundled public-domain CSV (`data/economic_events.csv`: FOMC/CPI/jobs) plus the per-ticker earnings the app already fetches, surfaced as a **pull-based** "Upcoming events" panel (`screener/ui/events_panel.py`) with days-until countdowns, a 3-tier impact tag + high-impact-only filter, advance-warning flags, and a per-symbol event-risk badge generalized from swing-only to any profile (reuses the engine's `days_to_earnings`, no refetch). Offline-first, manual ~annual refresh via `scripts/refresh_economic_calendar.py` (no host cron), educational "heightened expected volatility" framing reusing the existing disclaimer. **Next (Milestone B):** opt-in, gated/off-by-default MCP tool consumption. Prior: Stage-5 refactor + universe/defaults/persistence/auto-fetch (DONE); swappable LLM backend (DONE); tactical TA readouts (DONE). See PLAN.md / DECISIONS.md.

## Key decisions (full log in DECISIONS.md)
- Build a **dashboard first**, add a natural-language **agent layer later**.
- **Streamlit** for v1; graduate to FastAPI + React only if it becomes a multi-user product.
- **Equities only in v1**; **crypto is v2** (separate CoinGecko data pipeline; no fundamentals).
- Data access lives behind one interface so `yfinance` can be swapped for a paid API (e.g. FMP) later. The **NL agent's LLM backend is likewise swappable** — a provider registry (Anthropic native + an OpenAI-compatible family) selected in the sidebar, same env-driven, offline-first ethos (no key UI; degrades to the rule-based parser).
- The tool **describes and ranks**. It now also shows an **educational "Buy zone"** (entry band) **with a not-advice disclaimer** — still **no execution and no sell/exit management**; only a descriptive entry zone is surfaced.

## What to ignore
- `.venv/`, `__pycache__/`, `*.pyc`, `.cache/`, local data dumps, `*.log`

## Working agreement
- One milestone per conversation. Restate the plan before coding.
- Narrow diffs over broad refactors.
- Record any new choice in DECISIONS.md (newest first).
