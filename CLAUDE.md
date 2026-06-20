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
> Header-led UI redesign + Sector Heatmap (DONE, branch `feat/header-redesign-sector-heatmap`) — promoted the high-frequency controls out of the one-long-sidebar into a sticky **header** (`screener/ui/header.py`: view nav, dual-purpose search, profile switcher, ➕ add-ticker + ⚙ Settings popovers, ★ watchlist + 🕘 recent), slimmed the sidebar to the post-scan **Filters** block, and added a Finviz-style **Sector Heatmap** view — a pure, streamlit-free `screener/sector_heatmap.py` squarified treemap over the cached scan (tiles sized by combined market cap, coloured by 3-month momentum; `momentum_3m` is a fraction so labels use `%`), with a sector drill-down that stages a filter + view switch back into the Screener. Plus extras: watchlist `★` (persisted as a sorted list ↔ set via the existing localStorage path), recent-scans (re-applied via `_pending_*` staging through the one guarded scan site), and `/`-to-focus search. The **cold-scan guard (one engine call site) is preserved AND now locked** by `tests/test_app_guard.py` (the repo's first `AppTest`: 0 engine calls on cold load / view switch / filter, exactly 1 on Run). **Follow-on (DONE, same branch):** a dedicated **filter-by-ticker** box (sidebar `f_ticker`; symbol-only substring via a new `apply_filters(ticker=)` kwarg), a no-hard-filter **"All Tickers"** profile ranked by market cap (inserted after Swing; synced into `agent.VALID_PROFILES` + the `set_screen` enum), and a **ticker-centric results table** — Rank pinned as the true first column, and the Name column removed from the grid but surfaced as an `ⓘ` hover tooltip on the ticker (kept in row data + CSV; pure `column_order`/`table_view` contract unchanged). 376 tests pass. **Next (Milestone B):** opt-in, gated/off-by-default MCP tool consumption. Prior: economic-event calendar (DONE, merged via PR #9); Stage-5 refactor + universe/defaults/persistence/auto-fetch (DONE); swappable LLM backend (DONE); tactical TA readouts (DONE). See PLAN.md / DECISIONS.md.

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
