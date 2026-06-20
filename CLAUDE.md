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
> Post-MVP: chart-pattern TA (DONE) + dashboard help/explanations UI (DONE). Backlog: universe-wide pattern screening. See PLAN.md.

## Key decisions (full log in DECISIONS.md)
- Build a **dashboard first**, add a natural-language **agent layer later**.
- **Streamlit** for v1; graduate to FastAPI + React only if it becomes a multi-user product.
- **Equities only in v1**; **crypto is v2** (separate CoinGecko data pipeline; no fundamentals).
- Data access lives behind one interface so `yfinance` can be swapped for a paid API (e.g. FMP) later.
- The tool **describes and ranks — it does not give buy/sell advice**.

## What to ignore
- `.venv/`, `__pycache__/`, `*.pyc`, `.cache/`, local data dumps, `*.log`

## Working agreement
- One milestone per conversation. Restate the plan before coding.
- Narrow diffs over broad refactors.
- Record any new choice in DECISIONS.md (newest first).
