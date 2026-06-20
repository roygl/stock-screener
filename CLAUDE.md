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
> Post-MVP: tactical TA readouts (DONE) — Buy zone + support/resistance + overextension/parabolic (EOD 1w/1d/1mo); overextension + in-buy-zone also universe-wide (table columns + sidebar filters). Backlog: universe-wide chart-pattern screening. See PLAN.md.

## Key decisions (full log in DECISIONS.md)
- Build a **dashboard first**, add a natural-language **agent layer later**.
- **Streamlit** for v1; graduate to FastAPI + React only if it becomes a multi-user product.
- **Equities only in v1**; **crypto is v2** (separate CoinGecko data pipeline; no fundamentals).
- Data access lives behind one interface so `yfinance` can be swapped for a paid API (e.g. FMP) later.
- The tool **describes and ranks**. It now also shows an **educational "Buy zone"** (entry band) **with a not-advice disclaimer** — still **no execution and no sell/exit management**; only a descriptive entry zone is surfaced.

## What to ignore
- `.venv/`, `__pycache__/`, `*.pyc`, `.cache/`, local data dumps, `*.log`

## Working agreement
- One milestone per conversation. Restate the plan before coding.
- Narrow diffs over broad refactors.
- Record any new choice in DECISIONS.md (newest first).
