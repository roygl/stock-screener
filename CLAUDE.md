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
> Stage-5 refactor + universe/defaults/persistence/auto-fetch (DONE, branch `feat/stage5-universe-gemini-persistence`) — (5) did the deferred behavior-preserving split: `screener/display.py` → `screener/display/` package and `app.py` UI → `screener/ui/` package (68-line thin entrypoint; all Streamlit ordering + `session_state` keys + single engine call site preserved); deleted the dead top-level `universe.py`. (6) universe 503 → 618 (+115 popular non-S&P names incl. CRWV/NBIS/OKLO/BMNR/MSTR, GICS sectors). (7) default universe size = **all names** and default LLM = **Gemini** (`gemini-2.5-flash`). (8) **remember sidebar choices in the browser** via `streamlit-local-storage` (profile/engine/size/density survive refresh). (9) **NL free-text auto-fetch**: an unknown named ticker is fetched via yfinance, saved to `data/universe.csv`, and unioned into the scan. 334 tests; browser + live-fetch verified. **Deferred:** universe-wide chart-pattern screening. Prior: declutter/surface/Gemini-reliability (DONE); swappable LLM backend (DONE); tactical TA readouts (DONE). See PLAN.md / DECISIONS.md.

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
