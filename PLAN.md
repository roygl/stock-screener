# PLAN.md — Milestone Plan

Build order for v1. **One milestone per conversation.** Restate the plan and confirm
scope before writing code in each. Check each milestone's output against `spec.md`.

---

## Milestone 1 — Project setup  ✅ DONE
- Init repo; add `CLAUDE.md`, `spec.md`, `PLAN.md`, `DECISIONS.md`.
- Python env (`.venv`), `requirements.txt` (streamlit, yfinance, pandas, numpy).
- Streamlit "hello world" skeleton that runs locally.
- Define the **universe**: load a ticker list (S&P 500) from a static file to start.
- **Done when:** `streamlit run app.py` opens a page and the ticker list loads. ✅

## Milestone 2 — Data layer
- Define a `DataProvider` interface (methods: `price_history`, `fundamentals`,
  `earnings_date`).
- Implement `YFinanceProvider`.
- Add a **local cache** (e.g. parquet/SQLite) keyed by ticker + date so repeated
  runs don't re-hit Yahoo.
- Handle missing/partial data gracefully.
- **Done when:** one call returns clean price history + fundamentals for the universe,
  served from cache on the second run.

## Milestone 3 — Indicator engine  ✅ DONE
- Pure functions over price data:
  - Momentum: 1 / 3 / 6 / 12-mo trailing returns
  - Moving averages: generic SMA(n) and EMA(n) — supports 20/50/150 SMA and 5/9 EMA
  - RSI, MACD
  - Relative volume (vs N-day average)
  - Distance from 52-week high
- MA structure helpers: price-vs-MA, stacking (20>50>150), 5/9 cross detection.
- **Done when:** every variable in spec §6 computes correctly for a sample ticker
  (unit-tested against a hand-checked example). ✅ `screener/indicators.py` + `snapshot()`
  (21 spec §6 keys); 63 unit tests (hand-checked + property + edge), green under both
  `pytest` and the standalone runner.

## Milestone 4 — Profile + ranking engine
- Express each profile as a **config**: filters (hard cutoffs) + weights (for scoring).
  - Long-term, Swing, Momentum/Growth (per spec §5)
  - Swing config includes: relative volume > 2×, 5/9 EMA cross, MACD/RSI, 10/20 EMA
    pullback, sector-strength membership, earnings-window flag.
- Engine: apply filters → score → rank → output a table with a per-row reason breakdown.
- Compute **sector strength** (group performance) to support the swing "leading sector" filter.
- **Done when:** selecting a profile returns a sensible scored, ranked DataFrame.

## Milestone 5 — Dashboard
- Profile **toggle** (the asset-class toggle is a no-op stub until v2 crypto).
- Sortable / filterable results table.
- Per-row "why it ranks" expander (which signals fired).
- Earnings-window warning badge on swing results.
- **Done when:** a user can pick a profile and read/sort/filter the ranked list in the browser.

## Milestone 6 — Validate + polish
- Spot-check numbers against Yahoo Finance.
- Edge cases: missing data, thin volume, recent IPOs.
- Set refresh cadence for the end-of-day pull.
- Short README on how to run.
- **Done when:** numbers reconcile and the app handles a full run without crashing.

---

## v2 (after MVP ships)
- `CoinGeckoProvider` + crypto universe; swing/momentum profiles only (no fundamentals).
- Asset-class toggle becomes live.
- Natural-language agent layer on top of the deterministic engine.
