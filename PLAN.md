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

## Milestone 5 — Dashboard  ✅ DONE
- Profile **toggle** (the asset-class toggle is a no-op stub until v2 crypto).
- Sortable / filterable results table.
- Per-row "why it ranks" expander (which signals fired).
- Earnings-window warning badge on swing results.
- **Done when:** a user can pick a profile and read/sort/filter the ranked list in the browser. ✅
  `app.py` (thin Streamlit wiring) + `screener/display.py` (pure, streamlit/network-free) +
  `tests/test_display.py` (24 offline tests). Profile radio, disabled asset-class stub, native
  header-sort + four sidebar filters, and the swing earnings badge/banner all wired to `run_screen`.
  "Why it ranks" is a **select-then-explain detail panel** below the table (value/percentile/
  contribution per signal) — st.dataframe has no per-row expander; see DECISIONS.md. Cold-scan
  guard: engine called at one site behind a "Run scan" button + small default slice + `st.cache_data`.

## Milestone 6 — Validate + polish  ✅ DONE
- Spot-check numbers against Yahoo Finance.
- Edge cases: missing data, thin volume, recent IPOs.
- Set refresh cadence for the end-of-day pull.
- Short README on how to run.
- **Done when:** numbers reconcile and the app handles a full run without crashing. ✅
  Reconciled the math (offline hand-checks of SMA / momentum / Wilder-RSI / rel-volume / the full
  percentile score + the contributions-sum-to-score invariant, plus a live AAPL spot-check);
  `tests/test_edge_cases.py` (13 tests) pins fail-soft behavior on missing data / thin-or-zero
  volume / recent IPOs — incl. the documented `dist_52w_high` IPO tilt, kept by design (DECISIONS.md);
  `README.md` covers setup/run/profiles/ranking + a refresh-cadence section. 141 offline tests green;
  app boots clean.

---

## Post-MVP (priority order, set 2026-06-20)
- **Natural-language agent layer** ✅ DONE — plain English → validated screen params → `run_screen`.
  `screener/agent.py` (pure; anthropic lazy-imported) + `tests/test_agent.py` (20 offline tests) + a sidebar
  "Interpret & run" box. Offline-first: Anthropic Claude (`claude-opus-4-8`, strict forced tool use) when
  `ANTHROPIC_API_KEY` is set, else a deterministic rule-based parser; one `validate_request` safety layer clamps
  every result, so the engine stays the source of truth (a layer on top, not a replacement). See DECISIONS.md.
- **Chart-pattern technical analysis** ✅ DONE — descriptive detection of common shapes
  (head & shoulders + inverse, double top/bottom, cup & handle, ascending/descending/symmetric
  triangles, rising/falling wedges) via swing-pivot + geometric rules; a per-ticker readout in the
  detail panel across **EOD 1w / 1d / 1mo** (resampled from the daily bars; NO intraday/4h), never a
  buy/sell call. `screener/patterns.py` (pure) + `tests/test_patterns.py` (36 offline tests).
  Precision-first (volatility gates + mutual-exclusion de-dup + a confidence floor); the residual
  noisy-shape limitation is documented. See DECISIONS.md.
- **Dashboard help / explanations UI** ✅ DONE (shipped concurrently) — per-signal descriptions,
  score/percentile/contribution tooltips, profile descriptions, and a "How to read this" glossary in
  `screener/display.py` (`tests/test_display.py` 24→32).
- **Universe-wide pattern screening** (backlog) — let a profile filter/rank by detected pattern, not
  just show patterns for the inspected ticker.
- **Crypto / live asset-class toggle — DEPRIORITIZED** — `CoinGeckoProvider` + crypto
  universe (swing/momentum only, no fundamentals). On hold per the 2026-06-20 decision;
  the asset-class toggle stays a disabled stub.

## v3 (ML research track — gated on a backtest harness; see DECISIONS.md)
- **Backtest harness first** (prerequisite for any ML): walk-forward / purged CV,
  look-ahead guards, transaction-cost modelling. Likely its own milestone.
- **ML enters as a SIGNAL, not the engine:** a model's output becomes one more
  `SignalSpec` (e.g. `ml_forward_return_rank`) feeding the deterministic percentile
  ranker — the explainable ranking stays the backbone and the model stays toggle-off.
- **Start simple, escalate only on evidence:** gradient-boosted trees (LightGBM/XGBoost)
  over the existing `snapshot()` features first; reach for a sequence model
  (e.g. Temporal Fusion Transformer) only if the simple model shows real out-of-sample edge.
- Keep it descriptive ("model rates this setup top-decile vs peers"), not predictive —
  stays inside "describes, does not advise".
