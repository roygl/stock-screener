# DECISIONS.md — Decision Log

Newest first. Each entry: the decision, and *why*, so nothing gets re-argued later.

---

2026-06-19: **Milestone 4 ranking (PLANNED) = cross-sectional PERCENTILE-RANK scoring;
swing "leading sector" = TOP 3 sectors.** Each profile scores a name as a weighted sum of
per-signal percentile ranks — every signal ranked 0–1 across the screened universe and
direction-adjusted so "lower is better" inputs (e.g. forward P/E) invert. Chosen over z-score
and min–max because it is outlier-proof (no single runaway name dominates the score) and yields
the clean "Nth-percentile on signal X" reason breakdown the dashboard needs (spec §8). Sector
strength ranks the 11 GICS sectors by median 3-month member return; the swing leading-sector
filter passes only names in the top 3. Default policies (revisit during the build): a missing
signal scores at the neutral 0.5 rather than penalize, and a hard filter on a missing value
fails closed.

2026-06-19: **Milestone 3 indicator engine (DONE) — fixed numeric conventions, locked so
they don't get silently re-broken.** `screener/indicators.py` is a pure, network-free layer
over the canonical price frame; the profiles consume it in M4. The non-obvious choices:
- **RSI = Wilder's smoothing with the canonical seed.** The first average is the simple mean
  of the first `n` changes (placed at index `n`), then `avg = (avg·(n-1)+x)/n`. We do NOT use
  pandas `ewm(alpha=1/n, adjust=False)`: its recursion seeds from bar 0 and measurably diverges
  from Wilder (e.g. 54.73 vs 52.14 on a sample series). A test pins rsi() against an independent
  reference loop — do not "simplify" it back to `ewm`.
- **No lookahead; `min_periods == window`** so an under-filled window is NaN, with ONE
  documented exception: `distance_from_high` uses `min_periods=1` so a recent IPO measures
  distance from its *available* high instead of NaN.
- **`relative_volume` excludes the current bar** (`volume / volume.shift(1).rolling(n).mean()`)
  and zero-guards the divisor to NaN, so a degenerate window reads "undefined" rather than `+inf`
  (which would otherwise sort straight to the top of the M4 ranker).
- **`price_above_ma` returns `None` (not `False`) when undefined** — "unknown" stays distinct
  from "below"; `is_stacked` returns `False` if any MA is undefined.
- **`snapshot(price_df)` is a fixed 21-key per-ticker feature dict** (exactly the spec §6 PRICE
  variables) — the surface M4 reads. Profile-specific extras (e.g. swing's 10/20 EMA pullback)
  are computed on demand via the generic `sma(n)`/`ema(n)`/`pct_from_ma`, per the per-profile-MA
  decision below. Built via an implement → 4-lens adversarial-verify → fix workflow; the panel
  caught the RSI seed bug and four minor robustness gaps (incl. the rel-volume `+inf` guard).

2026-06-19: **Test layer = framework-agnostic; `pytest` is an optional runner.** `tests/`
holds plain `assert` + `math.isclose` functions with a `__main__` runner and imports neither
`pytest` nor `yfinance`, so the suite runs both under `pytest` and as a bare script
(`python tests/test_indicators.py`) — resilient when offline. `pytest>=8` is added to
requirements as a convenience, not a hard dependency. Tests use synthetic data only (no
network); a repo-root `sys.path` insert lets `import screener` resolve when run standalone.

2026-06-19: **Milestone 2 cache = Parquet (prices) + JSON (fundamentals/earnings),
keyed by `symbol + date`.** Files live under `.cache/<namespace>/<symbol>__<date>.<ext>`;
the date in the filename *is* the freshness check (today's file = hit, else refetch),
and a write prunes older-dated files for that key. Chose Parquet over SQLite because
`pyarrow` is already in the stack and price history is naturally a DataFrame — no schema
/ ORM overhead. EOD data only moves once a day, so a per-day key is the right grain.

2026-06-19: **Provider fails soft, per ticker.** `yfinance` is unofficial and
scraping-based, so every fetch is wrapped: a bad ticker yields an empty frame / `None`
and a logged warning instead of aborting a ~500-name universe scan. Only positive results
are cached (an empty fundamentals snapshot or a missing earnings date may be transient or
simply "not announced yet", so we retry next run). Internal symbols are upper-cased to
match `universe.csv`; Yahoo's dashed class-share form (`BRK.B` → `BRK-B`) is applied only
at the network edge.

2026-06-19: **`DataProvider` interface = `price_history` / `fundamentals` /
`earnings_date`** (+ concrete `bulk_*` fan-outs). Prices come back as ~2y of tz-naive
daily OHLCV (lower-cased columns, `date` index) — enough for 12-mo momentum, the 150-day
SMA, 52-wk high, and MACD/EMA warm-up. Batched `yf.download` for cold universe pulls is a
noted later speed optimization; the cache makes warm runs instant regardless.

2026-06-19: **Milestone 1 universe = static S&P 500 list (503 tickers).** Loaded from
`data/universe.csv` (symbol, name, sector) via `screener/universe.py`. Sector ships now
because M4's swing profile needs sector-strength membership. Russell 1000 / market-cap
floor remain options for later (spec §11).

2026-06-19: **Stayed with Python + Streamlit after revisiting Node/TS.** User is fluent
in TS/JS, so we weighed a TypeScript stack (Next.js + React + TanStack Table +
`yahoo-finance2`). Python won for v1: pandas/numpy + yfinance are the ideal fit for the
indicator/data core, and Streamlit makes the table/filter/toggle UI nearly free. Revisit
only if it goes multi-user with a custom frontend — at which point the layered design
makes a FastAPI + React port cheap.

2026-06-19: **Build a dashboard first; natural-language agent layer is later.** A
screener's core job is deterministic filtering/ranking — a dashboard does that fast and
without hallucination. An LLM is the wrong calculation engine; its right role is a later
natural-language layer on top.

2026-06-19: **Equity data = yfinance for the MVP, behind a `DataProvider` interface.**
Free, gives price history (compute momentum/technicals ourselves) and forward P/E for
US names. Unofficial/scraping-based, so cache and validate. Interface lets us swap to a
paid API (e.g. FMP) later without touching the engine.

2026-06-19: **Market = US large-cap equities for v1.** Crypto deferred to v2 (separate
CoinGecko pipeline, no fundamentals).

2026-06-19: **Moving averages are per-profile, not global.** 20/50/150 **SMA** for
long-term & momentum (trend-template structure); 5/9 **EMA** for swing (the 5/9 cross is
the live trigger). The engine computes any SMA(n)/EMA(n); the profile declares which it cares about.

2026-06-19: **Tool ranks/describes, does not advise.** No buy/sell calls — keeps it
useful and honest; the author is not a licensed advisor.
