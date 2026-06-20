# DECISIONS.md — Decision Log

Newest first. Each entry: the decision, and *why*, so nothing gets re-argued later.

---

2026-06-20: **Chart-pattern technical analysis (DONE) — a pure, descriptive per-ticker shape READOUT, EOD only.**
`screener/patterns.py` detects common shapes from price geometry and surfaces them in the inspected-ticker detail
panel; it describes what the chart shows, it never advises. The non-obvious choices:
- **Pure + offline, like the rest of the engine.** patterns.py imports only pandas/numpy/stdlib (NEVER streamlit /
  yfinance / network); `tests/test_patterns.py` (36 offline tests) drives canonical synthetic shapes + anti-shapes.
  The foundation is a swing-pivot/zigzag detector (per-timeframe fractal window + a min-move threshold, strictly
  alternating highs/lows, no lookahead beyond available bars); each pattern detector reads the pivot sequence +
  price arrays. Covered: head & shoulders (+ inverse), double top/bottom, cup & handle, ascending/descending/
  symmetric triangles, rising/falling wedges. Frozen `Pattern` dataclass → picklable so app.py can `st.cache_data` it.
- **EOD timeframes ONLY — 1w / 1d / 1mo, all resampled from the daily bars** (OHLCV-correct: open=first/high=max/
  low=min/close=last/volume=sum; right-labeled, no future-peek). NO intraday / NO 4h — preserves the locked
  end-of-day decision (spec §9) and the date-keyed cache (the user chose EOD-only over a 4h intraday path).
- **Precision over recall, with an honest residual limitation.** Detectors have hard geometric gates + a
  volatility-relative amplitude gate (a shape must clear a multiple of the span's realized volatility, so random
  wiggles don't qualify) + a 0.45 confidence floor + mutual-exclusion de-dup (an overlapping reversal and a
  triangle/wedge/cup container don't co-emit contradictory directions — the multi-pivot container wins). Anti-shapes
  (flat / monotonic / parallel non-converging channel) stay silent. DOCUMENTED residual: textbook reversal shapes —
  especially head & shoulders — genuinely occur on random walks, and no gate fully separates them without killing
  the canonical positives, so reversal-on-noise can appear; the feature is framed descriptively (the panel caption
  says these are mechanically-detected shapes, not signals to act on).
- **On-demand, never part of the universe scan.** Patterns are computed for the ONE inspected symbol via a
  module-level `@st.cache_data` helper keyed on `(symbol, cache_day)` (a warm read of that symbol's already-cached
  daily bars); the engine's single call site and the cold-scan guard are untouched. Built via a design → implement →
  3-lens review (detection / robustness / integration) workflow + a focused fix pass (de-dup contradiction +
  volatility gates + dead code). A **dashboard help/explanations UI** (per-signal descriptions, score/percentile/
  contribution tooltips, a "How to read this" glossary) shipped concurrently in `screener/display.py` and is
  committed alongside.

2026-06-20: **Natural-language agent layer (DONE) — an offline-first translation layer ON TOP of the engine,
never a replacement.** `screener/agent.py` maps a plain-English query to the same screen knobs the sidebar
exposes (profile, universe size, sectors, score floor, symbol filter, swing earnings toggle) plus a one-line
explanation; the deterministic engine still does the actual screening. The non-obvious choices:
- **The engine stays the source of truth — the LLM only fills a typed form.** A single `validate_request` SAFETY
  LAYER clamps/coerces EVERYTHING (profile → a valid registry name, n_names → 5..503, min_score → [0,1] with a
  NaN/inf guard, sectors → canonicalized against the live universe with unknowns dropped, earnings_only → forced
  False unless profile is swing) and NEVER raises. BOTH the LLM tool-call dict and the rule-based dict flow through
  it, so a hallucinated or odd parameter can at worst pick a bounded-but-valid setting — it can't break, bypass a
  filter, or mislead the engine.
- **Offline-first; anthropic is an OPTIONAL dependency.** The module imports only stdlib at top, NEVER imports
  streamlit, and LAZY-imports anthropic strictly inside the LLM call, so the app boots and the 20 offline tests run
  with no key and no SDK installed. With `ANTHROPIC_API_KEY` + the SDK it uses Claude (`claude-opus-4-8` default,
  overridable via `SCREENER_AGENT_MODEL`) via STRICT FORCED TOOL USE (a `set_screen` tool: every property required,
  `additionalProperties` false, NO numeric min/max — clamping lives only in `validate_request`); on ANY failure it
  degrades to the deterministic rule-based keyword parser. The system prompt constrains Claude to parameter
  extraction only — never buy/sell/hold advice (preserves "describes, does not advise" and the dashboard-first,
  agent-later decisions).
- **UI wiring preserves the cold-scan guard.** A sidebar "Interpret & run" box is a SECOND trigger into the ONE
  engine call site (`do_scan = run_clicked or <NL flag>`) via a two-phase stage → rerun → apply that seeds the
  existing widget keys without Streamlit's default-vs-session_state warning; the four states and the disclaimer in
  every state are untouched; a manual Run scan clears the stale NL banner. Built via a design → implement →
  3-lens-review (correctness / security / integration) workflow; review caught a `validate_request` OverflowError
  on a non-finite n_names and a pre-clamp explanation mismatch — both fixed before commit.

2026-06-20: **Milestone 6 (validate + polish, DONE) + post-MVP roadmap reset.** Validation confirmed
the engine/provider are already fail-soft end-to-end across every enumerated edge case (missing/partial
fundamentals, empty/bad ticker, thin/ZERO volume, all-NaN signal column, single/duplicate-symbol
universe, all-filtered or wholly-empty universe), and the numbers reconcile (offline hand-checks of
SMA / momentum / Wilder-RSI / rel-volume / the full percentile score and the contributions-sum-to-score
invariant, plus a live AAPL spot-check). Shipped `README.md` (setup / run / the 3 profiles / how
ranking works + a **refresh-cadence** section: run one scan per trading day in the morning, tied to the
date-keyed cache and `as_of = today`) and `tests/test_edge_cases.py` (13 offline tests; 141 total).
**One quality wart kept BY DESIGN, not "fixed":** a recent IPO sitting at its short available high gets
`dist_52w_high = 0.0` (the best value) via the documented `min_periods=1` exception, so it can top the
percentile on long_term/momentum despite thin history (every *other* short-history signal neutralizes to
0.5). A min-bars guard was REJECTED — it would reverse the locked M3 `distance_from_high` convention and
break two existing indicator tests — so M6 instead DOCUMENTED it (README "Known limitations") and
REGRESSION-PINNED current behavior (`test_ipo_outranks_established_name_on_dist_52w_high` is the canary
for any future guard). **Roadmap reset (user, 2026-06-20):** the v2 **crypto / live asset-class toggle
is DEPRIORITIZED** ("don't do crypto") and stays a disabled stub; **next is the natural-language agent
layer**, built **offline-first** (Anthropic Claude when `ANTHROPIC_API_KEY` is present, else a
deterministic rule-based parser) as a layer ON TOP of the engine — not a replacement, preserving the
locked "dashboard-first, agent-later, describes-not-advises" decisions; **then descriptive chart-pattern
detection** (wedges, head & shoulders, cup & handle, triangles, double top/bottom, flags) via
swing-pivot + geometric rules, **EOD timeframes ONLY — 1w / 1d / 1mo (resampled from daily); explicitly
NO intraday / 4h**, to keep the end-of-day decision (spec §9) and the date-keyed cache intact. v3
statistical/forecasting ML stays deferred (backtest harness first).

2026-06-20: **Milestone 5 dashboard (DONE) — thin `app.py` over a pure `screener/display.py`,
with a hard cold-scan guard and a select-then-explain "why it ranks" panel.** The Streamlit layer is
deliberately split so the project's "everything is unit-tested offline" ethos survives the UI: `app.py`
owns ONLY the widgets + `st.session_state` orchestration; every testable piece (the four-filter
pipeline, table-column selection, the `reasons`→tidy-frame builder, value formatting, the earnings
badge/summary, selection reconciliation, all messages, and a pure column-config *descriptor* dict)
lives in `screener/display.py`, which imports pandas/numpy/`Profile` and **never streamlit** — covered
by 24 framework-agnostic tests in `tests/test_display.py` (same plain-`assert` + `__main__`-runner style
as the M3/M4 suites; no pytest/yfinance/streamlit import). The non-obvious choices:
- **Cold-scan guard (the central risk).** A full 503-name scan is ~503×3 yfinance calls / many minutes,
  and the on-disk cache is only day-fresh, so the engine is invoked at **exactly one site** — inside
  `if run_clicked:` behind a "Run scan" button — with a **small default universe slice** (slider
  default 25, max = full universe) and an `@st.cache_data` memo keyed on
  `(profile_name, n_names, cache_day=date.today().isoformat())`. The day in the key aligns the memo
  with cache.py's date-keyed disk cache and `run_screen`'s `as_of=today` default (don't drop it, or a
  stale cross-day result gets served). Sorting (native `st.dataframe` header-click) and the four sidebar
  filters operate purely in pandas on the cached frame and **never re-run the engine** (verified by a
  headless `AppTest`: 0 engine calls on load, 1 on Run, 0 on filter/sort, 0 on identical re-Run).
- **"Why it ranks" = a select-then-explain detail panel below the table, NOT a literal per-row expander**
  (PLAN.md said "expander"): `st.dataframe` has no per-row expander, so a row is chosen via dual input
  (native single-row `on_select` click + a deterministic selectbox, reconciled to one symbol in
  session_state, validated against the current filtered view so a filtered-out selection can't crash a
  keyed selectbox) and its `reasons` render as value/percentile/contribution per signal with a
  contributions-sum-to-score caption.
- **Percent columns use the `st.column_config` `"percent"` PRESET, which multiplies the engine's
  FRACTION by 100 for display** (0.12 → "12.00%"); the pure descriptor carries `format="percent"` and a
  test pins it — a printf `"%.1f%%"` would silently render the raw 0.12 and drop the ×100 (a review lens
  wrongly flagged a 100× bug here; the installed `NumberColumn` docstring example `1234.567 → 123456.70%`
  settled it). All earnings UI is gated on `profile.flags` (swing-only), not column presence (the column
  exists for every profile in a non-empty result). Four mutually-exclusive states
  (PRE_SCAN / ENGINE_EMPTY / FILTERED_EMPTY / RESULTS); the not-advice disclaimer lives in the sidebar
  so it shows in all four. Built via a design-panel → synthesize → implement → 3-lens adversarial-review
  workflow; the review also caught the now-deprecated `use_container_width` (switched to `width="stretch"`).

2026-06-19: **Deep learning / ML (incl. the Temporal Fusion Transformer, TFT) is DEFERRED to a
post-MVP research track (PLAN.md v3); when it lands it enters as a *signal feeding the
deterministic percentile ranker*, never as the ranking/forecasting engine itself.** Raised by a
"should we use TFT?" question. TFT is a multi-horizon *forecasting* model — and predicting future
return is a buy/sell call in disguise, which collides with the locked "ranks/describes, does not
advise" decision (legal: the author isn't a licensed advisor) and with the explainable
"Nth-percentile on signal X" reason breakdown (spec §8) that is the product's whole value. It is
also premature and ill-fit *now*: M4's ranker isn't finished and there is no backtest harness to
measure a model against; ~503 names × ~2y of daily OHLCV is far too thin for a parameter-hungry
transformer (it will overfit in-sample) and equity series are low signal-to-noise / non-stationary
(naive DL rarely beats a simple baseline out-of-sample); responsible use needs walk-forward /
purged CV, look-ahead guards, and transaction-cost modelling — its own project. The sanctioned path
when the time comes: (1) build the **backtest harness FIRST**; (2) try **gradient-boosted trees**
(LightGBM/XGBoost) over the existing `snapshot()` features — robust on small tabular data, cheap,
and the feature-importances fit the "why it ranks" story; (3) escalate to a sequence model like TFT
**only if** the simple model shows real out-of-sample edge. Whatever the model, its output becomes
one more `SignalSpec` (e.g. `ml_forward_return_rank`) that the existing scorer weights — so the
percentile ranker stays the backbone, stays swappable, and the model stays toggle-off — and it is
surfaced descriptively ("model rates this setup top-decile vs peers"), not as a prediction.

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
