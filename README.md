# stock-screener

A **stock screener** for US large-cap equities. Pick an investing **profile**
(style) and get a ranked, sortable, filterable table of S&P 500 names that match
that style's signals — with a per-row "why it ranks" breakdown. It automates the
*systematic scan / watchlist-building* step; it does not replace charting, broker
execution, or live alerting.

> **Not financial advice.** This tool **describes and ranks** stocks from delayed,
> end-of-day data using mechanical rules. It makes **no buy, sell, or hold
> recommendation**, knows nothing about your goals or risk tolerance, and its
> signals can be wrong or stale. Past behaviour does not predict future returns.
> Do your own research and consult a licensed professional before investing.

---

## Setup

Requires **Python 3.9+**. From the repo root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Dependencies (`requirements.txt`): `streamlit`, `streamlit-aggrid` (results
table), `yfinance`, `pandas`, `numpy`, `pyarrow` (Parquet engine for the price
cache), `streamlit-local-storage` (remembers your sidebar choices in the
browser), `pytest` (an *optional* test runner — the suite also runs as a plain
script), and the *optional* LLM SDKs `anthropic` / `openai` (the latter also
drives the Gemini / xAI / Mistral / Ollama backends) for the natural-language
agent — without a key it degrades to the offline rule-based parser.

## Run the app

```bash
.venv/bin/python -m streamlit run app.py
```

Streamlit opens the dashboard in your browser. Then:

1. In the sidebar, pick a **Profile** and a **Universe size** (names to scan;
   **defaults to all ~618 names** — dial it down for a faster cold scan). Your
   profile, NL engine, universe size, and table density are **remembered in your
   browser** (localStorage), so they persist across refreshes and visits.
2. Press **Run scan**. The engine runs at **exactly one site** — behind this
   button — so the page never auto-runs a slow full-universe scan on a rerun.
3. **Sort** by clicking a table header and **filter** with the sidebar widgets
   (symbol/name, sector, minimum score, and — swing only — earnings-in-window).
   Sorting and filtering act purely on the last scan's result and **never refetch**.
4. Click a row (or use the **Inspect a row** selectbox) to see its
   "why it ranks" breakdown.

**Cold first scan is slow; warm scans are instant.** A cold scan hits Yahoo once
per name across price history, fundamentals, and the earnings date — many calls,
so start small. Results are written to a local **date-keyed cache** (`.cache/`),
so the *second* run the same day (and every sort/filter) is served from disk.
The in-memory `@st.cache_data` memo is keyed on `(profile, size, today)`, so
re-picking the same profile and size the same day is an instant hit. Use
**Clear cache & rescan** to drop the in-memory memo (the on-disk cache still
rolls over on its own each day — see *Refresh cadence* below).

The screener **fails soft**: a ticker that Yahoo can't serve, a recent IPO with
too little history, or a thin/zero-volume name yields blank/neutral signals and
is simply ranked or filtered accordingly — it never crashes a full run.

## Ask in plain English

The sidebar's **Ask in plain English** box turns a request like *"top 20 momentum
tech names, high conviction"* into screen parameters and runs the scan. It maps your
words onto the same knobs the manual controls expose (profile, universe size, sector,
score floor, symbol, the swing earnings toggle) and shows a one-line note of how it
read you — nothing is a black box. The deterministic engine still does the actual
ranking; the agent only fills in the form, and every value is clamped to a safe range.

By default it uses an **offline rule-based parser** (no setup, no network). For richer
interpretation, set `ANTHROPIC_API_KEY` and install the optional `anthropic` package
(`pip install anthropic`); it then uses Claude (`claude-opus-4-8`, overridable via
`SCREENER_AGENT_MODEL`). With no key it silently stays on the rule-based parser, and it
never gives buy/sell advice — it only translates your request into a screen.

## Chart patterns

When you inspect a ranked ticker, the detail panel also shows **mechanically-detected
chart patterns** across the **weekly / daily / monthly** end-of-day timeframes —
head & shoulders (and inverse), double tops/bottoms, cup & handle, ascending /
descending / symmetric triangles, and rising / falling wedges — each with a direction
and a confidence. These are **descriptive geometry, not signals to act on**: detection
is heuristic, and noisy shapes (especially head-and-shoulders on choppy data) can still
appear, so treat them as context. They're computed on demand for the one inspected symbol
from its cached daily bars — no extra universe scan, no intraday data.

## The three profiles

A profile is a **config**, not code: a set of **hard filters** (cutoffs a name
must clear to appear at all) plus **weighted signals** (the inputs it is ranked
on). Same engine, different lens. Defined declaratively in
[`screener/profiles.py`](screener/profiles.py).

| Profile | Hard filters | Ranked on (signals) |
|---|---|---|
| **Long-Term** | positive forward P/E; price above the 150-day SMA | forward P/E (cheaper better), revenue growth, earnings growth, stacked 20 > 50 > 150 SMA, distance from 52-week high, 12-month momentum |
| **Swing** | relative volume > 2×; in a top-3 (leading) sector | fresh 5/9 EMA cross, relative volume, MACD histogram, RSI health (strong but not overbought), 10/20 EMA pullback quality, sector strength — plus an **earnings-in-window badge** (next report within 7 days) |
| **Momentum / Growth** | price above the 50-day SMA | 1 / 3 / 6 / 12-month momentum, stacked 20 > 50 > 150 SMA, relative volume, distance from 52-week high, earnings growth |

Moving averages are **per profile**: 20/50/150 **SMA** for long-term and momentum
(trend structure); 5/9 **EMA** for swing (the cross is the live trigger).

## How ranking works

Scoring is **cross-sectional percentile rank** (see
[`screener/engine.py`](screener/engine.py) `score_and_rank`):

1. For each signal, every candidate is ranked **0–1 across the screened universe**
   (a percentile). "Lower is better" inputs like forward P/E are inverted
   (`1 − pct`) so cheap scores high; banded ideas (RSI overbought, a healthy
   pullback, 5/9-cross freshness) are pre-baked by the engine into higher-is-better
   `[0, 1]` features.
2. Each percentile is multiplied by its **normalized weight** (`weight ÷ sum of
   weights`) and summed into a `score` in `[0, 1]`. A **missing signal scores a
   neutral 0.5** rather than penalizing the name.
3. Rows are sorted by score (ties broken alphabetically) and given a 1-based
   `rank`.

Percentile rank is **outlier-proof** (no single runaway name dominates) and yields
the clean **per-row reasons** the dashboard shows: for each signal, its raw value,
its percentile, and its contribution to the score — and the contributions **sum
exactly to the score**. Sector strength ranks the 11 GICS sectors by median
3-month member return; the swing "leading sector" filter passes only the top 3.

## Known limitations

- **Recent IPOs can over-rank on `dist_52w_high`.** Distance-from-52-week-high is
  computed with `min_periods=1` (a deliberate exception — see the
  [`distance_from_high`](screener/indicators.py) docstring) so a freshly-listed
  name measures distance from its *available* high rather than reading `NaN`. The
  side effect: a recent IPO sitting at its short available high gets
  `dist_52w_high = 0.0`, which is the **best possible** value and therefore the
  **top percentile** on any profile that ranks on it ("higher" in **Long-Term**
  and **Momentum**). Every *other* short-history signal degrades to `NaN` and is
  neutralised to 0.5, so `dist_52w_high` is the lone indicator that turns thin
  history into a confident best-case score. A 30-bar IPO can thus out-rank an
  established name a few percent off its real 52-week high. This is **by design,
  not a crash** — the pipeline stays fail-soft — but sanity-check the history
  length of any unfamiliar top-ranked name. (Pinned by an offline regression test;
  see *Running the offline tests* below.)
- **End-of-day only.** Signals are computed from delayed, finalized daily bars —
  there is no intraday or real-time data (see *Refresh cadence*).
- **Static universe.** The S&P 500 list in `data/universe.csv` is a fixed snapshot;
  index add/drop changes are not tracked automatically.

## Refresh cadence

The data is **end-of-day**, and the cache is **date-keyed** (`Cache._path` stamps
each file `…__YYYY-MM-DD.<ext>`; `run_screen` defaults to `as_of = today`). So:

- A given trading day's bars are final after that day's **post-market close /
  next-day settle** — roughly the **next morning** once Yahoo has posted official
  EOD values.
- The cache day-key **rolls at local midnight**: the first scan after midnight
  misses every key and refetches; scans the rest of that day are served from disk.

**Recommended:** run a fresh scan **once per trading day, in the morning** (before
markets open, e.g. ~8–9am local), so you're scanning on finalized prior-day EOD
data. The first scan of the day pays the cold-fetch cost; everything after is warm.

Use **Clear cache & rescan** only when you need to **force a same-day refetch** —
e.g. you scanned mid-session before EOD values settled, or you suspect a stale /
partial fetch. On a normal next-day cadence you never need it: the day-key rolls
the cache for you.

## Running the offline tests

The test suite is **framework-agnostic and offline** — synthetic data only, no
network, and no required `pytest`/`yfinance`/`streamlit` imports. Run each suite
as a plain script:

```bash
.venv/bin/python tests/test_indicators.py    # 63 tests — indicator math
.venv/bin/python tests/test_engine.py         # 41 tests — filters, scoring, fail-soft
.venv/bin/python tests/test_display.py        # 24 tests — pure display helpers
.venv/bin/python tests/test_edge_cases.py     # edge cases — fail-soft + the IPO tilt
```

The first three (**128** tests) report `N/N passed`; `test_edge_cases.py` pins the
fail-soft edge behavior (missing data, thin/zero volume, recent IPO) and the
documented IPO `dist_52w_high` tilt. All run under `pytest` too if you prefer
(`.venv/bin/python -m pytest tests/`).

> **Writing new offline tests:** build every synthetic price frame on **one shared
> index**. Mixing a plain `RangeIndex` series into a `DatetimeIndex` frame makes
> pandas align by label and silently fill the column with `NaN`, so an indicator
> test can "pass" on all-`NaN` data without ever exercising the math. The helpers
> in the existing suites already share a single `date_range` index per frame —
> follow that pattern.

## Project layout

```
app.py                 Streamlit dashboard (thin wiring; engine called only behind "Run scan")
screener/
  universe.py          load_universe() -> DataFrame[symbol, name, sector] (static S&P 500)
  provider.py          DataProvider interface + YFinanceProvider (fail-soft, date-keyed cache)
  cache.py             local Parquet/JSON cache keyed by (namespace, symbol, date)
  indicators.py        pure technicals + snapshot() -> 21 per-ticker features
  profiles.py          the 3 declarative profiles (filters + weighted signals)
  engine.py            run_screen(): assemble -> sector strength -> filter -> score/rank
  display.py           pure, streamlit-free display helpers for the dashboard
tests/                 offline suites (run standalone or under pytest)
data/universe.csv      the static S&P 500 universe (symbol, name, sector)
```

For the bigger picture, read **[CLAUDE.md](CLAUDE.md)** (project brief),
**[PLAN.md](PLAN.md)** (milestones), and **[DECISIONS.md](DECISIONS.md)**
(decision log, newest first). Scope is fixed in **[spec.md](spec.md)**.

### Roadmap (post-v1)

Equities only in v1. The **natural-language "Interpret & run" box** and **descriptive
chart-pattern detection** (1w / 1d / 1mo — see the sections above) both ship as layers
over the deterministic engine, alongside an in-dashboard **help / glossary** for the
signals. A live crypto pipeline is on hold; universe-wide pattern *screening* (filter/rank
by detected pattern, not just show it per ticker) is a backlog idea. **v3** is a gated ML
research track — a backtest harness first, then a model that enters only as one more
*signal* feeding the percentile ranker, never as the ranking engine itself. The tool
stays descriptive: it ranks and describes, it does not advise.
