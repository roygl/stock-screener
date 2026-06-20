# DECISIONS.md — Decision Log

Newest first. Each entry: the decision, and *why*, so nothing gets re-argued later.

---

2026-06-20: **Dedicated "Add a ticker" control + non-silent NL auto-add.** Branch `feat/economic-event-calendar`.
Adding a ticker used to be possible ONLY by typing it into the natural-language box, which (a) recognised an
UPPER-CASE token only — a lowercase `pltr` or a dotted `BRK.B` silently did nothing — and (b) reported nothing
on success OR failure, so a user couldn't tell whether the row saved or why it didn't (`append_symbol` was
correct; the failure was UX). Also, the visible "~N names" caption is the *scan size* (`n_names`), not the
universe total, so even a successful add looked like "nothing changed". Decisions:
- **A first-class `screener/ui/add_ticker.py` sidebar control** (under the universe slider): strip/upper the
  input (lowercase + class-suffix tolerant), validate SHAPE against the writer's own
  `screener.universe._TICKER_RE` (single source of truth — the gate can't drift from `ensure_symbol`), then
  delegate the fetch + CSV write to `ensure_symbol` (NO duplicate persistence). It ALWAYS reports the outcome —
  `✓ Added PLTR — universe now N names`, "already in the universe", "doesn't look like a ticker", or "Yahoo
  returned no data" — feedback stashed in `st.session_state` so it survives the post-add `st.rerun()`.
- **Bump `n_names` to the new size only when the slider sat at the prior max** ("scan all"), staged through the
  existing `_pending_n_names` channel (setting the widget key directly would raise — its widget already rendered
  above). This makes the visible count grow AND puts the freshly-appended (last-ranked) row inside the next
  scan's head slice, so a just-added ticker is actually scannable. A non-max selection is left untouched.
- **The NL box is no longer silent and no longer upper-case-only.** `handle_interpret` recovers a lowercase
  bare ticker the case-sensitive parser skips (`_bare_ticker_candidate`: a single token, gated by the canonical
  shape + the agent stopword list, so prose/multi-word queries stay unaffected), normalises `req.text` to the
  canonical symbol when present/added so the filter + `include_symbol` surface it, and reports the result via
  `nl_add_msg` in the transparency banner (persistent, not the one-shot sidebar slot, since the NL flow reruns
  again for the scan). A lowercase token that fetches nothing stays QUIET (it may just be a non-ticker word — no
  spurious error, no filter regression); only an unambiguous upper-case ticker that fails earns an error.
- **Cross-process staleness is expected, not a bug:** `load_universe` is an in-process `@lru_cache`, so a ticker
  added in one running session isn't visible to a second session until that one reruns an add (which calls
  `cache_clear`) or restarts. Same-process adds always reflect immediately.

2026-06-20: **In-app economic-event calendar (Milestone A).** Branch `feat/economic-event-calendar`.
Surface upcoming high-impact US macro events (FOMC rate decisions, CPI, the monthly jobs report) plus the
per-ticker earnings the app already fetches, with "days until" countdowns, a 3-tier impact tag, and an
advance-warning flag — framed as *heightened expected volatility*, never advice. Decisions:
- **Bundled, public-domain CSV is the runtime source of truth** (`data/economic_events.csv`, cols
  `date,time_et,event,category,impact,tentative`). US government release dates are non-copyrightable facts
  (17 U.S.C. §105) and every third-party calendar API paywalls/restricts redistribution, so the dates are
  committed to the repo and read by the pure `screener/calendar.py` loader (`@lru_cache`, fail-soft — a
  missing/garbage CSV yields an empty frame, never a crash). Nothing is fetched at runtime.
- **Pull-based, not an alerting service.** Streamlit Community Cloud sleeps when idle and has no cron, so a
  date "alert" cannot fire server-side. The calendar is rendered when the user opens the app (an "Upcoming
  events" expander near the top of the main area), memoized per `cache_day` via `ui/caching.events_upcoming`
  (same date-keyed warm-read model as the per-symbol TA memos). A high-impact-only toggle (default on) follows
  the Forex Factory / Investing.com convention.
- **Manual ~annual refresh** via `scripts/refresh_economic_calendar.py` (local-only: `bls.gov` blocks
  datacenter IPs with an Akamai 403, so a live fetch would 403 on Cloud too — the committed CSV avoids it).
  FOMC dates carry a `tentative` flag (the Fed's own qualifier) surfaced as a UI caption.
- **Generalized the per-symbol event-risk badge** from swing-only to ANY profile: the engine already sets
  signed `days_to_earnings` on every row, so the detail panel reconstructs the next earnings date (no refetch)
  and folds it through `calendar.next_event_for_symbol`, showing a ⚠ badge inside the warning window.
- **Educational framing reused verbatim:** the panel carries the existing `display.DISCLAIMER_TEXT`; events are
  described as windows of heightened expected volatility, no buy/sell language (FINRA educational safe harbor).
- **Purity boundary held:** all date/tier logic lives in pure `screener/calendar.py`; rendering lives in the
  `screener/ui/` layer (new `ui/events_panel.py`), matching the Stage-5 split. `app.py` stays a thin entrypoint.

2026-06-20: **Stage-5 refactor + universe expansion + all-stocks/Gemini defaults + browser-remembered choices + NL ticker auto-fetch.**
Branch `feat/stage5-universe-gemini-persistence`. Six stages, each its own commit and verified between (313 → 334 tests;
app browser-smoke + localStorage round-trip + live yfinance fetch). Decisions:
- **Did the deferred Stage-5 split (behavior-preserving, move-don't-rewrite).** `screener/display.py` (1427 ln) →
  `screener/display/` package (`formatting`/`features`/`tables`/`reasons`/`tactical`/`radar`/`text` + `_base`); `__init__`
  re-exports the full 58-name public API so every `display.X` call site is unchanged. `app.py` (879 ln) → a 68-line thin
  ordered entrypoint + `screener/ui/` package (`secrets_bridge`/`caching`/`nl_state`/`grid`/`sidebar`/`scan`/`detail_panel`/
  `results_view`/`transparency`/`persistence`). UI package lives **under `screener/`** (the sole import root; no top-level
  `ui/`) so no packaging change. Streamlit's strict top-to-bottom order, the single engine call site, and every
  `session_state` key preserved exactly. Also deleted the dead duplicate top-level `universe.py`.
- **Universe 503 → 618** (+115 curated popular non-S&P names: AI infra, semis, nuclear/SMR, crypto-treasury, quantum,
  space, fintech, China ADRs, meme), incl. the requested CRWV/NBIS/OKLO/BMNR/MSTR (TSLA was already in). Each validated
  via yfinance; sectors mapped to the CSV's **GICS** taxonomy — the engine prefers the live yfinance sector at scan time,
  but the CSV seeds the sidebar sector filter + the NL sector enum, so new rows must match the existing GICS strings.
- **Default universe size = ALL names** (was 25) and **default LLM = Gemini** (`gemini-2.5-flash`; was Anthropic). Both
  still degrade gracefully — the slider dials down for a faster cold scan; no key → the offline rule-based parser.
- **Remember sidebar choices in the browser via localStorage** (`streamlit-local-storage`): profile, engine, universe
  size, and table density survive refresh / new tab / browser restart. Chose localStorage over `st.query_params` (the
  user asked for "browser cache" — clean URLs, cross-tab). Values are validated/clamped on read so a stale store can't
  crash a keyed widget; an explicit NL Interpret still wins for that run; degrades to non-persistent if the component is
  absent. (The Engine radio moved from `index=` to the initialize-then-no-default seed so persistence can pre-seed it.)
- **NL free-text auto-fetches an unknown ticker.** If a query names a ticker not in the universe, the app fetches it via
  the data provider, appends it to `data/universe.csv` (GICS-mapped sector), and **unions it into the scan** via a scoped
  `include_symbol` (only ever an exact universe ticker) so filtering never re-runs the engine. `agent.py` stays
  network-free — the fetch lives in the UI layer (`ui/nl_state` → `universe.ensure_symbol`), which never raises. NOTE: on
  Streamlit Cloud's ephemeral filesystem these runtime appends are lost on restart; durable additions need the CSV committed.

2026-06-20: **Declutter the UI, surface price/ATR/company info, and make Gemini NL search reliable.** The user found
the results table over-stacked (19–23 flat columns), couldn't see the current **price**, **ATR**, or **what the
company does**, and had a valid Gemini key that silently fell back to the offline parser. Shipped on branch
`feat/declutter-surface-gemini` as four behavior-additive stages (the planned Stage-5 file-split refactor was
**deferred** to avoid merge conflicts with concurrent work — it adds no user-facing value). Decisions:
- **Surfaced the headline data we already had access to:** current price, daily change %, ATR(14) + ATR%, market
  cap, industry, and the long business summary. Plumbed additively through provider → `indicators.snapshot()` →
  engine, all fail-soft to `NaN`/`None`; `business_summary` stays OUT of the grid (detail-panel only).
- **Compact-by-default table + a Compact/Detailed density toggle** (research-backed: Finviz/TradingView lead with a
  lean view + progressive disclosure). Compact = identity + Fit + price + signed green/red daily-change + the two
  tactical readouts; Detailed reveals the profile's signal columns. Symbol column **pinned left**. `column_order` is
  now the single source of truth for the tactical columns (previously appended in `app.py`); the grid is keyed to
  its column SET so st-aggrid's client-side column-order persistence can't carry one density's order into the other.
  Pure-vs-Streamlit boundary preserved (`display.py` still streamlit-free, unit-tested).
- **Company header card** in the detail panel: a 4-metric row (Price / Daily change / ATR / Market cap) + an
  Industry caption + a "What the company does" expander, each shown only when present. New pure formatters
  `format_price` / `format_market_cap` (human units $1.2T/$345.0B).
- **Gemini self-diagnosing, env/secrets-only (no in-app key box).** Accept `GOOGLE_API_KEY` as an alias for
  `GEMINI_API_KEY`; add `agent.availability_status() -> (ok, reason)`; `parse_query` folds the fallback reason into
  the explanation so the NL banner says WHY it degraded. `app.py` bridges `.streamlit/secrets.toml` → `os.environ`
  at startup (env wins) so a key works regardless of launch context, with a live ✓/✗ status line. **NL schema
  unchanged** — only made the existing coarse-knob path reliable + visible. Added `.streamlit/secrets.toml.example`.

2026-06-20: **Deploy host = Streamlit Community Cloud (free), NOT Cloudflare.** The user asked to "upload to
Cloudflare," but the app is a long-running Streamlit (Tornado + WebSocket) server with native deps
(`pandas`/`numpy`/`pyarrow`/`yfinance`), which **cannot** run on Cloudflare Workers or Pages — the only CF path is
**Containers** (a Worker + Durable Object fronting a Docker image), which requires a Workers **Paid** plan (~$5/mo).
After comparing hosts we chose **Streamlit Community Cloud**: free, purpose-built, deploys `app.py` straight from
GitHub (`roygl/stock-screener`, branch `main`) off `requirements.txt`, no Dockerfile. The non-obvious bits:
- **Deploy-ready as-is, zero repo changes.** Pure-pip deps ⇒ no `packages.txt`; the `requirements.txt` `>=` floors
  resolve cleanly on Cloud's newer Python (recommend pinning **Python 3.12** in Advanced settings); `.gitignore`
  already excludes `.venv/` / `.cache/` / `*.zip` / `.streamlit/secrets.toml`.
- **No required secrets — offline-first holds.** With no key the agent degrades to the rule-based parser, so the
  core screener runs unauthenticated. The optional LLM agent needs its provider key set as a **TOP-LEVEL** key in
  the Cloud secrets manager: Streamlit exports only root-level secrets as env vars (which the `anthropic`/`openai`
  SDKs read) — keys nested under a `[section]` are NOT exported and would silently leave the agent in rule-based mode.
- **RAM fallback = Hugging Face Spaces** (free, ~16 GB) if a full-universe scan ever OOMs the ~1 GB Community tier;
  same repo, no code changes. Runbook lives in `DEPLOY.md`.

2026-06-20: **"Surface what we compute" — fit-score + per-row narrative + signal radar + CSV, and a DOUBLE-CLICK
st-aggrid results table.** A presentation upgrade (off a competitive read: Finviz/TradingView/Stock Rover/ChartMill/
Simply Wall St/Danelfin/…) — we already compute a composite score + a per-signal percentile breakdown but presented
them like a spreadsheet, so this turns that into the visual idioms the market expects, plus the user's explicit ask
for a double-click row selector. The non-obvious choices:
- **All surfacing logic stays PURE in `screener/display.py`** (still streamlit-free, offline-tested): `fit_score`
  (0..1 score → 0..100 int), `narrative`/`narrative_series` (the per-row "Strongest on X … weakest on Y" clause,
  factored out of `explain_rank` via a shared `_highlight_clause` so the inspector summary stays byte-identical),
  `radar_spec` + `radar_svg` (a self-contained **pure-SVG snowflake STRING** — one axis per signal, value = the
  signal's percentile; zero new runtime dep, fully unit-testable), and `export_frame` (CSV of the filtered view).
  The synthetic `fit` and `why` columns ride the existing derived-column seam (`_with_link_columns` →
  `_with_derived_columns` + `column_order`/`column_config_spec`); the visible table now leads with **Fit (0..100)**
  in place of the raw 0..1 `score` (the `score` descriptor is retained for the still-0..1 min-score filter). Detail
  panel: a "Fit score N / 100" hero replaces the 0.xxx metric, and the radar renders ABOVE the reasons table via
  `st.components.v1.html` (an isolated iframe, so the SVG uses explicit theme-robust colours, not host CSS vars).
  `tests/test_display.py` +9 cases (302 total green).
- **DOUBLE-CLICK selection ⇒ st-aggrid (a new dep), because native `st.dataframe` is single-click only.** The main
  results table is now an `AgGrid` (`suppressRowClickSelection=True` + an `onRowDoubleClicked` JsCode
  `setSelected(true)`, `update_mode=SELECTION_CHANGED` so sort/filter do NOT rerun the engine — the cold-scan guard
  holds); the inspect selectbox stays as a fallback and `resolve_selection` is unchanged. The pure
  `column_config_spec` descriptor is realised as AgGrid colDefs in app.py (the new purity boundary, replacing
  `_build_column_config`). Three st-aggrid 1.0.5 gotchas, all caught by a live browser smoke-test and fixed:
  (1) it defaults to **AG Grid Enterprise** (trial WATERMARK) → `enable_enterprise_modules=False` (Community);
  (2) this AG Grid React build accepts **neither** an HTML-string cellRenderer (escaped to raw text) **nor** a
  DOM-node one (**React error #31**) → render ONLY via `valueFormatter` (plain text) + `cellStyle` (a plain style
  object); the Fit "bar" is a `cellStyle` background gradient behind the number, never a cellRenderer; (3) so the
  per-ticker external-link columns are **dropped from the grid** (the detail panel already has TradingView/Yahoo
  link buttons) — which also trims the wide table. `streamlit-aggrid>=1.0.5` added to requirements (its setup.py
  over-pins `altair<5`; altair 5 works fine and streamlit 1.50 prefers it — kept altair>=5). Built on a branch off
  the reconciled tree carrying the tactical TA readouts; overextension / support-resistance / buy-zone and the TA
  filters are all preserved.

2026-06-20: **The NL agent's LLM backend is now SWAPPABLE via a provider registry — Anthropic native plus an
OpenAI-compatible family over one optional `openai` SDK.** Generalizes the Anthropic-only agent (`screener/agent.py`)
so the user can pick the engine and so adding backends is a one-line edit. The non-obvious choices:
- **Cheap multi-backend via an OpenAI-compatible base path, not N native SDKs.** A data-driven `PROVIDERS` registry
  (frozen `Provider` dataclass: `id/label/kind/env_key/default_model/base_url/model_env`) holds 6 entries —
  `anthropic` (kind `anthropic`, native SDK) + `openai`, `xai` (Grok), `gemini`, `mistral`, `ollama` (all kind
  `openai`). The five OpenAI-compatible ones ride the SINGLE `openai` SDK and differ ONLY by `base_url` / env-var /
  model id, so the roster is **6 backends from just 2 optional deps** (`anthropic` already present + new `openai>=1.40`)
  and adding/dropping one is a single registry edit. Selection is a single-select sidebar **radio** ("Engine"),
  defaulting to `anthropic` so today's behavior is byte-for-byte preserved.
- **Env-vars-only credentials — no key UI, no session-state secret threading.** Keys come strictly from the
  environment (per-provider `env_key`; Ollama is keyless and special-cased with an `"ollama"` api_key placeholder +
  `SCREENER_OLLAMA_BASE_URL` override); model ids are registry-isolated and overridable via each provider's `model_env`
  or the existing global `SCREENER_AGENT_MODEL`. This keeps the diff small and keeps secrets out of Streamlit
  session state. `_resolve_provider` (explicit > `SCREENER_AGENT_PROVIDER` > default, unknown → default, never raises)
  and `_resolve_model` (explicit > per-provider `model_env` > global > provider default) do the wiring.
- **Offline-first, import-cheap, and the single safety layer are UNCHANGED — the engine stays the source of truth.**
  The module still imports only stdlib at top (`json` added) and NEVER imports streamlit; BOTH SDKs stay lazy *inside*
  their `_llm_extract` branch, and `agent_available(provider=None)` still probes via `importlib.util.find_spec` (no
  import) — so `from screener import agent` works with NEITHER SDK installed and no keys. Both paths emit an identical
  strict `set_screen` contract (shared `_set_screen_schema`: every property required, `additionalProperties` false, NO
  numeric min/max), funnel their raw dict through the one UNCHANGED `validate_request` clamp/coerce layer, and on ANY
  unavailable-provider/error degrade to the deterministic `rule_based_parse`. The transparency prefix is now
  provider-aware (`LLM (openai): …`); the offline path stays `Rule-based: …`. `SYSTEM_PROMPT` (describes, never
  advises) is shared and untouched; the cold-scan guard and single engine call site are not touched.

2026-06-20: **Tactical TA readouts (DONE) — explicit "Buy zone", support & resistance, and overextension/parabolic;
this RELAXES "describe, don't advise" to "no execution + no sell/exit management, but an educational entry zone is now
shown."** The screener now answers three tactical per-ticker questions — where is the buy zone, is the stock parabolic,
where are support & resistance (תמיכה והתנגדות) — alongside the existing momentum/RSI/MACD/MA snapshot and chart-shape
detection. The non-obvious choices:
- **The "Buy zone" is the one deliberate relaxation of the locked stance.** The user chose a real, labeled entry band
  rather than a neutral "support zone", which loosens the "ranks/describes, does not advise" rule (the bullet below; the
  CLAUDE.md key-decision line). We honor it but keep it bounded: a **descriptive** entry band derived from the **nearest
  historical support** (`high = support.price`, `low = support × (1 − cluster tol)`) or, failing that, a **rising-20-EMA
  pullback** band — surfaced **with an educational not-advice disclaimer** ("Educational entry zone, not financial
  advice") pinned by a framing-guard test and shown in both the detail panel and `BUY_ZONE_HELP`. The relaxation is
  STRICTLY scoped: still **no execution / no broker action** and still **NO sell/exit/stop management** — only an entry
  zone is described. A name with no support below current price shows no zone (never an invented one).
- **Support & resistance + overextension are pure descriptive readouts, EOD only.** S/R levels are **pivot-cluster**
  bands (greedy single-linkage clustering of `patterns.find_pivots` swings, volume-weighted centers, ≥2 touches, a 0..1
  strength from touches/recency/tightness, nearest-first, capped per side) in a new pure `screener/levels.py` that reuses
  the chart-pattern swing foundation with zero duplication. Overextension/parabolic lives in `screener/indicators.py`
  (new `true_range`/`atr`/`atr_latest`/`consecutive_up_run` + a frozen `ExtensionState`): a 0..1 score blends % above
  20/50-EMA, RSI, up-day run, acceleration, and ATR%, with a HARD FLOOR that a falling stock (≤0% above its 20-EMA) can
  never read "parabolic". Both honor the locked end-of-day constraint — **1w / 1d / 1mo resampled from daily, NO
  intraday / 4h** (consistent with the chart-pattern decision and the date-keyed cache) — and stay fail-soft (empty/NaN/
  None on degenerate input, never raise); pure pandas/numpy, no scipy.
- **Overextension state + an "in buy zone" flag are also UNIVERSE-WIDE, not just per-ticker.** `extension_state`/
  `extension_score` join `snapshot()`/`SNAPSHOT_KEYS` and `in_buy_zone`/`dist_to_buy_zone_pct` are attached in
  `assemble_features()` off the already-fetched daily frame (daily-only, no extra fetches), so they flow into the ranked
  results table as a colored **Extension** badge + an **In buy zone** flag, with sidebar **filters** ("hide overextended
  (parabolic)", "in buy zone only"). Support/resistance bands stay in the detail panel (bands don't tabulate). The S/R
  bands and buy-zone are surfaced in the inspected-ticker panel in the existing Chart-patterns visual grammar (badges +
  Strength `ProgressColumn` + metrics). Built via the `implement-ta-readouts` multi-agent Workflow (foundation → wiring →
  integration → loop-until-green verify → docs).

2026-06-20: **"Clear cache & rescan" now actually clears + rescans, and results name the hard filters that
narrowed them.** Two fixes prompted by a "rescanning doesn't fetch 500, it stays at 35" report. The 35 was the
**swing** profile working as designed (only ~35 of 500 clear `rel_volume_20 > 2×` AND `in_leading_sector`); the
real defects were in the rescan UX:
- **The button was a no-op-looking lie.** It was `st.button(..., on_click=st.cache_data.clear)`, which (a) only
  dropped the in-memory `@st.cache_data` memo, leaving cache.py's date-keyed on-disk parquet/JSON untouched — so a
  same-day rescan re-read identical files — and (b) never triggered a scan at all (`run_clicked` stays False on its
  rerun). Now it's a captured `clear_clicked` wired as a THIRD trigger into the single engine call site: it
  `Cache().clear()`s the on-disk cache (forcing a fresh Yahoo fetch) **and** `st.cache_data.clear()`s the memo, then
  rescans. Kept the one-call-site / cold-scan-guard invariant — no second engine path. Filters intentionally NOT
  reset on rescan (deferred; the user opted out).
- **Selectivity was invisible.** Added pure `display.selectivity_hint()` (+ `hard_filter_phrases`/`_filter_phrase`,
  threshold-synced so a tuned cutoff re-renders) rendered under the context line: "35 of 500 scanned cleared the
  Swing hard filters (relative volume > 2× and in a top-3 sector by 3-mo return) — … not a data error." So a small
  match count reads as intended selectivity, not a failed/partial fetch. Offline-tested in `tests/test_display.py`.

2026-06-20: **Per-ticker external chart/quote links (DONE) — one-click jump-out from the results table,
still descriptive (no advice).** The results table and the inspected-ticker panel now link each symbol out
to its TradingView chart and Yahoo Finance quote. The non-obvious choices:
- **Pure + offline, like the rest of `display.py`.** Two URL builders (`tradingview_url`, `yahoo_url` —
  stdlib `urllib.parse.quote` only) + `_with_link_columns` derive synthetic `tv_url`/`yf_url` columns from
  `symbol`; they thread through the existing `column_order` → `table_view` → `column_config_spec` seams (a
  new `kind:"link"` descriptor) and are realised as `st.column_config.LinkColumn` at app.py's single purity
  boundary. `tests/test_display.py` covers them offline (incl. class shares).
- **Dedicated link columns, not a linked `symbol` cell.** `symbol` stays the row-select handle that drives
  the "Why it ranks" panel; turning it into a link would hijack click-to-inspect and allow only one
  destination. The link columns sit right after `score` (before the signals) and coexist with row selection
  — clicking a link opens a new tab, clicking elsewhere selects the row.
- **Icon-first / minimal cells.** Each cell shows a single "↗"; the sticky column header (TradingView /
  Yahoo) + Streamlit's native URL-on-hover identify the destination (LinkColumn renders text/emoji only, so
  no inline SVG icon). The detail panel adds short `st.link_button`s (Material icons).
- **Bare symbol, no exchange map (v1).** No exchange field exists in the model; both sites resolve US
  large-cap symbols directly. Class shares differ by separator — TradingView swaps `-`→`.` (BRK-B→BRK.B),
  Yahoo keeps `-`. Exchange-qualified URLs + extra destinations (Finviz/Google Finance) are deferred.

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
