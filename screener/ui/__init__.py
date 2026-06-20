"""Streamlit UI package for the screener dashboard (behaviour-preserving split).

``app.py`` used to be one ~880-line top-to-bottom Streamlit script. This package
first held that UI relocated VERBATIM into ordered functions; a later header
redesign then promoted the high-frequency controls into a sticky
:mod:`~screener.ui.header` and added a few view/state keys (``view``,
``watchlist``, ``recent_scans``, ``f_in_watchlist_only``). ``app.py`` is a thin
entrypoint that calls these pieces in the SAME relative order the statements ran in.

Module map (call order, top to bottom of a run):

- :mod:`screener.ui.secrets_bridge` — ``bridge_secrets_to_env`` (the
  ``.streamlit/secrets.toml`` → ``os.environ`` copy). Runs before anything reads
  provider availability.
- :mod:`screener.ui.caching` — the five ``@st.cache_data`` memos (the single
  engine memo ``run_cached`` plus the four per-symbol detail memos). Their lazy
  imports stay INSIDE the functions.
- :mod:`screener.ui.nl_state` — natural-language staging machinery: the
  ``_pending_*`` apply loop (+ the ``n_names`` clamp) that MUST run before the
  sidebar widgets exist, plus the Interpret handler.
- :mod:`screener.ui.grid` — the AgGrid results table (all ``JsCode`` constants +
  column configuration + ``render_results_grid``).
- :mod:`screener.ui.header` — ``render_header``: the sticky top toolbar (view nav,
  search, profile, density, ➕ add-ticker + ⚙ Settings popovers, ★ watchlist + 🕘
  recent). OWNS the nav / profile / size / provider widget keys and RETURNS the three
  action flags ``(interpret_clicked, run_clicked, clear_clicked)``. Renders before the
  sidebar so the apply-before-render ordering still holds.
- :mod:`screener.ui.sidebar` — ``render_sidebar``: now renders the post-scan Filters
  block ONLY and returns ``None`` (the action flags moved to ``header``).
- :mod:`screener.ui.scan` — ``run_scan_if_requested``: the ONE engine call site
  (it is the only caller of ``caching.run_cached``); ends in ``st.rerun()``.
- :mod:`screener.ui.transparency` — ``render_nl_banner`` (the interpreted-request
  info banner).
- :mod:`screener.ui.detail_panel` — the post-symbol detail rendering (company
  header card, fit score, links, rank explanation, radar, reasons table, and the
  four TA detail sections).
- :mod:`screener.ui.results_view` — ``render_pre_scan``, ``render_results``
  (table + selection reconciliation), and ``render_state_switch`` (the
  four-state main switch) — the Screener view.
- :mod:`screener.ui.heatmap_panel` — ``render_heatmap_panel`` (the sector-treemap
  view), and :mod:`screener.ui.events_panel` — ``render_events_panel`` (the
  economic-calendar view). The ``view`` nav selects which of the three body views
  renders; only one renders per run. All read the cached scan; none call the engine.
"""
