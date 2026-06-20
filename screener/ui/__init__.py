"""Streamlit UI package for the screener dashboard (behaviour-preserving split).

``app.py`` used to be one ~880-line top-to-bottom Streamlit script. This package
holds that exact UI, relocated VERBATIM into ordered functions — no logic was
changed, nothing was renamed, and no new ``st.session_state`` keys were added.
``app.py`` is now a thin entrypoint that calls these pieces in the SAME relative
order the original statements ran in.

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
- :mod:`screener.ui.sidebar` — ``render_sidebar`` (the whole ``with st.sidebar``
  block); returns the action flags the downstream handlers consume.
- :mod:`screener.ui.scan` — ``run_scan_if_requested``: the ONE engine call site
  (it is the only caller of ``caching.run_cached``); ends in ``st.rerun()``.
- :mod:`screener.ui.transparency` — ``render_nl_banner`` (the interpreted-request
  info banner).
- :mod:`screener.ui.detail_panel` — the post-symbol detail rendering (company
  header card, fit score, links, rank explanation, radar, reasons table, and the
  four TA detail sections).
- :mod:`screener.ui.results_view` — ``render_pre_scan``, ``render_results``
  (table + selection reconciliation), and ``render_state_switch`` (the
  four-state main switch).
"""
