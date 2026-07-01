"""The sticky header: high-frequency controls relocated out of the sidebar.

:func:`render_header` owns the top-of-page control surface — the wordmark + global
disclaimer caption (folded in from ``app.py``), the view nav, the dual-purpose
search box, the Interpret / Run actions, an ➕ add-ticker popover, and a ⚙ Settings
popover (engine, asset class, universe size, refresh, disclaimer) — plus a context
row carrying the profile control (on every view) and, on the Screener view, the
table-density control.

It returns the SAME three action flags the sidebar used to
(``interpret_clicked``, ``run_clicked``, ``clear_clicked``); the profile / size the
rest of the script needs are read back out of ``st.session_state`` (the widget keys
this block owns), exactly as before.

Two contained, deliberate exceptions to the "no raw HTML/CSS" rule: the small sticky
``<style>`` block below (the only ``<style>`` the repo injects, mirroring the radar
iframe precedent), targeting the ``st-key-app_header`` container class.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from screener import agent, display
from screener.profiles import PROFILES
from screener.ui import add_ticker


def _mirror_search_to_filter() -> None:
    """on_change for the header search box: mirror its text into the client-side filter.

    Decision D1 — the header search box is dual-purpose. Typing in it should instantly
    narrow the cached table (0 engine calls), so we copy ``nl_query`` into ``f_text``
    (a plain session-state value no widget owns; ``apply_filters`` reads it). The
    Interpret button reads ``nl_query`` separately for the natural-language path.
    """
    st.session_state["f_text"] = st.session_state.get("nl_query", "")


def render_header(universe) -> "tuple[bool, bool, bool]":
    """Render the sticky header; return (interpret_clicked, run_clicked, clear_clicked)."""
    # --- sticky header CSS (Decision D7) ---------------------------------
    # Deliberate, contained exception: the ONLY <style> the repo injects (like the
    # radar iframe). Targets the DOM class Streamlit emits for st.container(key=...)
    # so the whole header bar pins to the top on scroll. Minimal on purpose.
    st.markdown(
        """
        <style>
        /* Pin the header to the top so the controls stay reachable while the results
           table scrolls. Streamlit wraps a keyed st.container in a shrink-to-fit
           [data-testid="stLayoutWrapper"], so the `position: sticky` must sit on THAT
           wrapper — its containing block is the tall main vertical block, which gives
           the header room to stay pinned for the whole scroll. Sticky on the inner
           .st-key-app_header alone fails: its wrapper parent is only as tall as the
           header, so there is nothing to stick within (verified in the running app).
           The :has() parent-match scopes it to our header's wrapper only. */
        [data-testid="stLayoutWrapper"]:has(> .st-key-app_header) {
            position: sticky;
            top: 0;
            z-index: 999;
        }
        /* The repo pins no theme in .streamlit/config.toml, so the default theme
           follows the system colour scheme — match it with a prefers-color-scheme pair
           (Streamlit's default light bg is #fff, dark bg is #0e1117) so the bar is
           OPAQUE; a transparent sticky bar would let scrolled rows bleed through. */
        .st-key-app_header {
            background-color: rgb(255, 255, 255);
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
            padding-top: 0.25rem;
            padding-bottom: 0.5rem;
        }
        @media (prefers-color-scheme: dark) {
            .st-key-app_header { background-color: rgb(14, 17, 23); }
        }
        /* --- Mobile (Decision D8) -------------------------------------------
           On a phone Streamlit stacks every column in the header vertically, so
           the control surface (search, Interpret, Run, profile bar, density /
           watchlist / recent row, captions) grew taller than the viewport and
           pushed the results table fully below the fold. Worse, a `position:
           sticky` header that is itself taller than the screen leaves no room to
           scroll the table "under" it and the scroll appears to fight back.
           So on narrow screens we (a) DROP the sticky pin — the header scrolls
           away normally, freeing the whole viewport for the table — and (b)
           tighten the vertical rhythm so the table is reachable in one short
           swipe. Desktop is untouched (the pin + spacing above still apply). */
        @media (max-width: 640px) {
            [data-testid="stLayoutWrapper"]:has(> .st-key-app_header) {
                position: static;
            }
            .st-key-app_header { padding-bottom: 0.25rem; }
            /* Shrink the gaps Streamlit puts between the stacked rows/widgets so
               the header isn't a full screen of whitespace on mobile. */
            .st-key-app_header [data-testid="stVerticalBlock"] { gap: 0.4rem; }
            .st-key-app_header [data-testid="stCaptionContainer"] { margin: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- "/"-to-focus the search box -------------------------------------
    # A tiny zero-height iframe (mirrors the radar-iframe precedent) whose script
    # listens on the PARENT document — components.html runs sandboxed in an iframe, so
    # the search <input> lives in window.parent. Pressing "/" while NOT already typing
    # focuses the search box (placeholder match) and swallows the keystroke. Wrapped in
    # try/catch so a cross-origin parent (it isn't, here) degrades to a silent no-op.
    components.html(
        """
        <script>
        (function () {
          try {
            var doc = window.parent.document;
            if (doc.__screenerSlashBound) return;   // bind once per page, not per rerun
            doc.__screenerSlashBound = true;
            doc.addEventListener('keydown', function (e) {
              if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
              var el = doc.activeElement;
              var tag = el && el.tagName ? el.tagName.toLowerCase() : '';
              if (tag === 'input' || tag === 'textarea' || tag === 'select' ||
                  (el && el.isContentEditable)) return;
              var box = doc.querySelector('input[placeholder*="describe a screen"]') ||
                        doc.querySelector('input[aria-label="Search"]');
              if (box) { e.preventDefault(); box.focus(); }
            });
          } catch (err) { /* cross-origin or no parent — no-op */ }
        })();
        </script>
        """,
        height=0,
    )

    # --- defensive seeding (Decision D4) ---------------------------------
    # setdefault is a no-op if the key already holds a user/remembered/staged value,
    # so every scan-input + nav key is DEFINED on every run regardless of which view
    # renders or whether a popover is opened (Architecture invariant #3). Every
    # relocated widget below then uses the initialize-then-no-default pattern (pass
    # key=, no default=/index=/value=) so Streamlit reads from session_state and never
    # emits the "default value but also set via Session State API" warning.
    universe_len = len(universe)
    slider_min = min(5, universe_len)
    slider_max = max(slider_min, universe_len)
    st.session_state.setdefault("view", "Screener")
    st.session_state.setdefault("profile_name", next(iter(PROFILES)))
    st.session_state.setdefault("nl_provider", agent.DEFAULT_PROVIDER)
    st.session_state.setdefault("n_names", slider_max)
    st.session_state.setdefault("table_density", "Compact")
    # The watchlist is a plain set of starred symbols (NOT a widget key): the ★ toggle
    # in the detail panel mutates it, the "★ Watchlist only" pill below filters on it,
    # and persistence.py round-trips it through localStorage as a sorted list.
    st.session_state.setdefault("watchlist", set())

    with st.container(key="app_header"):
        # --- sticky bar -------------------------------------------------
        (
            _brand_col,
            _view_col,
            _search_col,
            _interpret_col,
            _run_col,
            _add_col,
            _settings_col,
        ) = st.columns([1.0, 3.0, 3.0, 1.2, 1.0, 0.5, 0.5], vertical_alignment="center")

        with _brand_col:
            # Compact wordmark (folds in the old st.title). Bold body text — NOT an h3 —
            # so it stays on one line in this narrow column. The long global disclaimer
            # moves to a single full-width caption under the bar (it otherwise bloated
            # this column to six wrapped lines) and also still lives in ⚙ Settings.
            st.markdown("**📈 Screener**")

        with _view_col:
            # NO default= — the "view" key is seeded above (D4). segmented_control can
            # deselect to None in single mode; callers read it None-safe.
            st.segmented_control(
                "View",
                ["Screener", "Heatmap", "Events"],
                key="view",
                label_visibility="collapsed",
            )

        with _search_col:
            # Dual-purpose search box (Decision D1): the on_change mirrors its text into
            # the client-side f_text filter (instant table narrowing, 0 engine calls);
            # the Interpret button reads nl_query for the natural-language path.
            st.text_input(
                "Search",
                key="nl_query",
                placeholder="Search or describe a screen…  ( / )",
                label_visibility="collapsed",
                on_change=_mirror_search_to_filter,
            )

        with _interpret_col:
            interpret_clicked = st.button(
                "Interpret",
                key="nl_btn",
                help="Interpret the box as a natural-language screen and run it.",
            )

        with _run_col:
            run_clicked = st.button(
                "Run ▶",
                type="primary",
                key="run_btn",
                help="Run the scan with the current profile and universe size.",
            )

        with _add_col:
            # add_ticker is self-contained & rerun-driven — safe to drop verbatim.
            with st.popover("➕", help="Add a ticker"):
                add_ticker.render_add_ticker(universe)

        with _settings_col:
            # ⚙ Settings popover: the lower-frequency controls relocated VERBATIM from
            # the sidebar (engine + availability, asset-class stub, universe size,
            # refresh, disclaimer). Its child widgets are instantiated every run, but
            # the D4 seeding above still defines the scan inputs even before first open.
            with st.popover("⚙", help="Settings"):
                # Initialize-then-no-default: seed the key once (so a remembered choice
                # from localStorage or a staged NL value survives) and pass NO index=,
                # the canonical pattern that avoids Streamlit's default-vs-session_state
                # warning.
                if "nl_provider" not in st.session_state:
                    st.session_state["nl_provider"] = agent.DEFAULT_PROVIDER
                st.radio(
                    "Engine",
                    options=list(agent.PROVIDERS),
                    format_func=lambda pid: agent.PROVIDERS[pid].label,
                    key="nl_provider",
                )
                selected = st.session_state.get("nl_provider", agent.DEFAULT_PROVIDER)
                prov = agent.PROVIDERS[selected]
                # Self-diagnosing status: say whether the backend is live and, if not,
                # exactly WHY (missing key / missing SDK) instead of silently using the
                # offline parser.
                _ok, _reason = agent.availability_status(selected)
                if _ok:
                    st.caption(f"✓ LLM-backed ({prov.label}) — interprets your request, then runs the scan.")
                else:
                    # The reason already names the exact missing piece (which key env
                    # var, or the SDK + pip command); the hint adds the non-obvious
                    # secrets.toml option.
                    st.caption(f"✗ {prov.label}: {_reason}. Using the offline rule-based parser.")
                    if prov.env_key:
                        st.caption(
                            "Tip: the key can live in your shell env OR `.streamlit/secrets.toml` "
                            "(see `.streamlit/secrets.toml.example`)."
                        )

                # Supplementary external data (Milestone B): a READ-ONLY, self-
                # diagnosing status line, mirroring the LLM-engine status above. The
                # overlay is env-gated (OFF by default) and key-less by design, so
                # there is deliberately NO toggle or key widget here — only its state.
                from screener import mcp_provider
                _mcp_ok, _mcp_reason = mcp_provider.availability_status()
                st.caption(
                    (f"✓ Supplementary data (MCP): {_mcp_reason}." if _mcp_ok
                     else f"○ Supplementary data (MCP): {_mcp_reason}."),
                    help="Opt-in external stock-data MCP server (detail-panel fundamentals + "
                         "earnings). Enable with MCP_STOCK_DATA_ENABLED=true; configure the "
                         "stdio command with MCP_STOCK_DATA_CMD (default: uvx yfmcp@latest).",
                )

                # Asset-class toggle: a disabled no-op stub (crypto is v2).
                st.radio("Asset class", ["US equities"], disabled=True, help="Crypto arrives in v2.")

                # min 5 / max = the whole universe / default = ALL names. The bounds are
                # clamped only so a degenerate tiny universe (fewer than 5 names) can't
                # make min_value exceed max_value.
                if slider_max > slider_min:
                    # Initialize-then-no-default: seed the key once (so a staged NL value
                    # or a prior selection survives) and pass NO value= arg, the canonical
                    # pattern that avoids Streamlit's default-vs-session_state warning.
                    # Default scans the FULL universe; dial the slider down for a faster
                    # cold scan.
                    if "n_names" not in st.session_state:
                        st.session_state["n_names"] = slider_max
                    n_names = st.slider(
                        "Universe size (names to scan)",
                        min_value=slider_min,
                        max_value=slider_max,
                        step=5,
                        key="n_names",
                        help="Defaults to all names. A cold scan hits Yahoo once per name; "
                             "dial this down for a faster first run.",
                    )
                else:
                    # Universe too small for a range slider — scan all of it. Mirror into
                    # the session key so the scan block (which reads
                    # st.session_state["n_names"]) has a value even though no keyed slider
                    # rendered.
                    n_names = universe_len
                    st.session_state["n_names"] = universe_len
                    st.caption(f"Universe has {universe_len} name(s); scanning all of them.")
                st.caption(display.universe_size_hint(n_names))

                clear_clicked = st.button(
                    "Clear cache & rescan",
                    key="clear_btn",
                    help="Delete today's cached prices/fundamentals and re-fetch from Yahoo, then rescan.",
                )

                st.divider()
                st.caption(display.DISCLAIMER_TEXT)
                with st.expander("Disclaimer — not financial advice"):
                    st.write(display.DISCLAIMER_DETAIL)

        # --- context row ------------------------------------------------
        # The profile control renders on ALL views (Decision D3) so its widget is
        # always present → profile_name is never garbage-collected (invariant #3).
        # Results-table-specific controls (density) render only on the Screener view.
        st.segmented_control(
            "Profile",
            options=list(PROFILES),
            format_func=lambda k: PROFILES[k].label,
            key="profile_name",
            label_visibility="collapsed",
        )
        # Decision D2: segmented_control has no captions= arg, so render the ACTIVE
        # profile's description as a caption line directly under the profile bar.
        st.caption(display.profile_description(st.session_state.get("profile_name")))

        if st.session_state.get("view") == "Screener":
            # Results-table controls (display-only — they restyle the LAST scan, never
            # refetch): density, the ★ watchlist-only filter, and a 🕘 Recent popover.
            _density_col, _wl_col, _recent_col = st.columns(
                [2.0, 1.4, 1.0], vertical_alignment="center"
            )
            with _density_col:
                # Compact = the lean finance-site view (price, daily change, tactical
                # readouts); Detailed reveals this profile's signal columns.
                st.segmented_control(
                    "Table density",
                    ["Compact", "Detailed"],
                    key="table_density",
                    label_visibility="collapsed",
                    help="Compact: rank, symbol, price, daily change, and the tactical readouts. "
                         "Detailed: also shows this profile's signal columns.",
                )
            with _wl_col:
                # Pure client-side filter (read by apply_filters via results_view). It
                # owns its own key and renders here for the first time this run, so a
                # direct key= is safe — no _pending_* staging needed (unlike Recent).
                st.checkbox(
                    "★ Watchlist only",
                    key="f_in_watchlist_only",
                    help="Show only tickers you've starred (☆/★) in the detail view.",
                )
            with _recent_col:
                # Re-apply a past scan. CRITICAL: this row renders AFTER the header's own
                # profile/n_names widgets exist this run, so we MUST NOT assign those
                # widget keys directly (Streamlit raises). Instead STAGE _pending_* and
                # rerun — app.py's apply_pending writes them BEFORE the widgets next run,
                # and _nl_run_after_apply fires the ONE guarded scan exactly once.
                with st.popover("🕘 Recent", help="Re-run a recent scan."):
                    _recent = st.session_state.get("recent_scans", [])
                    if not _recent:
                        st.caption("No recent scans yet — press Run ▶ to start.")
                    for _i, (_p, _n, _day) in enumerate(_recent[:5]):
                        _label = f"{PROFILES[_p].label} · {_n} names · {_day}" \
                            if _p in PROFILES else f"{_p} · {_n} names · {_day}"
                        if st.button(_label, key=f"recent_{_i}", use_container_width=True):
                            # Re-apply ONLY profile + size; a re-run must use TODAY's date
                            # (the stored cache_day is just a display label), so we never
                            # stage it — scan.run_scan_if_requested derives today itself.
                            st.session_state["_pending_profile_name"] = _p
                            st.session_state["_pending_n_names"] = _n
                            # A Recent re-run is NOT a natural-language scan, so drop any
                            # stale NL banner/add-result — otherwise it would describe a
                            # prior Interpret over these results. scan.py only clears those
                            # on a manual Run/Clear, not on an _nl_run_after_apply run.
                            st.session_state.pop("nl_last_req", None)
                            st.session_state.pop("nl_add_msg", None)
                            st.session_state["_nl_run_after_apply"] = True
                            st.rerun()

        # The global "educational, not advice" line (folded in from app.py's old title
        # caption). One full-width caption here instead of a six-line wrap in the brand
        # column; stays visible on every view.
        st.caption(
            "US large-cap · end-of-day · ranks and describes · educational buy zone, not advice"
        )

    return interpret_clicked, run_clicked, clear_clicked
