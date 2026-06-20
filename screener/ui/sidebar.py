"""The sidebar: controls, filters, disclaimer (relocated verbatim from app.py).

:func:`render_sidebar` renders the WHOLE ``with st.sidebar`` block in order and
returns the three action flags the downstream handlers consume
(``interpret_clicked``, ``run_clicked``, ``clear_clicked``). The profile / size
the rest of the script needs are read back out of ``st.session_state`` (the
widget keys this block owns), not returned, exactly as the original did.
"""

from __future__ import annotations

import streamlit as st

from screener import agent, display
from screener.profiles import PROFILES, get_profile


# --- sidebar: controls, filters, disclaimer ------------------------------
def render_sidebar(universe) -> "tuple[bool, bool, bool]":
    """Render the sidebar; return (interpret_clicked, run_clicked, clear_clicked)."""
    with st.sidebar:
        # Natural-language box FIRST (above Controls). It is a SECOND trigger into the
        # single engine call site — never a separate scan path.
        st.subheader("Ask in plain English")
        st.text_input(
            "Describe what to screen for",
            key="nl_query",
            placeholder="e.g. top 20 momentum tech names, high conviction",
        )
        st.radio(
            "Engine",
            options=list(agent.PROVIDERS),
            index=list(agent.PROVIDERS).index(agent.DEFAULT_PROVIDER),
            format_func=lambda pid: agent.PROVIDERS[pid].label,
            key="nl_provider",
        )
        selected = st.session_state.get("nl_provider", agent.DEFAULT_PROVIDER)
        prov = agent.PROVIDERS[selected]
        # Self-diagnosing status: say whether the backend is live and, if not, exactly
        # WHY (missing key / missing SDK) instead of silently using the offline parser.
        _ok, _reason = agent.availability_status(selected)
        if _ok:
            st.caption(f"✓ LLM-backed ({prov.label}) — interprets your request, then runs the scan.")
        else:
            # The reason already names the exact missing piece (which key env var, or
            # the SDK + pip command); the hint adds the non-obvious secrets.toml option.
            st.caption(f"✗ {prov.label}: {_reason}. Using the offline rule-based parser.")
            if prov.env_key:
                st.caption(
                    "Tip: the key can live in your shell env OR `.streamlit/secrets.toml` "
                    "(see `.streamlit/secrets.toml.example`)."
                )
        interpret_clicked = st.button("Interpret & run", key="nl_btn")
        st.divider()

        st.header("Controls")

        profile_name = st.radio(
            "Profile",
            options=list(PROFILES),
            format_func=lambda k: PROFILES[k].label,
            captions=[display.profile_description(k) for k in PROFILES],
            key="profile_name",
        )
        profile = get_profile(profile_name)

        # Asset-class toggle: a disabled no-op stub (crypto is v2).
        st.radio("Asset class", ["US equities"], disabled=True, help="Crypto arrives in v2.")

        # Results-table density (a display preference; restyles the last scan, no refetch).
        # Compact = the lean finance-site view (price, daily change, tactical readouts);
        # Detailed reveals this profile's signal columns.
        st.radio(
            "Table density",
            ["Compact", "Detailed"],
            key="table_density",
            horizontal=True,
            help="Compact: rank, symbol, price, daily change, and the tactical readouts. "
                 "Detailed: also shows this profile's signal columns.",
        )

        # The production S&P 500 universe is 503 names, so this is min 5 / max 503 /
        # default 25 as specced. The bounds are clamped only so a degenerate tiny
        # universe (fewer than 5 names) can't make min_value exceed max_value.
        universe_len = len(universe)
        slider_min = min(5, universe_len)
        slider_max = max(slider_min, universe_len)
        if slider_max > slider_min:
            # Initialize-then-no-default: seed the key once (so a staged NL value or a
            # prior selection survives) and pass NO value= arg, the canonical pattern
            # that avoids Streamlit's default-vs-session_state warning.
            if "n_names" not in st.session_state:
                st.session_state["n_names"] = min(25, slider_max)
            n_names = st.slider(
                "Universe size (names to scan)",
                min_value=slider_min,
                max_value=slider_max,
                step=5,
                key="n_names",
                help="A cold scan hits Yahoo once per name; start small.",
            )
        else:
            # Universe too small for a range slider — scan all of it. Mirror into the
            # session key so the scan block (which reads st.session_state["n_names"])
            # has a value even though no keyed slider rendered.
            n_names = universe_len
            st.session_state["n_names"] = universe_len
            st.caption(f"Universe has {universe_len} name(s); scanning all of them.")
        st.caption(display.universe_size_hint(n_names))

        run_clicked = st.button("Run scan", type="primary", key="run_btn")
        st.caption("Sorting and filtering act on the last scan — no refetch.")
        clear_clicked = st.button(
            "Clear cache & rescan",
            key="clear_btn",
            help="Delete today's cached prices/fundamentals and re-fetch from Yahoo, then rescan.",
        )

        # Filters appear only after a scan exists.
        if "scan" in st.session_state:
            st.divider()
            st.subheader("Filters")
            scanned = st.session_state["scan"]["df"]
            st.text_input("Filter symbol / name", key="f_text")
            st.multiselect("Sector", options=display.sector_options(scanned), key="f_sectors")
            # Initialize-then-no-default (drop the positional 0.0) so a staged NL score
            # floor is read from session_state without the default-vs-session_state warning.
            if "f_min_score" not in st.session_state:
                st.session_state["f_min_score"] = 0.0
            st.slider("Minimum score", 0.0, 1.0, step=0.01, key="f_min_score")
            if "earnings_in_window" in profile.flags:
                st.checkbox("Only earnings-in-window", key="f_earnings_only")
            # Universe-wide TA filters (columns exist on every profile's scan; gate on
            # presence so they're no-ops if the engine ever omits them).
            if "extension_state" in scanned.columns:
                st.checkbox("Hide overextended (parabolic)", key="f_extended_hidden")
            if "in_buy_zone" in scanned.columns:
                st.checkbox("In buy zone only", key="f_in_buy_zone_only")

        st.divider()
        st.caption(display.DISCLAIMER_TEXT)
        with st.expander("Disclaimer — not financial advice"):
            st.write(display.DISCLAIMER_DETAIL)

    return interpret_clicked, run_clicked, clear_clicked
