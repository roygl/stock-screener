"""Dedicated "add a ticker to the universe" sidebar control.

A first-class, explicit alternative to typing a symbol into the natural-language
box. The NL path only auto-adds an ALL-CAPS token it happens to recognise and
reports NOTHING back, so a lowercase symbol, a dotted class share (``BRK.B``), or
a failed fetch all look identical to "nothing happened". This control instead:

* canonicalises the input (strip/upper) so ``pltr`` and ``brk.b`` work;
* validates the SHAPE against the same :data:`screener.universe._TICKER_RE` the
  writer uses, so the error message can distinguish "not a ticker" from the other
  failure modes;
* delegates the network fetch + CSV write to :func:`screener.universe.ensure_symbol`
  (the single, already-tested writer — no duplicate persistence logic here);
* ALWAYS surfaces the outcome: ``✓ Added PLTR — universe now 613 names``, "already
  in the universe", "doesn't look like a ticker", or "Yahoo returned no data".

Feedback is stashed in ``st.session_state`` and rendered on the next run because a
successful add ends in ``st.rerun()`` (so the universe-size slider and counts, all
rendered ABOVE this control, pick up the new name).
"""

from __future__ import annotations

import streamlit as st

# Survives the post-add st.rerun(): (kind, message) where kind ∈ success/info/error.
_FEEDBACK_KEY = "_add_ticker_feedback"
_INPUT_KEY = "add_ticker_input"


def render_add_ticker(universe) -> None:
    """Render the add-ticker input + button in the sidebar and handle a click.

    Renders (then clears) any feedback staged by a previous run, draws the symbol
    input and "Add to universe" button, and on click delegates to :func:`_handle_add`.
    ``universe`` is the CURRENT (pre-add) frame loaded at the top of this run; it is
    used only to read the prior size so a "scan all" slider can be kept at "all".
    """
    feedback = st.session_state.pop(_FEEDBACK_KEY, None)
    if feedback is not None:
        kind, msg = feedback
        {"success": st.success, "info": st.info, "error": st.error}.get(kind, st.info)(msg)

    st.text_input(
        "Add a ticker to the universe",
        key=_INPUT_KEY,
        placeholder="e.g. PLTR, brk.b",
        help="Fetches the company name + sector from Yahoo and saves the symbol to "
             "data/universe.csv. Lowercase is fine. Press Run scan afterwards to include it.",
    )
    if st.button("Add to universe", key="add_ticker_btn"):
        _handle_add(universe)


def _handle_add(universe) -> None:
    """Validate, fetch + persist, stage feedback, then rerun so the counts refresh.

    Computes a single ``(kind, message)`` outcome and reruns once. The two
    pre-checks (shape, membership) exist so the message can name the precise reason;
    only an otherwise-valid, not-yet-present symbol reaches the network call. On a
    successful add the scan memo is dropped, and if the universe-size slider was at
    its old maximum ("scan everything") it is bumped — via the same ``_pending_*``
    channel the NL flow uses — to the new size so the freshly appended (last-ranked)
    row is actually inside the next scan's head slice.
    """
    # _TICKER_RE is the writer's single source of truth for "ticker-shaped"; reuse it
    # here (rather than re-deriving a regex) so this gate can never drift from the one
    # inside ensure_symbol. Imported lazily to keep the sidebar import-cheap.
    from screener.universe import _TICKER_RE, ensure_symbol, load_universe

    raw = str(st.session_state.get(_INPUT_KEY, "")).strip()
    sym = raw.upper()

    if not sym:
        feedback = ("info", "Type a ticker symbol first.")
    elif not _TICKER_RE.match(sym):
        feedback = (
            "error",
            f"“{raw}” doesn’t look like a ticker "
            "(1–5 letters, optional .class suffix like BRK.B).",
        )
    elif sym in set(load_universe()["symbol"]):
        feedback = ("info", f"{sym} is already in the universe ({len(load_universe())} names).")
    elif ensure_symbol(sym):
        # ensure_symbol cleared load_universe's cache, so this reads the fresh count.
        from screener.ui.caching import run_cached

        run_cached.clear()
        old_size = len(universe)
        new_size = len(load_universe())
        # Keep a "scan all" selection at "all": only bump when the slider sat at the
        # prior max. Staged into _pending_n_names so nl_state.apply_pending applies it
        # at the TOP of the next run (setting n_names here would raise — its widget was
        # already instantiated above this control).
        if int(st.session_state.get("n_names", 0)) >= old_size:
            st.session_state["_pending_n_names"] = new_size
        feedback = ("success", f"✓ Added {sym} — universe now {new_size} names. Press Run scan to include it.")
    else:
        feedback = ("error", f"Couldn’t add {sym}: Yahoo returned no data for that symbol.")

    st.session_state[_FEEDBACK_KEY] = feedback
    st.rerun()
