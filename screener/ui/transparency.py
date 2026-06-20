"""NL transparency banner (relocated verbatim from app.py).

Shows how the last natural-language request was interpreted. Rendered ABOVE the
four-state switch so the user always sees the interpretation that drove the
current scan (the explanation + the resolved knobs).
"""

from __future__ import annotations

import streamlit as st


# --- NL transparency: show how the last request was interpreted ----------
# Above the four-state switch so the user always sees the interpretation that
# drove the current scan (the explanation + the resolved knobs).
def render_nl_banner() -> None:
    """Render the interpreted-request banner if an NL request drove this scan."""
    # The add-result of the last Interpret (success/error from the auto-add), so the
    # natural-language path is never silent. Persists alongside ``nl_last_req`` until
    # the next Interpret (which clears it) or a manual scan (scan.py pops both).
    add_msg = st.session_state.get("nl_add_msg")
    if add_msg is not None:
        kind, text = add_msg
        {"success": st.success, "error": st.error}.get(kind, st.info)(text)
    last = st.session_state.get("nl_last_req")
    if last is not None:
        st.info(f"Interpreted your request — {last.explanation}")
        st.caption(
            f"profile={last.profile} · names={last.n_names} · min_score={last.min_score:g}"
            + (f" · sectors={', '.join(last.sectors)}" if last.sectors else "")
            + (f" · symbol={last.text}" if last.text else "")
            + (" · earnings-only" if last.earnings_only else "")
        )
