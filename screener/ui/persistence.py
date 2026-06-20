"""Remember the sidebar choices in the visitor's browser via localStorage.

The profile, NL engine, universe size, and table density survive a page refresh,
a new tab, and a browser restart by round-tripping through
``streamlit_local_storage`` (a localStorage-backed component). Nothing leaves the
browser — the values are stored client-side only.

localStorage is asynchronous: the component returns ``{}`` on the very first
script run of a session and reruns once the browser has handed its data back.
So the flow is:

* :func:`apply_remembered` runs BEFORE the sidebar widgets. The first run on which
  the browser's stored values are actually available, it copies the valid ones
  into the matching ``st.session_state`` widget keys (overwriting the
  just-rendered defaults) — exactly once per session.
* :func:`remember_current` runs AFTER the sidebar and writes back any value that
  changed, so the latest selection is what gets remembered next time.

Precedence: an explicit NL "Interpret" (the ``_pending_*`` staging in
:mod:`screener.ui.nl_state`) runs AFTER :func:`apply_remembered`, so a
natural-language request still wins for that run. Everything degrades to a no-op
if the component/package is unavailable (the app keeps working, just forgetful).
"""

from __future__ import annotations

import streamlit as st

from screener import agent
from screener.profiles import PROFILES

# The sidebar widget keys we persist. All are plain scalars except "watchlist",
# which is a set of starred symbols (serialized as a sorted list, see below).
_PERSIST_KEYS = ("profile_name", "nl_provider", "table_density", "n_names", "watchlist")
_DENSITIES = ("Compact", "Detailed")
_APPLIED_FLAG = "_persist_applied"


def _coerce(key, raw, universe):
    """Coerce a stored value (localStorage returns strings) to a valid widget
    value, or ``None`` if it is missing/invalid — so a stale or hand-edited store
    can never push an out-of-range value into a keyed widget (which would raise)."""
    if raw is None:
        return None
    if key == "profile_name":
        return raw if raw in PROFILES else None
    if key == "nl_provider":
        return raw if raw in agent.PROVIDERS else None
    if key == "table_density":
        return raw if raw in _DENSITIES else None
    if key == "n_names":
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return None
        lo = min(5, len(universe))
        hi = max(lo, len(universe))
        return max(lo, min(n, hi))
    if key == "watchlist":
        # Stored as a JSON list; rehydrate to the in-app set type. An empty list
        # coerces to an empty set (not None) so a cleared watchlist is honored.
        if not isinstance(raw, (list, tuple)):
            return None
        return {str(s) for s in raw}
    return None


def _make_storage():
    """Instantiate the localStorage handle, or ``None`` if the component/package
    is missing or fails to mount (keeps the app working without persistence)."""
    try:
        from streamlit_local_storage import LocalStorage

        return LocalStorage()
    except Exception:
        return None


def apply_remembered(universe):
    """Seed remembered choices into ``st.session_state`` before the widgets render.

    Applies the stored values at most once per session — on the first run where
    the browser has actually returned them — and returns the localStorage handle
    (or ``None``) for :func:`remember_current` to reuse.
    """
    ls = _make_storage()
    if ls is None or st.session_state.get(_APPLIED_FLAG):
        return ls
    try:
        stored = ls.getAll() or {}
    except Exception:
        stored = {}
    if not stored:
        # Browser hasn't responded yet (first run) or nothing saved — retry next run.
        return ls
    for key in _PERSIST_KEYS:
        value = _coerce(key, stored.get(key), universe)
        if value is not None:
            st.session_state[key] = value
    st.session_state[_APPLIED_FLAG] = True
    return ls


def remember_current(ls, universe):
    """Write back any choice that differs from what's stored, so the latest
    selection is remembered. A no-op when nothing changed (avoids redundant
    component writes/reruns)."""
    if ls is None:
        return
    try:
        stored = ls.getAll() or {}
    except Exception:
        stored = {}
    for key in _PERSIST_KEYS:
        current = st.session_state.get(key)
        if current is None:
            continue
        if key == "watchlist":
            current = sorted(current)   # set -> deterministic, JSON-serializable list
        if str(stored.get(key)) != str(current):
            try:
                ls.setItem(key, current, key=f"_persist_set_{key}")
            except Exception:
                pass
