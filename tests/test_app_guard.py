"""Cold-scan-guard tests for the whole app (``app.py``) via ``AppTest``.

This LOCKS the architecture invariant the redesign rests on: ``app.py`` invokes the
engine at EXACTLY one site (``screener.ui.scan.run_scan_if_requested`` →
``screener.ui.caching.run_cached``). A cold page load, a nav/view switch, and any
client-side filter must trigger ZERO engine calls; only pressing **Run ▶** triggers
exactly one. Sorting/filtering/heatmap/watchlist are pure views over the cached
``st.session_state["scan"]["df"]`` and must never re-run the engine.

House style (matches ``tests/test_calendar.py`` / ``tests/test_sector_heatmap.py``):
no ``pytest`` import and no fixtures — every test is a plain ``test_*`` function using
``assert``, so the suite runs BOTH under ``python -m pytest tests/test_app_guard.py``
AND standalone as ``python tests/test_app_guard.py`` (the ``__main__`` runner counts
pass/fail, prints a summary, and exits non-zero on any failure).

Two neutralizations are REQUIRED for ``AppTest`` to run this app headlessly; both are
applied by :func:`make_app` to a FRESH ``AppTest`` per test (so no state leaks):

1. **Block the localStorage component hang.** ``persistence._make_storage`` mounts a
   custom localStorage component that never settles under ``AppTest`` (no browser to
   answer), hanging the run. Patching it to return ``None`` makes
   ``apply_remembered`` / ``remember_current`` no-op and the script settle — the app
   already degrades to "forgetful but working" when storage is absent.
2. **Count engine calls at the lookup site.** ``scan.py`` does
   ``from screener.ui.caching import run_cached``, binding the name into the
   ``scan`` module, so we replace ``screener.ui.scan.run_cached`` (NOT
   ``caching.run_cached``) with a counter stub. The stub returns an EMPTY
   ``pd.DataFrame()`` — which ``display.is_empty_result`` treats as the ENGINE_EMPTY
   state, so the Screener renders a "no matches" message and NEVER the AgGrid, so a
   missing-column render can't crash the headless run — and bumps a counter.

AppTest quirk worked around (Streamlit 1.50): ``st.segmented_control`` in SINGLE
selection mode is represented by ``AppTest``'s ``ButtonGroup`` proxy, whose
``.indices`` property does ``[... for v in self.value]``. For a single-mode control
the session value is a SCALAR string (e.g. ``"Screener"``), so the proxy iterates it
CHARACTER-by-character and ``format_func("S")`` raises ``ValueError: "S" is not in
list``. ``AppTest`` recomputes all widget states on every ``ElementTree.run`` AFTER
the first, so the SECOND ``.run()`` (and every ``.click().run()``) crashes on the
app's three single-mode segmented_controls (``view`` / ``profile_name`` /
``table_density``). :func:`_patch_button_group_single_mode` shims ``.indices`` to
treat a scalar value as a one-element selection (and skip non-matching options),
which is exactly what a single-mode control means.
"""

import os
import sys

# Put the repo root on sys.path so "import screener" / AppTest's app import resolves
# when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import streamlit.testing.v1.element_tree as element_tree  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import screener.ui.persistence as persistence  # noqa: E402
import screener.ui.scan as scan  # noqa: E402

# The app entrypoint, resolved relative to this file so the runner works from any cwd.
_APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)


def _patch_button_group_single_mode() -> None:
    """Shim ``AppTest``'s ``ButtonGroup.indices`` for single-mode segmented_control.

    See the module docstring: the stock proxy iterates a scalar value char-by-char.
    This version normalizes a non-list value to a one-element list and skips options
    that don't map back (defensive), so recomputing widget states on a second run
    doesn't crash on ``view`` / ``profile_name`` / ``table_density``.
    """

    def _safe_indices(self):
        value = self.value
        if not isinstance(value, (list, tuple)):
            value = [] if value is None else [value]
        out = []
        for v in value:
            formatted = self.format_func(v)
            if formatted in self.options:
                out.append(self.options.index(formatted))
        return out

    element_tree.ButtonGroup.indices = property(_safe_indices)


def make_app():
    """Return ``(AppTest, counter)`` for a FRESH app with both neutralizations applied.

    The counter is a ``{"n": int}`` dict the engine stub increments; per-test freshness
    means no scan/session state leaks between tests. The patches are re-applied here
    (module-global monkeypatches), so each test is self-contained even if another test
    ran first.
    """
    # (1) neutralize the localStorage component hang.
    persistence._make_storage = lambda: None

    # (2) count engine calls at the name scan.py looks up.
    counter = {"n": 0}

    def _engine_stub(profile_name, n_names, cache_day, include_symbol=""):
        counter["n"] += 1
        return pd.DataFrame()  # empty -> ENGINE_EMPTY state, never the AgGrid

    scan.run_cached = _engine_stub

    # AppTest quirk shim (idempotent; safe to re-apply per test).
    _patch_button_group_single_mode()

    return AppTest.from_file(_APP_PATH), counter


def _find_button(at, key):
    """The keyed button, via ``at.button(key=...)`` with a ``.key``-match fallback.

    ``AppTest``'s keyed lookup works here, but the spec asks for a fallback in case it
    can't find the button — so we try the iteration form if the accessor misses.
    """
    try:
        return at.button(key=key)
    except Exception:
        pass
    for b in at.button:
        if b.key == key:
            return b
    raise AssertionError(f"no button with key={key!r} found")


# --- 1. cold load -> ZERO engine calls -----------------------------------
def test_cold_load_runs_no_scan():
    at, counter = make_app()
    at.run(timeout=60)
    assert not at.exception, f"cold load raised: {at.exception}"
    assert counter["n"] == 0, f"cold load triggered {counter['n']} engine call(s)"
    # Sanity: the page defaulted to the Screener view (cold-scan guard, no table).
    assert at.session_state["view"] == "Screener"


# --- 2. view switch -> ZERO engine calls ---------------------------------
def test_view_switch_runs_no_scan():
    at, counter = make_app()
    at.run(timeout=60)
    assert counter["n"] == 0
    # Drive the nav by setting the session_state key directly + rerun (simplest way to
    # simulate the segmented_control without the single-mode .select quirk).
    for view in ("Heatmap", "Events", "Screener"):
        at.session_state["view"] = view
        at.run()
        assert not at.exception, f"view={view} raised: {at.exception}"
        assert counter["n"] == 0, f"switching to {view} triggered an engine call"


# --- 3. client-side filter -> ZERO engine calls --------------------------
def test_client_side_filter_runs_no_scan():
    at, counter = make_app()
    at.run(timeout=60)
    assert counter["n"] == 0
    # The header search mirrors into f_text; set it (and the watchlist-only pill)
    # directly and rerun — both are pure pandas filters over the cached df.
    at.session_state["f_text"] = "AAPL"
    at.run()
    assert not at.exception
    assert counter["n"] == 0, "text filter triggered an engine call"
    at.session_state["f_in_watchlist_only"] = True
    at.run()
    assert not at.exception
    assert counter["n"] == 0, "watchlist-only filter triggered an engine call"


# --- 4. Run button -> EXACTLY one engine call ----------------------------
def test_run_button_runs_one_scan():
    at, counter = make_app()
    at.run(timeout=60)
    assert counter["n"] == 0
    _find_button(at, "run_btn").click().run()
    assert not at.exception, f"Run raised: {at.exception}"
    assert counter["n"] == 1, f"Run triggered {counter['n']} engine call(s), expected 1"


def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report any unexpected error
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
    total = passed + failed
    print(f"\n{passed}/{total} passed" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
