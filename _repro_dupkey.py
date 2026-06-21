"""Throwaway: reproduce the StreamlitDuplicateElementKey by driving the RESULTS path.

The guard test's engine stub returns an EMPTY frame (ENGINE_EMPTY), so the grid /
selectbox / detail / sidebar filters never render. Here we return a REAL multi-row
ranked frame (offline FakeProvider) so the full RESULTS path renders, then click Run
and print the un-redacted exception + traceback.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit.testing.v1.element_tree as element_tree
from streamlit.testing.v1 import AppTest

import screener.engine as eng
import screener.ui.persistence as persistence
import screener.ui.scan as scan
from tests.test_engine import AS_OF, FakeProvider, _make_frame, _universe

_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


def _patch_button_group_single_mode():
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


# A small, all-trending synthetic universe so the "all" profile yields >1 ranked row.
_ROWS = [("AAA", "Alpha Co", "Technology"), ("BBB", "Beta Co", "Technology"),
         ("CCC", "Gamma Co", "Energy")]
_FRAMES = {sym: _make_frame(seed=i, drift=0.3 + 0.05 * i)
           for i, (sym, _, _) in enumerate(_ROWS)}
_UNIV = _universe(_ROWS)
_PROVIDER = FakeProvider(frames=_FRAMES)


def _report(at, label):
    if at.exception:
        for e in at.exception:
            print(f"\n===== EXCEPTION @ {label} =====")
            print("type:", getattr(e, "type", "?"))
            print("message:", getattr(e, "message", e))
            tb = getattr(e, "stack_trace", None)
            if tb:
                print("".join(tb) if isinstance(tb, list) else tb)
        return True
    print(f"ok @ {label}")
    return False


def _click(at, key):
    for b in at.button:
        if b.key == key:
            b.click()
            return True
    return False


def main():
    persistence._make_storage = lambda: None
    _patch_button_group_single_mode()

    def _engine_stub(profile_name, n_names, cache_day, include_symbol=""):
        return eng.run_screen(profile_name, _UNIV, _PROVIDER, as_of=AS_OF)

    scan.run_cached = _engine_stub

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=60)
    if _report(at, "cold load"):
        return

    # Step 1: Run with default "all" profile -> RESULTS.
    _click(at, "run_btn"); at.run(timeout=60)
    if _report(at, "run(all)"):
        return

    # Step 2: Run AGAIN (populates recent_scans, renders the Recent popover content).
    _click(at, "run_btn"); at.run(timeout=60)
    if _report(at, "run(all) #2 + Recent popover"):
        return

    # Step 3: Detailed density.
    at.session_state["table_density"] = "Detailed"; at.run(timeout=60)
    if _report(at, "density=Detailed"):
        return

    # Step 4: switch profile to swing, run.
    at.session_state["profile_name"] = "swing"; at.run(timeout=60)
    _report(at, "profile=swing (no rerun-scan)")
    _click(at, "run_btn"); at.run(timeout=60)
    if _report(at, "run(swing)"):
        return

    # Step 5: each profile in turn.
    for prof in ("momentum", "long_term", "all"):
        at.session_state["profile_name"] = prof
        at.run(timeout=60)
        _click(at, "run_btn"); at.run(timeout=60)
        if _report(at, f"run({prof})"):
            return

    # Step 6: view round-trips after a scan.
    for view in ("Heatmap", "Events", "Screener"):
        at.session_state["view"] = view; at.run(timeout=60)
        if _report(at, f"view={view}"):
            return

    # Step 7: watchlist toggle on the inspected symbol, twice.
    if _click(at, "wl_toggle_AAA"):
        at.run(timeout=60)
        if _report(at, "watchlist toggle AAA on"):
            return
    at.session_state["f_in_watchlist_only"] = True; at.run(timeout=60)
    if _report(at, "watchlist-only filter"):
        return

    print("\nNo duplicate-key exception across all steps.")


if __name__ == "__main__":
    main()
