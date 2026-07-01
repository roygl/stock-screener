#!/usr/bin/env python3
"""Apply the mobile fix (Decision D8) to grid.py and header.py in place.

Run from the repo root:  python3 apply_mobile_fix.py
Safe to run more than once — it detects an already-applied edit and skips it.
"""
from __future__ import annotations
import sys
from pathlib import Path

GRID = Path("screener/ui/grid.py")
HEADER = Path("screener/ui/header.py")

# --- grid.py: insert the 60vh cap right after the GridOptionsBuilder line ---
GRID_ANCHOR = "    gb = GridOptionsBuilder.from_dataframe(table_df)\n"
GRID_BLOCK = (
    "    gb = GridOptionsBuilder.from_dataframe(table_df)\n"
    "    # --- Mobile grid sizing (Decision D8) --------------------------------\n"
    "    # On a phone the AgGrid's own vertical scroll competes with the page scroll:\n"
    "    # the grid can grow to 540px (most of a small viewport), so a swipe that\n"
    "    # starts inside it scrolls rows, and only \"catches\" the page at the grid's\n"
    "    # edge — the trapped-scroll feel the user hit. Capping the grid to a fraction\n"
    "    # of the viewport (max-height) keeps it a compact, self-contained scroll area\n"
    "    # with clear page above/below it to grab. A contained <style> (mirrors the\n"
    "    # header's single-style precedent); desktop keeps the Python `height` below.\n"
    "    st.markdown(\n"
    '        """\n'
    "        <style>\n"
    "        @media (max-width: 640px) {\n"
    '            [data-testid="stAgGrid"], .st-key-results_grid div[class*="ag-theme"] {\n'
    "                max-height: 60vh !important;\n"
    "            }\n"
    "        }\n"
    "        </style>\n"
    '        """,\n'
    "        unsafe_allow_html=True,\n"
    "    )\n"
)
GRID_IMPORT_ANCHOR = "import pandas as pd\nfrom st_aggrid import"
GRID_IMPORT_NEW = "import pandas as pd\nimport streamlit as st\nfrom st_aggrid import"

# --- header.py: insert the mobile media query before the closing </style> ---
HEADER_ANCHOR = (
    "        @media (prefers-color-scheme: dark) {\n"
    "            .st-key-app_header { background-color: rgb(14, 17, 23); }\n"
    "        }\n"
    "        </style>"
)
HEADER_BLOCK = (
    "        @media (prefers-color-scheme: dark) {\n"
    "            .st-key-app_header { background-color: rgb(14, 17, 23); }\n"
    "        }\n"
    "        /* --- Mobile (Decision D8) -------------------------------------------\n"
    "           On a phone Streamlit stacks every column in the header vertically, so\n"
    "           the control surface (search, Interpret, Run, profile bar, density /\n"
    "           watchlist / recent row, captions) grew taller than the viewport and\n"
    "           pushed the results table fully below the fold. Worse, a `position:\n"
    "           sticky` header that is itself taller than the screen leaves no room to\n"
    '           scroll the table "under" it and the scroll appears to fight back.\n'
    "           So on narrow screens we (a) DROP the sticky pin — the header scrolls\n"
    "           away normally, freeing the whole viewport for the table — and (b)\n"
    "           tighten the vertical rhythm so the table is reachable in one short\n"
    "           swipe. Desktop is untouched (the pin + spacing above still apply). */\n"
    "        @media (max-width: 640px) {\n"
    '            [data-testid="stLayoutWrapper"]:has(> .st-key-app_header) {\n'
    "                position: static;\n"
    "            }\n"
    "            .st-key-app_header { padding-bottom: 0.25rem; }\n"
    "            /* Shrink the gaps Streamlit puts between the stacked rows/widgets so\n"
    "               the header isn't a full screen of whitespace on mobile. */\n"
    '            .st-key-app_header [data-testid="stVerticalBlock"] { gap: 0.4rem; }\n'
    '            .st-key-app_header [data-testid="stCaptionContainer"] { margin: 0; }\n'
    "        }\n"
    "        </style>"
)


def patch(path: Path, anchor: str, block: str, marker: str, label: str) -> bool:
    if not path.exists():
        print(f"  SKIP {label}: {path} not found (are you in the repo root?)")
        return False
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"  OK   {label}: already applied")
        return True
    if anchor not in text:
        print(f"  FAIL {label}: anchor not found — file may differ from expected")
        return False
    path.write_text(text.replace(anchor, block, 1), encoding="utf-8")
    print(f"  DONE {label}: patched")
    return True


def main() -> int:
    print("Applying mobile fix (Decision D8)...")
    ok = True
    # grid.py needs the streamlit import too (it uses st.markdown now)
    gtext = GRID.read_text(encoding="utf-8") if GRID.exists() else ""
    if GRID.exists() and "import streamlit as st" not in gtext:
        GRID.write_text(gtext.replace(GRID_IMPORT_ANCHOR, GRID_IMPORT_NEW, 1), encoding="utf-8")
        print("  DONE grid.py import: added `import streamlit as st`")
    ok &= patch(GRID, GRID_ANCHOR, GRID_BLOCK, "max-height: 60vh", "grid.py")
    ok &= patch(HEADER, HEADER_ANCHOR, HEADER_BLOCK, "max-width: 640px", "header.py")
    print()
    if ok:
        print("All edits applied. Verify:  grep -c 60vh screener/ui/grid.py   (expect 1)")
        return 0
    print("Something didn't apply cleanly — paste the output above back to me.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
