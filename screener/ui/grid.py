"""The AgGrid results table (relocated verbatim from app.py).

All ``JsCode`` constants, the per-column configuration, and ``render_results_grid``
live here. The grid is the double-click-to-inspect main table; it reruns on
SELECTION_CHANGED only (NOT on sort/filter) so the cold-scan guard holds.
"""

from __future__ import annotations

import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from screener import display


# --- results grid (st-aggrid): DOUBLE-CLICK a row to inspect it ----------
# The main results table is an AgGrid because the native st.dataframe only
# supports single-click row selection. `suppressRowClickSelection` + an
# `onRowDoubleClicked` handler make DOUBLE-CLICK the selection gesture; the grid
# reruns on SELECTION_CHANGED (NOT on sort/filter, so the cold-scan guard holds)
# and we read the chosen symbol back out. The pure column descriptors
# (display.column_config_spec) are realised here as AgGrid colDefs — the same
# purity boundary the old _build_column_config used for st.dataframe.
_JS_DBLCLICK_SELECT = JsCode("function(e){ e.node.setSelected(true); }")
# This AG Grid React build accepts NEITHER an HTML-string cellRenderer (escaped to
# raw text) NOR a DOM-node one (React error #31). So we render ONLY via
# valueFormatters (plain text) + cellStyle (a plain style object) — never a
# cellRenderer. The Fit "bar" is a cellStyle background gradient behind the number.
_JS_YESNO = JsCode(
    "function(p){ return p.value === true ? 'Yes' : (p.value === false ? 'No' : '—'); }"
)
_JS_PERCENT = JsCode(
    "function(p){ return (p.value==null||isNaN(p.value)) ? '—' : "
    "(p.value*100).toFixed(1) + '%'; }"
)
# Headline-price formatters: a "$"-prefixed money cell and a SIGNED percent
# (fraction ×100 with an explicit +/-), plus a green/red colour for the daily
# change — finance-site convention. Style stays a plain object (the file's
# "valueFormatter + cellStyle only, never cellRenderer" rule).
_JS_MONEY = JsCode(
    "function(p){ return (p.value==null||isNaN(p.value)) ? '—' : "
    "'$' + Number(p.value).toFixed(2); }"
)
_JS_PERCENT_SIGNED = JsCode(
    "function(p){ if(p.value==null||isNaN(p.value)) return '—'; "
    "var v=p.value*100; return (v>=0?'+':'') + v.toFixed(2) + '%'; }"
)
_JS_CHANGE_STYLE = JsCode(
    "function(p){ if(p.value==null||isNaN(p.value)) return {}; "
    "return {color: p.value >= 0 ? '#188038' : '#d93025'}; }"
)
_JS_FIT_STYLE = JsCode(
    "function(p){"
    " if(p.value==null||isNaN(p.value)) return {};"
    " var v=Math.max(0,Math.min(100,p.value));"
    " return {background:'linear-gradient(90deg,#bcd3f7 '+v+'%, rgba(0,0,0,0) '+v+'%)',"
    " borderRadius:'3px'};"
    " }"
)


def _js_number(decimals: int, suffix: str = "") -> JsCode:
    """A numeric valueFormatter: fixed decimals + optional suffix, '—' for NaN."""
    return JsCode(
        "function(p){ return (p.value==null||isNaN(p.value)) ? '—' : "
        f"Number(p.value).toFixed({decimals}) + '{suffix}'; }}"
    )


def _configure_aggrid_column(gb, col: str, desc: dict) -> None:
    """Realise one pure column descriptor as an AgGrid column on ``gb``."""
    label = desc.get("label", col)
    tip = desc.get("help") or ""
    kind = desc.get("kind")
    fmt = desc.get("format")
    if col == "fit":
        gb.configure_column(col, header_name=label, headerTooltip=tip,
                            valueFormatter=_js_number(0), cellStyle=_JS_FIT_STYLE,
                            type=["numericColumn"], width=86)
    elif col == "price":
        gb.configure_column(col, header_name=label, headerTooltip=tip,
                            valueFormatter=_JS_MONEY, type=["numericColumn"], width=96)
    elif col == "change_pct":
        gb.configure_column(col, header_name=label, headerTooltip=tip,
                            valueFormatter=_JS_PERCENT_SIGNED, cellStyle=_JS_CHANGE_STYLE,
                            type=["numericColumn"], width=104)
    elif col == "why":
        gb.configure_column(col, header_name=label, headerTooltip=tip, minWidth=260,
                            flex=1, sortable=False, tooltipField=col)
    elif kind == "percent":
        gb.configure_column(col, header_name=label, headerTooltip=tip,
                            valueFormatter=_JS_PERCENT, type=["numericColumn"], width=110)
    elif kind == "number":
        decimals = {"%.0f": 0, "%.1f": 1, "%.2f": 2, "%.3f": 3, "%d": 0}.get(fmt, 2)
        suffix = "×" if col == "rel_volume_20" else ""
        gb.configure_column(col, header_name=label, headerTooltip=tip,
                            valueFormatter=_js_number(decimals, suffix),
                            type=["numericColumn"], width=(72 if col == "rank" else 110))
    elif kind == "progress":  # a derived [0,1] score (fit is handled above)
        gb.configure_column(col, header_name=label, headerTooltip=tip,
                            valueFormatter=_js_number(2), type=["numericColumn"], width=110)
    elif kind == "checkbox":
        gb.configure_column(col, header_name=label, headerTooltip=tip,
                            valueFormatter=_JS_YESNO, width=100)
    else:  # text (symbol / name / sector / extension badge)
        widths = {"symbol": 88, "name": 150, "sector": 130}
        # Pin the symbol column so it stays visible while scrolling the wider
        # Detailed view (finance-site convention).
        pinned = "left" if col == "symbol" else None
        gb.configure_column(col, header_name=label, headerTooltip=tip,
                            width=widths.get(col, 120), pinned=pinned)


def render_results_grid(table_df: pd.DataFrame, profile) -> "str | None":
    """Render the results table as a double-click-selectable AgGrid.

    Returns the symbol of the double-clicked row (or ``None``). ``table_df``
    already carries the synthetic ``fit`` / link / ``why`` columns (+ the TA badge
    columns app.py appends) in display order; we only attach formatting + the
    double-click selection gesture.
    """
    gb = GridOptionsBuilder.from_dataframe(table_df)
    gb.configure_default_column(sortable=True, filter=False, resizable=True,
                                suppressMovable=True)
    gb.configure_selection("single", use_checkbox=False)
    gb.configure_grid_options(
        suppressRowClickSelection=True,         # a single click does NOT select
        onRowDoubleClicked=_JS_DBLCLICK_SELECT,  # ...a double click does
        rowHeight=30,
    )
    spec = display.column_config_spec(profile)
    for col in table_df.columns:
        _configure_aggrid_column(gb, col, spec.get(col, {"kind": "text", "label": col}))
    n = len(table_df)
    resp = AgGrid(
        table_df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,  # Community edition — no trial watermark
        fit_columns_on_grid_load=False,
        theme="streamlit",
        height=min(540, 56 + 30 * max(1, n)),
        # Key the grid to its column SET: st_aggrid persists client-side column
        # order per widget key, so a fixed key would carry the Compact order over
        # when toggling to Detailed (or switching profiles). A column-derived key
        # remounts the grid on any column change, so each view renders column_order
        # cleanly. Stable within a session for an unchanged column set.
        key=f"results_grid_{abs(hash(tuple(table_df.columns)))}",
    )
    sel = getattr(resp, "selected_rows", None)
    if isinstance(sel, pd.DataFrame):
        if len(sel) and "symbol" in sel.columns:
            return str(sel.iloc[0]["symbol"])
        return None
    if isinstance(sel, list) and sel:
        return sel[0].get("symbol")
    return None
