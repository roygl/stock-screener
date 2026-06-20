"""Deterministic, network-free tests for the heatmap engine (screener/sector_heatmap.py).

No ``pytest`` import and no streamlit/network: every test is a plain ``test_*``
function using ``assert`` so the suite runs BOTH under
``python -m pytest tests/test_sector_heatmap.py`` AND standalone as
``python tests/test_sector_heatmap.py`` (the ``__main__`` runner counts pass/fail,
prints a summary, and exits non-zero on any failure).

All frames are built in-memory with ``pd.DataFrame`` so nothing touches the cache,
the provider, or the wall clock; the module under test is pure, so every case is
fully reproducible and the fail-soft paths assert *no raise*.
"""

import os
import sys

# Put the repo root on sys.path so "import screener" resolves when run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from screener import sector_heatmap as sh  # noqa: E402


# --- helpers -------------------------------------------------------------
def _full_scan_frame() -> pd.DataFrame:
    """A realistic post-scan frame: carried sector-constant cols DIFFER from per-row.

    Two Technology rows, two Energy rows, one Healthcare row. The carried
    ``sector_median_3m`` is deliberately NOT equal to the median of the per-row
    ``momentum_3m`` within a sector, so a test can prove the carried value is preferred.

    The momentum magnitudes here are abstract (chosen to make selection/aggregation
    easy to read) — the realistic fraction→percent display convention is covered
    separately by ``test_fmt_pct_treats_momentum_as_fraction`` and the label test.
    """
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "sector": ["Technology", "Technology", "Energy", "Energy", "Healthcare"],
            "score": [0.80, 0.70, 0.40, 0.50, 0.60],
            # Per-row momentum: Tech median = 9.0, Energy median = -4.0.
            "momentum_3m": [12.0, 6.0, -3.0, -5.0, 2.0],
            # Carried universe-wide median: Tech = 99.0, Energy = -42.0 (≠ per-row).
            "sector_median_3m": [99.0, 99.0, -42.0, -42.0, 2.0],
            "change_pct": [1.0, 3.0, -1.0, -3.0, 0.5],
            "market_cap": [3.0e12, 1.0e12, 4.0e11, 2.0e11, 8.0e11],
            "sector_rank": [1.0, 1.0, 3.0, 3.0, 2.0],
            "in_leading_sector": [True, True, False, False, False],
        }
    )


# --- 1. one row per sector + counts --------------------------------------
def test_sector_summary_one_row_per_sector():
    summ = sh.sector_summary(_full_scan_frame())
    assert list(summ.columns) == list(sh.SUMMARY_COLUMNS)
    assert len(summ) == 3, f"expected 3 sectors, got {len(summ)}"
    assert set(summ["sector"]) == {"Technology", "Energy", "Healthcare"}
    counts = dict(zip(summ["sector"], summ["n"]))
    assert counts == {"Technology": 2, "Energy": 2, "Healthcare": 1}
    assert summ["n"].dtype == "int64"
    # Strongest sector first: sector_rank ascending.
    assert list(summ["sector"]) == ["Technology", "Healthcare", "Energy"]


# --- 2. carried sector_median_3m preferred over re-grouped momentum_3m ----
def test_sector_summary_prefers_carried_median():
    summ = sh.sector_summary(_full_scan_frame())
    by_sector = dict(zip(summ["sector"], summ["median_momentum_3m"]))
    # Carried values (99 / -42 / 2) must win over per-row medians (9 / -4 / 2).
    assert by_sector["Technology"] == 99.0
    assert by_sector["Energy"] == -42.0
    assert by_sector["Healthcare"] == 2.0


def test_sector_summary_falls_back_to_grouped_when_no_carried():
    df = _full_scan_frame().drop(columns=["sector_median_3m"])
    summ = sh.sector_summary(df)
    by_sector = dict(zip(summ["sector"], summ["median_momentum_3m"]))
    # No carried column -> grouped median of momentum_3m: Tech (12,6)->9, Energy (-3,-5)->-4.
    assert by_sector["Technology"] == 9.0
    assert by_sector["Energy"] == -4.0
    assert by_sector["Healthcare"] == 2.0


# --- 3. market_cap_sum + fallback ----------------------------------------
def test_market_cap_sum_and_carried_values():
    summ = sh.sector_summary(_full_scan_frame())
    caps = dict(zip(summ["sector"], summ["market_cap_sum"]))
    assert caps["Technology"] == 3.0e12 + 1.0e12
    assert caps["Energy"] == 4.0e11 + 2.0e11
    assert caps["Healthcare"] == 8.0e11
    # carried sector_rank / in_leading_sector survive the aggregation.
    leading = dict(zip(summ["sector"], summ["in_leading_sector"]))
    assert leading["Technology"] is True or leading["Technology"] == True  # noqa: E712
    assert not leading["Energy"]
    assert summ["in_leading_sector"].dtype == "bool"


def test_market_cap_sum_falls_back_to_n_when_absent():
    df = _full_scan_frame().drop(columns=["market_cap"])
    summ = sh.sector_summary(df)
    caps = dict(zip(summ["sector"], summ["market_cap_sum"]))
    # No market_cap column -> size by the row count.
    assert caps["Technology"] == 2.0
    assert caps["Energy"] == 2.0
    assert caps["Healthcare"] == 1.0


def test_mean_change_and_score_aggregated():
    summ = sh.sector_summary(_full_scan_frame())
    mc = dict(zip(summ["sector"], summ["mean_change_pct"]))
    ms = dict(zip(summ["sector"], summ["mean_score"]))
    assert abs(mc["Technology"] - 2.0) < 1e-9   # mean(1, 3)
    assert abs(mc["Energy"] - (-2.0)) < 1e-9     # mean(-1, -3)
    assert abs(ms["Technology"] - 0.75) < 1e-9   # mean(0.80, 0.70)


# --- 4. dropping null / blank sectors ------------------------------------
def test_sector_summary_drops_blank_sector():
    df = _full_scan_frame()
    df.loc[len(df)] = {
        "symbol": "ZZZ", "sector": None, "score": 0.5, "momentum_3m": 1.0,
        "sector_median_3m": 1.0, "change_pct": 0.1, "market_cap": 1.0e11,
        "sector_rank": 4.0, "in_leading_sector": False,
    }
    df.loc[len(df)] = {
        "symbol": "YYY", "sector": "   ", "score": 0.5, "momentum_3m": 1.0,
        "sector_median_3m": 1.0, "change_pct": 0.1, "market_cap": 1.0e11,
        "sector_rank": 5.0, "in_leading_sector": False,
    }
    summ = sh.sector_summary(df)
    assert set(summ["sector"]) == {"Technology", "Energy", "Healthcare"}
    assert len(summ) == 3


# --- 5. fail-soft -> empty, no raise -------------------------------------
def test_sector_summary_empty_frame_is_empty_summary():
    summ = sh.sector_summary(pd.DataFrame())
    assert summ.empty
    assert list(summ.columns) == list(sh.SUMMARY_COLUMNS)


def test_sector_summary_none_is_empty_summary():
    summ = sh.sector_summary(None)
    assert summ.empty
    assert list(summ.columns) == list(sh.SUMMARY_COLUMNS)


def test_sector_summary_missing_sector_column_is_empty():
    # A frame with momentum but no 'sector' column -> well-formed empty, no raise.
    df = pd.DataFrame({"symbol": ["X"], "momentum_3m": [1.0]})
    summ = sh.sector_summary(df)
    assert summ.empty
    assert list(summ.columns) == list(sh.SUMMARY_COLUMNS)


def test_sector_summary_missing_all_momentum_is_empty():
    # Has 'sector' but NEITHER momentum source -> nothing to colour by -> empty.
    df = pd.DataFrame({"symbol": ["X"], "sector": ["Technology"]})
    summ = sh.sector_summary(df)
    assert summ.empty
    assert list(summ.columns) == list(sh.SUMMARY_COLUMNS)


def test_sector_summary_lead_columns_only_still_aggregates():
    # The guaranteed lead columns + momentum_3m (no carried/market_cap) -> still works.
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "name": ["a", "b"],
            "sector": ["Energy", "Energy"],
            "score": [0.3, 0.5],
            "rank": [2, 1],
            "momentum_3m": [-2.0, -6.0],
        }
    )
    summ = sh.sector_summary(df)
    assert len(summ) == 1
    assert summ.iloc[0]["sector"] == "Energy"
    assert summ.iloc[0]["median_momentum_3m"] == -4.0
    assert summ.iloc[0]["market_cap_sum"] == 2.0  # fell back to n


# --- 6. color_for: endpoints, midpoint, clamp, degenerate, NaN ------------
def _rgb(hex_str: str) -> "tuple[int, int, int]":
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def test_color_for_endpoints_and_midpoint():
    vmin, vmax = -10.0, 10.0
    red = sh.color_for(-10.0, vmin, vmax)
    green = sh.color_for(10.0, vmin, vmax)
    neutral = sh.color_for(0.0, vmin, vmax)
    assert red.startswith("#") and len(red) == 7
    r_red, g_red, _ = _rgb(red)
    r_grn, g_grn, _ = _rgb(green)
    # Negative endpoint: red channel dominates green.
    assert r_red > g_red
    # Positive endpoint: green channel dominates red.
    assert g_grn > r_grn
    # Midpoint 0 -> the neutral grey anchor.
    assert neutral == sh._hex(sh._RAMP_MID)


def test_color_for_clamps_beyond_range():
    vmin, vmax = -5.0, 5.0
    # Beyond the endpoints clamps to the endpoint colour.
    assert sh.color_for(-999.0, vmin, vmax) == sh.color_for(-5.0, vmin, vmax)
    assert sh.color_for(999.0, vmin, vmax) == sh.color_for(5.0, vmin, vmax)


def test_color_for_degenerate_range_is_neutral():
    # vmin == vmax -> neutral grey, no divide-by-zero, no raise.
    assert sh.color_for(3.0, 4.0, 4.0) == sh._hex(sh._RAMP_MID)
    assert sh.color_for(0.0, 0.0, 0.0) == sh._hex(sh._RAMP_MID)


def test_color_for_nan_is_neutral():
    assert sh.color_for(float("nan"), -10.0, 10.0) == sh._RAMP_NAN


def test_color_for_monotone_within_half():
    vmin, vmax = -10.0, 10.0
    # Deeper negative -> stronger red (red channel rises toward the red anchor as we
    # move from 0 to vmin; green/blue fall).
    near0 = _rgb(sh.color_for(-1.0, vmin, vmax))
    deep = _rgb(sh.color_for(-9.0, vmin, vmax))
    assert deep[0] >= near0[0]   # red channel grows toward the red anchor
    assert deep[1] <= near0[1]   # green channel shrinks


# --- 7. treemap_svg ------------------------------------------------------
def test_treemap_svg_basic_shape():
    summ = sh.sector_summary(_full_scan_frame())
    svg = sh.treemap_svg(summ)
    assert isinstance(svg, str)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    # One <rect> per sector.
    assert svg.count("<rect") == len(summ) == 3
    # Each sector name appears (XML-escaped form).
    for name in summ["sector"]:
        assert sh._svg_escape(name) in svg


def test_treemap_svg_escapes_sector_names():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "sector": ["Foo & <Bar>", "Health"],
            "score": [0.5, 0.6],
            "momentum_3m": [5.0, -5.0],
            "sector_median_3m": [5.0, -5.0],
            "market_cap": [2.0e12, 1.0e12],
            "sector_rank": [1.0, 2.0],
            "in_leading_sector": [True, False],
        }
    )
    svg = sh.treemap_svg(df.pipe(sh.sector_summary))
    # Raw ampersand/angle brackets must be escaped in the output.
    assert "Foo &amp; &lt;Bar&gt;" in svg
    assert "Foo & <Bar>" not in svg


def test_treemap_svg_area_roughly_proportional():
    # A single dominant sector should consume far more tile area than a tiny one.
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "sector": ["Big", "Small"],
            "score": [0.5, 0.5],
            "momentum_3m": [1.0, 1.0],
            "sector_median_3m": [1.0, 1.0],
            "market_cap": [9.0e12, 1.0e11],
            "sector_rank": [1.0, 2.0],
            "in_leading_sector": [True, False],
        }
    )
    summ = sh.sector_summary(df)
    rects = sh._squarify(summ["market_cap_sum"].tolist(), 0.0, 0.0, 1000.0, 520.0)
    areas = {sec: w * h for sec, (x, y, w, h) in zip(summ["sector"], rects)}
    # 'Big' is ~90x the cap of 'Small' -> dominant area share (sanity, not pixel-exact).
    assert areas["Big"] > areas["Small"] * 50


def test_treemap_svg_signed_percent_label_present():
    # momentum_3m is a FRACTION in the engine (0.182 == +18.2%, per display/_base.py),
    # so the tile label must show the CONVERTED percent, not the bare fraction.
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "sector": ["Technology", "Energy"],
            "score": [0.8, 0.4],
            "momentum_3m": [0.182, -0.091],
            "sector_median_3m": [0.182, -0.091],
            "market_cap": [3.0e12, 5.0e11],
            "sector_rank": [1.0, 2.0],
            "in_leading_sector": [True, False],
        }
    )
    svg = sh.treemap_svg(sh.sector_summary(df))
    assert "+18.2%" in svg
    assert "-9.1%" in svg


def test_fmt_pct_treats_momentum_as_fraction():
    # Guards the fraction->percent convention (the bug live-preview caught): 0.12 must
    # render "+12.0%", NOT "+0.1%". % format multiplies by 100; a bare .1f would not.
    assert sh._fmt_pct(0.12) == "+12.0%"
    assert sh._fmt_pct(-0.091) == "-9.1%"
    assert sh._fmt_pct(0.0) == "+0.0%"
    assert sh._fmt_pct(float("nan")) == "n/a"


def test_treemap_svg_empty_summary_is_placeholder():
    svg = sh.treemap_svg(sh._empty_summary())
    assert isinstance(svg, str)
    assert svg.startswith("<svg")
    assert "No sector data" in svg
    assert "<rect" not in svg


def test_treemap_svg_none_is_placeholder_no_raise():
    svg = sh.treemap_svg(None)
    assert svg.startswith("<svg")
    assert "No sector data" in svg


def test_squarify_covers_box_and_preserves_order():
    areas = [4.0, 2.0, 1.0, 1.0]
    rects = sh._squarify(areas, 0.0, 0.0, 800.0, 400.0)
    assert len(rects) == len(areas)
    total = sum(w * h for (x, y, w, h) in rects)
    # Tiles tile the whole box (within float tolerance).
    assert abs(total - 800.0 * 400.0) < 1.0
    # All rectangles lie inside the box.
    for (x, y, w, h) in rects:
        assert x >= -1e-6 and y >= -1e-6
        assert x + w <= 800.0 + 1e-6
        assert y + h <= 400.0 + 1e-6
    # Larger weight -> larger area (first tile is the biggest).
    first_area = rects[0][2] * rects[0][3]
    last_area = rects[-1][2] * rects[-1][3]
    assert first_area > last_area


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
