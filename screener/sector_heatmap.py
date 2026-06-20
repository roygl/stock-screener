"""Sector heatmap (pure, offline) — aggregation, colour ramp, squarified treemap.

A descriptive overlay on the cached scan: it collapses the per-row scan frame
(``st.session_state["scan"]["df"]``) into ONE ROW PER SECTOR, then renders a
squarified treemap where each tile is **sized by combined market cap** and
**coloured by 3-month sector momentum** (red weak → neutral → green strong). It
exists to surface *where the rotation is*, not to issue buy/sell guidance.

Everything here is pure: pandas/stdlib only, no streamlit and no network at
import or call time. Like :mod:`screener.calendar`, the public functions are
deterministic (inputs come in as args; nothing reads globals or the wall clock)
and FAIL-SOFT — a ``None``/empty/missing-column frame yields a well-formed empty
summary or a valid placeholder ``<svg>`` rather than raising, so the optional
panel never crashes the app. The SVG carries explicit theme-robust hex colours
(it renders inside an iframe that doesn't inherit Streamlit's theme), mirroring
:mod:`screener.display.radar`.
"""

from __future__ import annotations

import math

import pandas as pd

# Canonical summary schema (column order preserved on the returned frame), mirroring
# calendar.EVENT_COLUMNS. One row per sector.
SUMMARY_COLUMNS = (
    "sector",
    "n",
    "median_momentum_3m",
    "mean_change_pct",
    "mean_score",
    "sector_rank",
    "market_cap_sum",
    "in_leading_sector",
)

# Diverging ramp anchors (explicit hex so the treemap is theme-robust inside its
# iframe). Red = weak momentum, neutral grey = ~0, green = strong.
_RAMP_NEG = (0xD6, 0x60, 0x4D)   # #d6604d  (red, at vmin)
_RAMP_MID = (0x9E, 0x9E, 0x9E)   # #9e9e9e  (neutral grey, at 0)
_RAMP_POS = (0x1A, 0x98, 0x50)   # #1a9850  (green, at vmax)
_RAMP_NAN = "#bdbdbd"            # NaN / undefined momentum -> a lighter neutral grey


def _empty_summary() -> pd.DataFrame:
    """An empty, well-formed sector summary with the canonical columns + dtypes.

    Mirrors :func:`screener.calendar._empty_events`: a zero-row frame the UI can
    treat exactly like a populated one (``.empty`` is ``True``, columns/dtypes are
    stable) so callers never special-case the no-data path beyond ``if summary.empty``.
    """
    return pd.DataFrame(
        {
            "sector": pd.Series([], dtype="object"),
            "n": pd.Series([], dtype="int64"),
            "median_momentum_3m": pd.Series([], dtype="float64"),
            "mean_change_pct": pd.Series([], dtype="float64"),
            "mean_score": pd.Series([], dtype="float64"),
            "sector_rank": pd.Series([], dtype="float64"),
            "market_cap_sum": pd.Series([], dtype="float64"),
            "in_leading_sector": pd.Series([], dtype="bool"),
        }
    )


def _first_non_null(series: pd.Series):
    """First non-null value in ``series`` (a sector-constant carried value), else NaN."""
    nn = series.dropna()
    return nn.iloc[0] if len(nn) else float("nan")


def sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the cached scan frame into one row per sector.

    Returns a frame with the :data:`SUMMARY_COLUMNS`: ``sector``, ``n`` (row count),
    ``median_momentum_3m``, ``mean_change_pct``, ``mean_score``, ``sector_rank``,
    ``market_cap_sum`` and ``in_leading_sector``. Aggregation rules:

    * ``median_momentum_3m`` PREFERS the carried universe-wide ``sector_median_3m``
      (constant within a sector → take the first non-null per sector) over re-grouping
      the post-filter ``momentum_3m``; it falls back to the grouped median of
      ``momentum_3m`` only when ``sector_median_3m`` is absent.
    * ``market_cap_sum`` sums ``market_cap`` per sector; if that column is absent or
      all-NaN it falls back to ``n`` so tiles can still be sized.
    * ``sector_rank`` and ``in_leading_sector`` carry the per-sector value (first
      non-null), since they are constant within a sector.

    Rows with a null/blank ``sector`` are dropped. The result is sorted by
    ``sector_rank`` ascending (then ``median_momentum_3m`` descending) so the strongest
    sector is first. FAIL-SOFT: a ``None``/empty frame, or one lacking BOTH ``sector``
    and any momentum source, yields :func:`_empty_summary` rather than raising. Every
    column access is guarded (``sector_median_3m``/``market_cap``/``change_pct`` may be
    ABSENT on a wholly-empty scan).
    """
    if df is None or len(df) == 0 or "sector" not in df.columns:
        return _empty_summary()
    has_carried = "sector_median_3m" in df.columns
    has_grouped = "momentum_3m" in df.columns
    if not has_carried and not has_grouped:
        # No momentum source at all -> nothing meaningful to colour by.
        return _empty_summary()

    work = df.copy()
    # Drop null/blank sectors (treat "" / whitespace as missing, like the universe).
    sect = work["sector"].astype("object")
    blank = sect.isna() | sect.map(lambda s: str(s).strip() == "")
    work = work[~blank].copy()
    if work.empty:
        return _empty_summary()
    work["sector"] = work["sector"].map(lambda s: str(s).strip())

    grouped = work.groupby("sector", sort=False)
    rows: "list[dict]" = []
    for sector_name, g in grouped:
        n = int(len(g))
        # Momentum: prefer the carried sector-constant value; fall back to the grouped
        # median of the (post-filter) per-row momentum only when carried is absent.
        if has_carried and g["sector_median_3m"].notna().any():
            median_mom = float(_first_non_null(g["sector_median_3m"]))
        elif has_grouped and g["momentum_3m"].notna().any():
            # Guard notna() so an all-NaN slice (a sector whose every member lacks a
            # 3-month value) doesn't emit numpy's "Mean of empty slice" RuntimeWarning;
            # the result is the same NaN we fall through to below.
            median_mom = float(g["momentum_3m"].median())
        else:
            median_mom = float("nan")

        mean_change = (
            float(g["change_pct"].mean()) if "change_pct" in g.columns else float("nan")
        )
        mean_score = (
            float(g["score"].mean()) if "score" in g.columns else float("nan")
        )
        sector_rank = (
            float(_first_non_null(g["sector_rank"]))
            if "sector_rank" in g.columns
            else float("nan")
        )
        # market_cap_sum: sum the per-row caps; fall back to the row count when the
        # column is absent or yields no finite total (so tiles can still be sized).
        if "market_cap" in g.columns:
            cap_total = float(g["market_cap"].sum(skipna=True))
            market_cap_sum = cap_total if math.isfinite(cap_total) and cap_total > 0 else float(n)
        else:
            market_cap_sum = float(n)
        if "in_leading_sector" in g.columns:
            lead = _first_non_null(g["in_leading_sector"])
            in_leading = bool(lead) if lead == lead else False  # NaN-safe
        else:
            in_leading = False

        rows.append(
            {
                "sector": sector_name,
                "n": n,
                "median_momentum_3m": median_mom,
                "mean_change_pct": mean_change,
                "mean_score": mean_score,
                "sector_rank": sector_rank,
                "market_cap_sum": market_cap_sum,
                "in_leading_sector": in_leading,
            }
        )

    out = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    # Stable dtypes regardless of the per-row content.
    out["n"] = out["n"].astype("int64")
    for col in ("median_momentum_3m", "mean_change_pct", "mean_score", "sector_rank", "market_cap_sum"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["in_leading_sector"] = out["in_leading_sector"].astype("bool")
    # Strongest sector first: sector_rank asc (NaN ranks sink), then momentum desc.
    out = out.sort_values(
        by=["sector_rank", "median_momentum_3m"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    return out


def _lerp_channel(a: int, b: int, t: float) -> int:
    """Linearly interpolate one 0–255 channel from ``a`` to ``b`` at fraction ``t``."""
    return int(round(a + (b - a) * t))


def _hex(rgb: "tuple[int, int, int]") -> str:
    """Format an (r, g, b) triple (each 0–255) as ``#rrggbb``."""
    r, g, b = (max(0, min(255, c)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def color_for(value: float, vmin: float, vmax: float) -> str:
    """Diverging red→neutral→green hex for a momentum ``value`` in ``[vmin, vmax]``.

    The value is CLAMPED to ``[vmin, vmax]``; the negative half (``vmin``→0) ramps
    red→neutral grey and the positive half (0→``vmax``) ramps neutral grey→green, with
    the neutral midpoint pinned at exactly 0 (so a symmetric range puts grey at the
    middle). Returns an explicit ``#rrggbb`` (theme-robust, like
    :mod:`screener.display.radar`). Degenerate ``vmin == vmax`` and a NaN ``value`` both
    map to a neutral grey rather than raising or dividing by zero.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _RAMP_NAN
    if v != v:  # NaN
        return _RAMP_NAN
    try:
        lo = float(vmin)
        hi = float(vmax)
    except (TypeError, ValueError):
        return _hex(_RAMP_MID)
    if not (hi > lo):  # degenerate or inverted range -> neutral
        return _hex(_RAMP_MID)
    # Clamp into range.
    v = max(lo, min(hi, v))
    if v >= 0.0:
        # 0 -> grey, hi -> green. If hi <= 0 the whole positive half collapses to grey.
        t = (v / hi) if hi > 0.0 else 0.0
        t = max(0.0, min(1.0, t))
        rgb = tuple(_lerp_channel(_RAMP_MID[i], _RAMP_POS[i], t) for i in range(3))
    else:
        # lo -> red, 0 -> grey. lo < 0 here (range is valid and v < 0).
        t = v / lo  # both negative -> t in (0, 1]; nearer lo -> nearer red
        t = max(0.0, min(1.0, t))
        rgb = tuple(_lerp_channel(_RAMP_MID[i], _RAMP_NEG[i], t) for i in range(3))
    return _hex(rgb)  # type: ignore[arg-type]


def _svg_escape(s) -> str:
    """Minimal XML-text escaping for a label inside an SVG ``<text>`` (``&``/``<``/``>``)."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_pct(value: float) -> str:
    """Signed percent label for a momentum value (e.g. ``+4.2%``); ``n/a`` for NaN.

    Momentum is stored as a FRACTION in the engine (``0.12`` == 12%, per
    ``display/_base.py``), so format with ``%`` (which multiplies by 100) — not a
    bare ``.1f`` + "%", which would render a real +20% move as a misleading "+0.2%".
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:+.1%}"


def _placeholder_svg(width: int, height: int, message: str = "No sector data") -> str:
    """A valid, centred-message ``<svg>`` for the empty/degenerate case (never ``""``)."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_svg_escape(message)}">'
        f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
        'dominant-baseline="middle" font-size="16" font-family="sans-serif" '
        f'fill="#80868b">{_svg_escape(message)}</text></svg>'
    )


# --- squarified treemap (Bruls / Huizing / van Wijk) ---------------------
def _worst_ratio(row_areas: "list[float]", side: float, total_area: float, total_side: float) -> float:
    """Worst (max) aspect ratio of the rectangles a ``row`` would produce.

    ``row_areas`` are the (already-scaled) areas laid along a strip of length ``side``.
    Mirrors the classic ``worst`` function: ``max(w^2*r / s^2, s^2 / (w^2*r))`` where
    ``s`` is the sum of the row's areas and ``w`` the strip length.
    """
    s = sum(row_areas)
    if s <= 0 or side <= 0:
        return float("inf")
    rmax = max(row_areas)
    rmin = min(row_areas)
    w2 = side * side
    return max((w2 * rmax) / (s * s), (s * s) / (w2 * rmin))


def _squarify(areas: "list[float]", x: float, y: float, width: float, height: float) -> "list[tuple[float, float, float, float]]":
    """Squarified treemap layout: place ``areas`` into the box, return (x, y, w, h) each.

    A plain-Python implementation of the Bruls/Huizing/van Wijk algorithm. ``areas`` are
    raw weights (any positive scale); they are normalised to the box area internally. The
    returned rectangles are in the SAME order as ``areas``. Non-positive / non-finite
    weights are floored to a tiny epsilon so every input still yields a (vanishingly
    small) tile rather than disappearing or breaking the layout.
    """
    eps = 1e-9
    clean = [a if (a == a and a != float("inf") and a > 0) else eps for a in areas]
    total = sum(clean)
    box_area = max(width * height, eps)
    scale = box_area / total if total > 0 else 0.0
    scaled = [a * scale for a in clean]

    rects: "list[tuple[float, float, float, float]]" = []
    # Remaining free sub-rectangle.
    rx, ry, rw, rh = float(x), float(y), float(width), float(height)
    i = 0
    n = len(scaled)
    while i < n:
        # Length of the shorter side -> the strip we lay the current row along.
        side = min(rw, rh)
        if side <= eps:
            # Degenerate remaining box: dump the rest as zero-area tiles at the corner.
            for a in scaled[i:]:
                rects.append((rx, ry, 0.0, 0.0))
            break
        row: "list[float]" = [scaled[i]]
        i += 1
        # Greedily extend the row while it keeps (or improves) the worst aspect ratio.
        while i < n:
            if _worst_ratio(row + [scaled[i]], side, box_area, side) <= _worst_ratio(row, side, box_area, side):
                row.append(scaled[i])
                i += 1
            else:
                break
        # Lay the finalised row across the shorter side and carve it off the box.
        row_sum = sum(row)
        if rw <= rh:
            # Strip spans the full width; its height is row_sum / width.
            strip_h = row_sum / rw if rw > eps else 0.0
            cx = rx
            for a in row:
                tile_w = (a / row_sum) * rw if row_sum > eps else 0.0
                rects.append((cx, ry, tile_w, strip_h))
                cx += tile_w
            ry += strip_h
            rh -= strip_h
        else:
            # Strip spans the full height; its width is row_sum / height.
            strip_w = row_sum / rh if rh > eps else 0.0
            cy = ry
            for a in row:
                tile_h = (a / row_sum) * rh if row_sum > eps else 0.0
                rects.append((rx, cy, strip_w, tile_h))
                cy += tile_h
            rx += strip_w
            rw -= strip_w
    return rects


def treemap_svg(summary: pd.DataFrame, *, width: int = 1000, height: int = 520) -> str:
    """A squarified sector treemap as a complete ``<svg>`` string.

    Tiles are **sized by ``market_cap_sum``** (falling back to ``n`` when the cap sum is
    missing/non-positive) and **coloured by ``median_momentum_3m``** via
    :func:`color_for`, with ``vmin``/``vmax`` taken from the summary's momentum range and
    made symmetric around 0 so the neutral midpoint is exactly 0. Each tile is a
    ``<rect>`` (fill = colour, thin separating stroke) plus a ``<text>`` label with the
    sector name and the median momentum as a signed percent (e.g. ``+4.2%``); labels on
    tiles too small to fit the text are skipped. Pure (pandas/stdlib only). An empty or
    wholly-degenerate summary returns a valid placeholder ``<svg>`` with a centred
    "No sector data" message — it NEVER raises and NEVER returns ``""``.
    """
    if summary is None or len(summary) == 0 or "sector" not in summary.columns:
        return _placeholder_svg(width, height)

    sectors = [str(s) for s in summary["sector"].tolist()]
    # Sizes: prefer market_cap_sum, fall back to n, floor at a tiny positive epsilon.
    if "market_cap_sum" in summary.columns:
        raw_sizes = pd.to_numeric(summary["market_cap_sum"], errors="coerce").tolist()
    else:
        raw_sizes = [float("nan")] * len(sectors)
    n_fallback = (
        pd.to_numeric(summary["n"], errors="coerce").tolist()
        if "n" in summary.columns
        else [1.0] * len(sectors)
    )
    sizes: "list[float]" = []
    for cap, n in zip(raw_sizes, n_fallback):
        val = cap if (cap == cap and cap not in (float("inf"),) and cap and cap > 0) else None
        if val is None:
            val = n if (n == n and n and n > 0) else 1.0
        sizes.append(float(val))

    if "median_momentum_3m" in summary.columns:
        moms = pd.to_numeric(summary["median_momentum_3m"], errors="coerce").tolist()
    else:
        moms = [float("nan")] * len(sectors)

    if sum(sizes) <= 0:
        return _placeholder_svg(width, height)

    # Symmetric colour range around 0 from the finite momentum values.
    finite = [m for m in moms if m == m]
    extent = max((abs(m) for m in finite), default=0.0)
    if extent <= 0:
        extent = 1.0  # avoid a zero-width ramp; everything reads near-neutral
    vmin, vmax = -extent, extent

    rects = _squarify(sizes, 0.0, 0.0, float(width), float(height))

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        'aria-label="Sector treemap: tile size is combined market cap, colour is '
        '3-month sector momentum (red weak to green strong).">'
    ]
    for sector_name, (rx, ry, rw, rh), mom in zip(sectors, rects, moms):
        fill = color_for(mom, vmin, vmax)
        parts.append(
            f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{max(rw, 0.0):.2f}" '
            f'height="{max(rh, 0.0):.2f}" fill="{fill}" stroke="#ffffff" '
            'stroke-width="1.5" shape-rendering="crispEdges"/>'
        )
        # Labels: only when the tile is big enough to plausibly hold the text, so small
        # tiles aren't littered with clipped/overlapping strings.
        name_label = _svg_escape(sector_name)
        pct_label = _fmt_pct(mom)
        cx = rx + rw / 2.0
        cy = ry + rh / 2.0
        # Rough fit test: need room for the name (~7px/char at 13px) and a second line.
        fits_name = rw >= 56.0 and rh >= 34.0 and rw >= 7.0 * len(sector_name) * 0.62
        if fits_name:
            parts.append(
                f'<text x="{cx:.1f}" y="{cy - 6:.1f}" text-anchor="middle" '
                'dominant-baseline="middle" font-size="13" font-family="sans-serif" '
                f'font-weight="600" fill="#ffffff">{name_label}</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{cy + 11:.1f}" text-anchor="middle" '
                'dominant-baseline="middle" font-size="12" font-family="sans-serif" '
                f'fill="#ffffff" fill-opacity="0.92">{pct_label}</text>'
            )
        elif rw >= 34.0 and rh >= 18.0:
            # Tight tile: drop the name, keep just the signed percent.
            parts.append(
                f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
                'dominant-baseline="middle" font-size="11" font-family="sans-serif" '
                f'fill="#ffffff" fill-opacity="0.9">{pct_label}</text>'
            )
        # else: tile too small for any legible label -> skip (clip).
    parts.append("</svg>")
    return "".join(parts)


if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "sector": ["Technology", "Technology", "Energy", "Healthcare", "Energy"],
            "score": [0.8, 0.7, 0.4, 0.6, 0.5],
            "momentum_3m": [12.0, 8.0, -3.0, 2.0, -5.0],
            "sector_median_3m": [10.0, 10.0, -4.0, 2.0, -4.0],
            "change_pct": [1.2, 0.8, -0.4, 0.3, -0.6],
            "market_cap": [3.0e12, 1.2e12, 4.0e11, 8.0e11, 2.0e11],
            "sector_rank": [1.0, 1.0, 3.0, 2.0, 3.0],
            "in_leading_sector": [True, True, False, False, False],
        }
    )
    summ = sector_summary(demo)
    print(summ.to_string(index=False))
    print(f"\nSVG length: {len(treemap_svg(summ))} chars")
