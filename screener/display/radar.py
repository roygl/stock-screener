"""Pure SVG signal radar (the percentile snowflake) and its short axis labels.

Builds a ``{"labels", "values"}`` spec from a row's ``reasons`` (one axis per
scored signal, valued by percentile) and renders a self-contained, theme-robust
SVG string ``app.py`` drops into an iframe. Pandas/numpy/stdlib only — never
streamlit. Re-exported by :mod:`screener.display`.
"""

from __future__ import annotations

import math

import numpy as np

from .features import feature_label
from ._base import _is_missing
from .reasons import _signal_items


# --- signal radar (pure SVG snowflake) -----------------------------------
# Short axis labels so the radar's spokes don't overlap; anything unmapped falls
# back to the full feature_label.
_RADAR_SHORT_LABELS: "dict[str, str]" = {
    "momentum_1m": "1M Mom",
    "momentum_3m": "3M Mom",
    "momentum_6m": "6M Mom",
    "momentum_12m": "12M Mom",
    "sma_stacked_20_50_150": "Trend",
    "dist_52w_high": "Dist 52wH",
    "forward_pe": "Fwd P/E",
    "revenue_growth": "Rev Grw",
    "earnings_growth": "Earn Grw",
    "ema_5_9_cross_score": "5/9 EMA",
    "rel_volume_20": "Rel Vol",
    "macd_hist": "MACD",
    "rsi_health": "RSI Health",
    "pullback_quality": "Pullback",
    "sector_strength_score": "Sector",
}


def radar_label(feature: str) -> str:
    """Short axis label for the radar (full :func:`feature_label` fallback)."""
    return _RADAR_SHORT_LABELS.get(feature, feature_label(feature))


def _svg_escape(s) -> str:
    """Minimal XML-text escaping for a label rendered inside an SVG ``<text>``."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def radar_spec(reasons, profile=None) -> "dict[str, list]":
    """One radar axis per scored signal: ``{"labels": [...], "values": [...]}``.

    ``values`` are each signal's **percentile** (0..1, where the name sits versus
    the scan) in the ``reasons`` order — the same data the reasons table shows, so
    the radar is a visual TL;DR of it. A missing / non-finite percentile becomes
    the neutral ``0.5`` (matching the engine's missing-signal default). ``profile``
    is accepted for symmetry but unused (the axes come from ``reasons``).
    """
    labels: "list[str]" = []
    values: "list[float]" = []
    for feat, entry in _signal_items(reasons):
        pct = entry.get("percentile")
        labels.append(radar_label(feat))
        if _is_missing(pct):
            values.append(0.5)
            continue
        try:
            v = float(pct)
        except (TypeError, ValueError):
            v = 0.5
        if not np.isfinite(v):
            v = 0.5
        values.append(max(0.0, min(1.0, v)))
    return {"labels": labels, "values": values}


def radar_svg(spec, size: int = 320) -> str:
    """A self-contained SVG radar (snowflake) string for a :func:`radar_spec`.

    Pure string output (NO streamlit) — ``app.py`` drops it into an iframe via
    ``st.components.v1.html``. Uses explicit, theme-robust colours (mid-gray
    structure/labels, a blue accent for the data polygon) since that iframe does
    not inherit Streamlit's theme. Returns ``""`` for an empty / malformed spec or
    a non-positive radius (so the caller renders nothing).
    """
    labels = list(spec.get("labels", [])) if spec else []
    values = list(spec.get("values", [])) if spec else []
    n = len(labels)
    if n == 0 or n != len(values):
        return ""
    cx = cy = size / 2.0
    radius = (size / 2.0) - 60.0
    if radius <= 0:
        return ""

    def _pt(frac, i):
        ang = -math.pi / 2.0 + 2.0 * math.pi * (i / n)
        r = radius * frac
        return (cx + r * math.cos(ang), cy + r * math.sin(ang))

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
        'aria-label="Signal radar: each axis is a signal; further from the centre '
        'means a higher percentile versus the scan.">'
    ]
    # Concentric grid rings.
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_pt(frac, i) for i in range(n)))
        parts.append(
            f'<polygon points="{pts}" fill="none" stroke="#9aa0a6" '
            'stroke-opacity="0.4" stroke-width="1"/>'
        )
    # Spokes + axis labels.
    for i, label in enumerate(labels):
        ex, ey = _pt(1.0, i)
        parts.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            'stroke="#9aa0a6" stroke-opacity="0.4" stroke-width="1"/>'
        )
        lx, ly = _pt(1.15, i)
        anchor = "middle"
        if lx > cx + 1.0:
            anchor = "start"
        elif lx < cx - 1.0:
            anchor = "end"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            'dominant-baseline="middle" font-size="11" font-family="sans-serif" '
            f'fill="#80868b">{_svg_escape(label)}</text>'
        )
    # Data polygon + vertices.
    dpts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_pt(values[i], i) for i in range(n)))
    parts.append(
        f'<polygon points="{dpts}" fill="#4c8bf5" fill-opacity="0.3" '
        'stroke="#4c8bf5" stroke-width="2"/>'
    )
    for i in range(n):
        x, y = _pt(values[i], i)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#4c8bf5"/>')
    parts.append("</svg>")
    return "".join(parts)
