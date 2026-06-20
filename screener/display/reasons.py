"""The "why it ranks" reasons table and per-row plain-English narrative.

Turns the engine's per-row ``reasons`` OrderedDict into a tidy display frame, the
contribution-sum caption, the strongest/weakest narrative phrasing (standalone and
as a ranked-line sentence), and the per-profile signal glossary. Purely
descriptive — never advice. Pandas/numpy/stdlib only — never streamlit.
Re-exported by :mod:`screener.display`.
"""

from __future__ import annotations

import pandas as pd

from ..profiles import Profile
from ._base import _is_missing
from .features import feature_description, feature_label
from .formatting import format_value


# --- the "why it ranks" reasons table ------------------------------------
def _signal_items(reasons) -> "list[tuple[str, dict]]":
    """The reasons OrderedDict's signal entries (the ``"flags"`` key excluded).

    Preserves insertion order — the engine writes signals in the profile's
    signal order and the engine test asserts it, so we never re-sort.
    """
    if not reasons:
        return []
    return [(k, v) for k, v in reasons.items() if k != "flags" and isinstance(v, dict)]


def reasons_to_frame(reasons, profile: Profile) -> pd.DataFrame:
    """Tidy the per-row ``reasons`` OrderedDict into a display frame.

    Columns ``Signal`` (humanized), ``What it measures`` (plain-English
    definition via :func:`feature_description`), ``Value`` (via
    :func:`format_value`), ``Percentile`` (float 0..1), ``Contribution`` (float
    0..1), in the OrderedDict's signal order, excluding the ``"flags"`` key. The
    numeric Percentile/Contribution stay numeric so ``app.py`` can render them as
    progress bars; ``Value`` is the pre-formatted string. Tolerant of
    ``NaN``/``None`` values and an empty/``None`` ``reasons`` (-> empty frame
    with the right columns).
    """
    columns = ["Signal", "What it measures", "Value", "Percentile", "Contribution"]
    items = _signal_items(reasons)
    if not items:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    rows = []
    for feat, entry in items:
        pct = entry.get("percentile")
        contrib = entry.get("contribution")
        rows.append(
            {
                "Signal": feature_label(feat),
                "What it measures": feature_description(feat),
                "Value": format_value(feat, entry.get("value")),
                "Percentile": float(pct) if not _is_missing(pct) else float("nan"),
                "Contribution": float(contrib) if not _is_missing(contrib) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def max_contribution(reasons) -> float:
    """Largest signal contribution (excl ``flags``), clamped strictly ``> 0``.

    Used as the max of the Contribution progress bar so small contributions are
    still visible (a hardcoded 1.0 max would make them look empty). Defaults to a
    small positive number when there are no contributions or all are zero, so the
    progress column never divides by zero.
    """
    items = _signal_items(reasons)
    best = 0.0
    for _, entry in items:
        c = entry.get("contribution")
        if not _is_missing(c):
            c = float(c)
            if c > best:
                best = c
    return best if best > 0.0 else 0.01


def contribution_caption(reasons, score: float) -> str:
    """Assert-the-math caption: per-signal contributions sum to the score.

    e.g. ``"Signal contributions sum to the score (0.719 ≈ 0.719)"``. Sums the
    contributions (excl ``flags``); a missing contribution counts as 0.
    """
    items = _signal_items(reasons)
    total = 0.0
    for _, entry in items:
        c = entry.get("contribution")
        if not _is_missing(c):
            total += float(c)
    score_val = 0.0 if _is_missing(score) else float(score)
    return f"Signal contributions sum to the score ({total:.3f} ≈ {score_val:.3f})"


def _highlight_clause(reasons) -> str:
    """The "strongest on … (weakest on …)" clause for a row's ``reasons``.

    Ranks the signals by **percentile** (where the name genuinely sits versus the
    scan), names the top 1–2 and — when distinct — the single weakest. Lowercase,
    no leading subject/period, so both :func:`explain_rank` (which prepends a head)
    and :func:`narrative` (which capitalizes it) can reuse it. ``""`` when there is
    no scored signal. Purely descriptive — never advice.
    """
    scored = [
        (feat, float(entry.get("percentile")))
        for feat, entry in _signal_items(reasons)
        if not _is_missing(entry.get("percentile"))
    ]
    if not scored:
        return ""
    by_pct = sorted(scored, key=lambda kv: kv[1], reverse=True)
    top = by_pct[: min(2, len(by_pct))]
    clause = "strongest on " + " and ".join(feature_label(f) for f, _ in top)
    weak_feat = by_pct[-1][0]
    if weak_feat not in {f for f, _ in top}:
        clause += f", weakest on {feature_label(weak_feat)}"
    return clause


def explain_rank(row, reasons, profile=None, total=None) -> str:
    """One descriptive sentence: why this row ranks where it does.

    Names the 1–2 signals the stock stands strongest on and the single weakest,
    ranked by **percentile** (where it genuinely sits versus the scan), e.g.
    ``"AFL ranks #2 of 50 — strongest on Earnings Growth and Revenue Growth, "``
    ``"weakest on 12M Momentum."``. Returns ``""`` when there is nothing useful to
    say (the caller then renders nothing). Purely descriptive — never advice.

    ``row`` is the result row (needs ``symbol`` / ``rank``); ``profile`` is
    accepted for signature symmetry but unused; ``total`` (e.g. ``len(df)``) adds
    the "of N" when given.
    """
    symbol = "" if row is None else str(row.get("symbol", "") or "").strip()
    rank = None if row is None else row.get("rank")
    head = symbol or "This stock"
    if not _is_missing(rank):
        head += f" ranks #{int(rank)}"
        if total:
            head += f" of {int(total)}"

    clause = _highlight_clause(reasons)
    if not clause:
        # No signal detail — only worth a line if we at least have a rank.
        return f"{head}." if not _is_missing(rank) else ""
    return f"{head} — {clause}."


def narrative(reasons) -> str:
    """A standalone per-row "why" phrase from ``reasons`` (``""`` when none).

    The same strongest/weakest read as :func:`explain_rank` but WITHOUT the
    rank/symbol head — capitalized and full-stopped so it stands alone in a table
    cell (e.g. ``"Strongest on Earnings Growth and Revenue Growth, weakest on 12M
    Momentum."``). Descriptive only — never advice.
    """
    clause = _highlight_clause(reasons)
    if not clause:
        return ""
    return f"{clause[:1].upper()}{clause[1:]}."


def narrative_series(df: pd.DataFrame, profile=None) -> pd.Series:
    """Vectorized :func:`narrative` over a frame's ``reasons`` column.

    Returns an all-empty-string Series (indexed like ``df``) when ``reasons`` is
    absent. ``profile`` is accepted for signature symmetry but unused (the
    narrative reads only the per-row ``reasons``).
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype="object")
    if "reasons" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return pd.Series(
        [narrative(r) for r in df["reasons"]], index=df.index, dtype="object"
    )


def signal_glossary(profile: Profile) -> "list[tuple[str, str]]":
    """``(label, description)`` per signal in ``profile``, in signal order.

    Feeds the "How to read this" expander. Reuses :func:`feature_label` /
    :func:`feature_description`, inheriting their safe fallbacks.
    """
    if profile is None or not getattr(profile, "signals", None):
        return []
    return [
        (feature_label(s.feature), feature_description(s.feature)) for s in profile.signals
    ]
