"""Declarative profile configs — each investing *style* as data, not code.

A profile (spec §5) is the *lens* the generic engine looks through: a set of
HARD filters (cutoffs a name must clear to appear at all) plus WEIGHTED signals
(the inputs the cross-sectional percentile ranker scores it on). Same engine,
different config — adding a variable to a style is a one-line edit here, never a
change to :mod:`screener.engine` (spec §5: "New variables get added to a profile
config, not hard-coded into the engine").

Two rules keep these configs purely declarative:
- **The scorer only ever sees ``direction`` "higher" or "lower".** Any "banded"
  logic — RSI overbought, a healthy pullback, the freshness of a 5/9 cross — is
  pre-baked by the engine into a higher-is-better DERIVED feature in ``[0, 1]``
  (``rsi_health``, ``pullback_quality``, ``ema_5_9_cross_score``,
  ``sector_strength_score``). So a config never carries a curve or a band, only
  ``feature -> weight -> direction``.
- **Weights need not sum to 1.** The scorer normalizes (``weight / sum(weights)``)
  before summing, so a config can be tuned by editing one number without
  rebalancing the rest.

Frozen dataclasses (mirroring the provider/indicator style — small, immutable,
self-documenting): :class:`SignalSpec`, :class:`Filter`, :class:`Profile`. The
three shipped styles plus the :data:`PROFILES` registry and :func:`get_profile`
live at the bottom; the feature keys they reference are the 21 ``snapshot()``
keys, the fundamentals columns, and the engine's derived columns — all assembled
into one row per ticker by :func:`screener.engine.assemble_features`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Filter operators understood by :func:`screener.engine.apply_filters`. "is_true"
# passes only on a Python ``True`` (so a ``None`` price-above-MA flag fails closed).
FILTER_OPS = frozenset({">", ">=", "<", "<=", "==", "is_true"})

# Direction of a signal as seen by the percentile ranker. "higher" -> bigger is
# better (use the raw percentile); "lower" -> smaller is better (invert: 1 - pct).
SIGNAL_DIRECTIONS = frozenset({"higher", "lower"})


@dataclass(frozen=True)
class SignalSpec:
    """One weighted ranking input: a feature, its weight, and its direction.

    ``feature``   — a column produced by :func:`screener.engine.assemble_features`
                    (a ``snapshot()`` key, a fundamentals field, or a derived one).
    ``weight``    — relative importance; the scorer normalizes across a profile's
                    signals, so weights need not sum to 1.
    ``direction`` — ``"higher"`` (bigger ranks better) or ``"lower"`` (smaller
                    ranks better, e.g. ``forward_pe``). The scorer never sees a
                    band; banded ideas are pre-baked into a ``"higher"`` derived
                    feature by the engine.
    """

    feature: str
    weight: float
    direction: str = "higher"

    def __post_init__(self) -> None:
        if self.direction not in SIGNAL_DIRECTIONS:
            raise ValueError(
                f"SignalSpec.direction must be one of {sorted(SIGNAL_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )


@dataclass(frozen=True)
class Filter:
    """A hard cutoff: a row survives only if ``feature op threshold`` holds.

    ``op`` is one of :data:`FILTER_OPS`. ``"is_true"`` ignores ``threshold`` and
    passes only on a Python ``True``. A missing / ``NaN`` / ``None`` value FAILS
    CLOSED (the row is dropped) — see DECISIONS.md and
    :func:`screener.engine.apply_filters`.
    """

    feature: str
    op: str
    threshold: "float | None" = None

    def __post_init__(self) -> None:
        if self.op not in FILTER_OPS:
            raise ValueError(
                f"Filter.op must be one of {sorted(FILTER_OPS)}, got {self.op!r}"
            )


@dataclass(frozen=True)
class Profile:
    """A named investing style: hard ``filters`` + weighted ``signals`` (+ ``flags``).

    ``flags`` are feature keys surfaced as informational BADGES (e.g.
    ``earnings_in_window``), not cutoffs and not scored — the engine carries them
    into the result and the reason breakdown so the UI (M5) can show them.
    ``label`` is the human-facing name for that UI.
    """

    name: str
    label: str
    filters: "tuple[Filter, ...]" = ()
    signals: "tuple[SignalSpec, ...]" = ()
    flags: "tuple[str, ...]" = ()


# --- the three shipped styles (spec §5) ----------------------------------
# Weights are a sensible starting point and are meant to be tuned in-config; the
# scorer normalizes them, so they need not sum to 1.

LONG_TERM = Profile(
    name="long_term",
    label="Long-Term",
    filters=(
        Filter("forward_pe", ">", 0.0),          # must have a positive forward P/E
        Filter("price_above_sma_150", "is_true"),  # in a long-term uptrend
    ),
    signals=(
        SignalSpec("forward_pe", 0.20, "lower"),           # cheaper is better
        SignalSpec("revenue_growth", 0.18, "higher"),
        SignalSpec("earnings_growth", 0.15, "higher"),
        SignalSpec("sma_stacked_20_50_150", 0.17, "higher"),  # clean trend template
        SignalSpec("dist_52w_high", 0.15, "higher"),          # nearer the high (less negative)
        SignalSpec("momentum_12m", 0.15, "higher"),
    ),
)

SWING = Profile(
    name="swing",
    label="Swing",
    filters=(
        Filter("rel_volume_20", ">", 2.0),          # spec §5: relative volume > 2x
        Filter("in_leading_sector", "is_true"),     # a top-3 sector by 3-mo median
    ),
    signals=(
        SignalSpec("ema_5_9_cross_score", 0.25, "higher"),  # fresh bullish 5/9 cross
        SignalSpec("rel_volume_20", 0.15, "higher"),
        SignalSpec("macd_hist", 0.15, "higher"),
        SignalSpec("rsi_health", 0.10, "higher"),           # strong but not overbought
        SignalSpec("pullback_quality", 0.20, "higher"),     # healthy 10/20 EMA pullback
        SignalSpec("sector_strength_score", 0.15, "higher"),
    ),
    flags=("earnings_in_window",),
)

MOMENTUM = Profile(
    name="momentum",
    label="Momentum / Growth",
    filters=(
        Filter("price_above_sma_50", "is_true"),
    ),
    signals=(
        SignalSpec("momentum_3m", 0.18, "higher"),
        SignalSpec("momentum_6m", 0.18, "higher"),
        SignalSpec("momentum_12m", 0.14, "higher"),
        SignalSpec("momentum_1m", 0.06, "higher"),
        SignalSpec("sma_stacked_20_50_150", 0.12, "higher"),
        SignalSpec("rel_volume_20", 0.10, "higher"),
        SignalSpec("dist_52w_high", 0.10, "higher"),
        SignalSpec("earnings_growth", 0.12, "higher"),
    ),
)

# A "browse everything" lens: NO hard filters, so every scanned name appears
# (the user asked for an unfiltered view alongside the styled profiles). With no
# style to "fit", a single market-cap signal gives the full list a conventional
# ordering — the biggest, most-liquid names first — and keeps the headline
# Fit/Score meaningful (a size percentile) instead of a flat zero for every row.
ALL_TICKERS = Profile(
    name="all",
    label="All Tickers",
    filters=(),
    signals=(
        SignalSpec("market_cap", 1.0, "higher"),
    ),
)


# Registry keyed by ``Profile.name``; the single source of truth for the engine,
# the CLI smoke block, and the M5 dashboard's profile toggle. ALL_TICKERS sits
# FIRST so the unfiltered "browse everything" lens is the leftmost option in the
# switcher AND the default (``next(iter(PROFILES))``); the styled profiles follow.
PROFILES: "dict[str, Profile]" = {
    p.name: p for p in (ALL_TICKERS, LONG_TERM, SWING, MOMENTUM)
}


def get_profile(name: str) -> Profile:
    """Look up a profile by ``name`` (case-insensitive), raising on an unknown key.

    The error names the valid keys so a typo at the CLI / UI boundary is obvious.
    """
    key = name.strip().lower()
    try:
        return PROFILES[key]
    except KeyError:
        raise KeyError(
            f"Unknown profile {name!r}. Available: {sorted(PROFILES)}"
        ) from None
