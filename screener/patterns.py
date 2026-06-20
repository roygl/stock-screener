"""Descriptive chart-pattern detection — a pure, offline read over price geometry.

The data layer (:mod:`screener.provider`) hands every ticker a canonical daily
price frame: columns ``open/high/low/close/volume`` (lower-cased), a tz-naive
``DatetimeIndex`` named ``date``, OLDEST row first, on an adjusted-close basis.
This module turns that frame into a list of named *chart shapes* — double tops,
head-and-shoulders, triangles, wedges, etc. — each carried as a frozen
:class:`Pattern`.

It is DESCRIPTIVE only. A :class:`Pattern` reports the geometry it found (the
levels, the span, a confidence) and a *shape-orientation* ``direction`` label —
NEVER a buy/sell/hold call, a price target, or a trade. The ``direction``
convention is purely about which way the textbook shape points:

  * ``"bullish"``  — double_bottom, inverse_head_and_shoulders, cup_and_handle,
                     ascending_triangle, falling_wedge
  * ``"bearish"``  — double_top, head_and_shoulders, descending_triangle,
                     rising_wedge
  * ``"neutral"``  — symmetric_triangle

Pipeline (all pure functions; no network, no streamlit, no yfinance):
  1. :func:`resample_ohlc` — derive a 1d / 1w / 1mo OHLCV frame from daily bars.
  2. :func:`find_pivots`   — the FOUNDATION: an alternating swing-high/low zigzag
     with a per-timeframe fractal window and a min-move de-noise threshold.
  3. ``_detect_*``         — per-pattern detectors over the pivot sequence; each
     has hard gates (precision over recall) and an honest confidence in ``[0, 1]``.
  4. :func:`detect` / :func:`detect_all_timeframes` — public entry points.

Design notes:
- **Precision over recall (volatility-relative gates).** The reversal detectors
  (double top/bottom, H&S/inverse) and the cup detector apply a *volatility-relative
  amplitude gate*: the pattern's characteristic vertical move (trough/peak depth,
  head prominence, cup depth) must clear a multiple of the series' realized
  volatility (std of bar-to-bar % returns) over the SAME span, so small random
  wiggles don't qualify while genuine textbook shapes do. **Honest limitation:** a
  textbook head-and-shoulders has large legs by construction, so its
  prominence/volatility ratio overlaps the random-walk distribution; the gate
  silences the weakest spurious H&S but cannot eliminate all of them. Reversal
  shapes (especially H&S) can therefore still occasionally appear on noisy data —
  this readout is DESCRIPTIVE, not a claim that every reported shape is significant.
- **Mutual-exclusion de-dup.** When contradictory shapes overlap (share >= 2
  pivots) only one survives (:func:`_dedup_mutual_exclusion`): a multi-pivot
  triangle/wedge/cup *container* outranks a reversal embedded in the same pivots
  (so a canonical ascending triangle reports the TRIANGLE, not the bearish double
  top its flat-top touches incidentally satisfy), an H&S outranks a plain double,
  and otherwise the higher-confidence shape wins.
- **Picklability.** :class:`Pattern` and :class:`Pivot` are ``frozen`` dataclasses
  of only ``(str, float, pd.Timestamp, tuple-of-frozen)`` fields, so the whole
  ``dict[str, list[Pattern]]`` round-trips through ``st.cache_data`` exactly like
  the existing :class:`~screener.indicators.EmaCrossState`.
- **No lookahead.** A pivot at bar ``i`` is confirmed from the symmetric window
  around ``i``, which only exists once bars ``i + k`` are present — fine for a
  descriptive readout over completed history. The single concession is a
  *provisional* most-recent pivot computed from trailing bars ONLY (never future
  bars), so a still-forming last leg can be seen; that latest pivot may flip as
  new data arrives and is documented as provisional.
- **Adjusted close.** ``yfinance`` is split/dividend-adjusted, so very long
  pre-adjustment shapes can bend slightly — but consistently across O/H/L/C, so
  the geometry (and the same adjusted series the rest of the app uses) holds.
- **Fail-soft.** Short / empty / degenerate frames yield ``[]`` (or ``{tf: []}``);
  :func:`detect` wraps its body in ``try/except`` and returns ``[]`` on any error,
  mirroring the provider/engine ethos. Only a bad ``timeframe`` string raises
  (programmer error), and only out of :func:`resample_ohlc`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# --- human labels --------------------------------------------------------
# name -> display label. The keys are the canonical Pattern.name values; every
# detector and the public API use these exact strings.
PATTERN_LABELS: dict[str, str] = {
    "head_and_shoulders": "Head & Shoulders",
    "inverse_head_and_shoulders": "Inverse Head & Shoulders",
    "double_top": "Double Top",
    "double_bottom": "Double Bottom",
    "cup_and_handle": "Cup & Handle",
    "ascending_triangle": "Ascending Triangle",
    "descending_triangle": "Descending Triangle",
    "symmetric_triangle": "Symmetric Triangle",
    "rising_wedge": "Rising Wedge",
    "falling_wedge": "Falling Wedge",
}


def human_label(name: str) -> str:
    """Map a pattern name to its display label (title-cases unknown names)."""
    return PATTERN_LABELS.get(name, name.replace("_", " ").title())


# --- canonical empty frame (replicated, NOT imported, to stay provider-free) -
_CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume"]


def _empty_frame() -> pd.DataFrame:
    """The provider's canonical empty OHLCV shape, rebuilt inline.

    Kept here (rather than importing ``provider.empty_price_frame``) so this module
    stays pure / standalone with no provider dependency.
    """
    return pd.DataFrame(columns=_CANONICAL_COLUMNS, index=pd.DatetimeIndex([], name="date"))


# --- dataclasses ---------------------------------------------------------
@dataclass(frozen=True)
class Pivot:
    """One swing point in the zigzag.

    ``idx`` is the 0-based POSITIONAL index into the (resampled) frame; ``date`` is
    the index label at that position; ``price`` is the swing high (for ``"H"``) or
    swing low (for ``"L"``); ``kind`` is ``"H"`` or ``"L"``. Frozen so it is
    hashable and pickles cleanly; used internally and optionally surfaced via
    :attr:`Pattern.key_points`.
    """

    idx: int
    date: pd.Timestamp
    price: float
    kind: str  # "H" swing high | "L" swing low


@dataclass(frozen=True)
class Pattern:
    """A detected chart shape — DESCRIPTIVE geometry, never advice.

    ``direction`` is a shape-orientation label (bullish/bearish/neutral per the
    module docstring), NOT a recommendation. ``confidence`` is a ``[0, 1]`` score
    built from the detector's named sub-scores. ``start`` / ``end`` are the index
    labels of the first / last defining bar in the span. ``detail`` is a short
    human sentence describing the levels (no trade language). ``key_points`` holds
    the defining pivots, oldest-first, as a tuple (so the dataclass stays frozen
    and picklable). ``timeframe`` is set by :func:`detect`.
    """

    name: str
    direction: str
    confidence: float
    start: pd.Timestamp
    end: pd.Timestamp
    detail: str
    key_points: tuple[Pivot, ...] = ()
    timeframe: str = "1d"

    def label(self) -> str:
        """Human display label for this pattern's name."""
        return human_label(self.name)

    def to_dict(self) -> dict:
        """Plain-dict view (Timestamps left as Timestamp; the app formats them)."""
        return asdict(self)


# --- resampling ----------------------------------------------------------
# Right-labeled weekly (Friday label covers Mon–Fri, never peeking past the
# bucket) and month-end monthly. Verified on a 2y business-day span:
# weekly -> ~105 bars, monthly -> ~25 bars.
_FREQ = {"1w": "W-FRI", "1mo": "ME"}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlc(daily_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a canonical daily OHLCV frame to ``1d`` / ``1w`` / ``1mo``.

    Pure; never raises on short/empty input (only a bad ``timeframe`` string
    raises ``ValueError``, which is programmer error — :func:`detect` only ever
    passes valid values). ``"1d"`` returns the frame UNCHANGED (identity); the
    contract guarantees its canonical shape, so no copy/normalize is done.
    Empty / ``None`` / close-less input returns the canonical empty frame.
    Aggregation is OHLCV-correct (open=first, high=max, low=min, close=last,
    volume=sum); empty calendar buckets are dropped.
    """
    if timeframe == "1d":
        return daily_df  # identity by contract — do NOT copy/normalize

    # Guard FIRST (before the timeframe-validity check would matter for data):
    # a missing / empty / malformed frame fails soft to the canonical empty shape.
    if daily_df is None or len(daily_df) == 0 or "close" not in getattr(daily_df, "columns", []):
        return _empty_frame()

    if timeframe not in ("1w", "1mo"):
        raise ValueError(f"timeframe must be 1d/1w/1mo, got {timeframe!r}")

    freq = _FREQ[timeframe]
    if timeframe == "1w":
        # Right-labeled / right-closed so the Friday bucket label is the period END.
        out = daily_df.resample(freq, label="right", closed="right").agg(_AGG)
    else:  # "1mo": ME already anchors to month-end (right/period-end by default).
        out = daily_df.resample(freq).agg(_AGG)

    # A fully-NaN row is a calendar bucket with zero trading days; then mirror the
    # provider's close-anchored cleaning.
    out = out.dropna(how="all")
    out = out.dropna(subset=["close"])
    out.index.name = "date"  # resample preserves it, but be explicit
    return out.reindex(columns=_CANONICAL_COLUMNS)


# --- pivot detection (the FOUNDATION) ------------------------------------
MIN_BARS_FOR_PIVOTS = 15  # below this a zigzag is meaningless -> []

# Fractal half-width base per timeframe (daily noisier -> wider; monthly smooth).
_PIVOT_BASE = {"1d": 3, "1w": 2, "1mo": 1}
_PIVOT_MAX_K = {"1d": 8, "1w": 4, "1mo": 2}
# Min confirmed swing vs the last accepted opposite pivot (bigger bars need more).
MIN_MOVE_PCT = {"1d": 0.03, "1w": 0.05, "1mo": 0.08}


def _window_k(n: int, timeframe: str) -> int:
    """Fractal half-width: base per timeframe, widened slightly for long series."""
    base = _PIVOT_BASE[timeframe]
    adaptive = max(base, int(round(n / 40)))
    k = min(adaptive, _PIVOT_MAX_K[timeframe])
    return max(1, k)


def _raw_extrema(highs, lows, n: int, k: int) -> list[tuple[int, str]]:
    """Symmetric-fractal raw swing highs/lows (with provisional last-bar edge).

    Index ``i`` is a raw HIGH iff ``highs[i]`` is the max of the symmetric window
    ``[i-k, i+k]`` AND strictly exceeds both window ends (ties allowed inside, ends
    strict, so flat plateaus don't spawn many pivots). Symmetric for LOW. The most
    recent bars (where a full forward window doesn't exist yet) may be a
    PROVISIONAL extremum using only the trailing ``2k+1`` available bars — past
    bars only, never future.
    """
    out: list[tuple[int, str]] = []
    for i in range(k, n - k):
        win_hi = highs[i - k : i + k + 1]
        if highs[i] == win_hi.max() and highs[i] > highs[i - k] and highs[i] > highs[i + k]:
            out.append((i, "H"))
            continue
        win_lo = lows[i - k : i + k + 1]
        if lows[i] == win_lo.min() and lows[i] < lows[i - k] and lows[i] < lows[i + k]:
            out.append((i, "L"))

    # Provisional last pivot: let a forming final leg be seen using trailing bars.
    for i in range(max(k, n - k), n):
        lo = max(0, i - 2 * k)
        if i - lo < k:  # need at least a half-window of history behind it
            continue
        if highs[i] == highs[lo : i + 1].max() and highs[i] > highs[lo]:
            if not out or out[-1][0] != i:
                out.append((i, "H"))
        elif lows[i] == lows[lo : i + 1].min() and lows[i] < lows[lo]:
            if not out or out[-1][0] != i:
                out.append((i, "L"))
    out.sort(key=lambda t: t[0])
    # De-dup any index that the main loop and the provisional pass both produced.
    seen: set[int] = set()
    deduped: list[tuple[int, str]] = []
    for i, kind in out:
        if i not in seen:
            seen.add(i)
            deduped.append((i, kind))
    return deduped


def find_pivots(df: pd.DataFrame, *, timeframe: str = "1d") -> list[Pivot]:
    """Alternating swing-high/low zigzag — the foundation every detector reads.

    Pure; returns ``[]`` on short/empty input; oldest-first; STRICTLY alternating
    ``H, L, H, L, ...``; no lookahead beyond the available bars (see module
    docstring on the provisional last pivot). Uses ``high`` / ``low`` arrays for the
    swing series (falling back to ``close`` for both if either is missing); a NaN in
    that high/low swing series fails soft to ``[]``. A swing is kept only if it
    clears :data:`MIN_MOVE_PCT` vs the last accepted opposite pivot (de-noise); two
    same-kind candidates in a row collapse to the more extreme one (dominance),
    enforcing the alternation.
    """
    if df is None or len(df) < MIN_BARS_FOR_PIVOTS:
        return []
    cols = getattr(df, "columns", [])
    if "high" in cols and "low" in cols and "close" in cols:
        highs = df["high"].to_numpy(dtype="float64")
        lows = df["low"].to_numpy(dtype="float64")
    elif "close" in cols:
        highs = lows = df["close"].to_numpy(dtype="float64")
    else:
        return []
    n = len(highs)
    # Degenerate / NaN-laden swing series (the high/low arrays the zigzag reads):
    # don't try to read a shape off it. The < MIN_BARS check is already enforced
    # above via len(df), so this guard is purely about NaNs here.
    if np.isnan(highs).any() or np.isnan(lows).any():
        return []

    k = _window_k(n, timeframe)
    move = MIN_MOVE_PCT[timeframe]
    index = df.index

    raw = _raw_extrema(highs, lows, n, k)

    def _price(i: int, kind: str) -> float:
        return float(highs[i] if kind == "H" else lows[i])

    # Single forward pass building the zigzag with alternation + dominance.
    pivots: list[Pivot] = []
    for i, kind in raw:
        price = _price(i, kind)
        cand = Pivot(idx=i, date=index[i], price=price, kind=kind)
        if not pivots:
            pivots.append(cand)
        elif cand.kind != pivots[-1].kind:
            base = pivots[-1].price
            if base != 0 and abs(price - base) / abs(base) >= move:
                pivots.append(cand)
            # else: sub-threshold wiggle — skip (de-noise).
        else:  # same kind in a row: keep the more extreme (dominant) swing
            if kind == "H" and price > pivots[-1].price:
                pivots[-1] = cand
            elif kind == "L" and price < pivots[-1].price:
                pivots[-1] = cand
            # else discard the weaker same-kind candidate.
    return pivots


# --- pivot / geometry helpers (reused by detectors) ----------------------
def _highs(pivots: Sequence[Pivot]) -> list[Pivot]:
    return [p for p in pivots if p.kind == "H"]


def _lows(pivots: Sequence[Pivot]) -> list[Pivot]:
    return [p for p in pivots if p.kind == "L"]


def _last_n_pivots(pivots: Sequence[Pivot], m: int) -> list[Pivot]:
    return list(pivots[-m:])


def _slope(p1: Pivot, p2: Pivot) -> float:
    """Price slope per bar between two pivots (0.0 if same index)."""
    di = p2.idx - p1.idx
    return 0.0 if di == 0 else (p2.price - p1.price) / di


def _line_at(p1: Pivot, p2: Pivot, idx: int) -> float:
    """Linear interp/extrapolation of the (p1, p2) line at a positional index."""
    return p1.price + _slope(p1, p2) * (idx - p1.idx)


def _pct_diff(a: float, b: float) -> float:
    """Symmetric relative difference — |a-b| over the mean magnitude."""
    denom = (abs(a) + abs(b)) / 2 or 1.0
    return abs(a - b) / denom


def _sim(a: float, b: float, tol: float) -> float:
    """Closeness score in ``[0, 1]``: 1 when equal, 0 once the rel-diff hits ``tol``."""
    if tol <= 0:
        return 1.0 if a == b else 0.0
    return max(0.0, min(1.0, 1.0 - _pct_diff(a, b) / tol))


def _clamp01(c: float) -> float:
    return max(0.0, min(1.0, c))


def _fit_line(points: Sequence[Pivot]) -> tuple[float, float, float]:
    """Least-squares line over (idx, price); returns (slope, intercept, R^2).

    Guards ``< 2`` points -> ``(0.0, price, 0.0)``. ``R^2`` from residual vs total
    sum of squares (0.0 when prices are constant).
    """
    if len(points) < 2:
        price = points[0].price if points else 0.0
        return 0.0, float(price), 0.0
    idxs = np.array([p.idx for p in points], dtype="float64")
    prices = np.array([p.price for p in points], dtype="float64")
    slope, intercept = np.polyfit(idxs, prices, 1)
    pred = slope * idxs + intercept
    ss_res = float(np.sum((prices - pred) ** 2))
    ss_tot = float(np.sum((prices - prices.mean()) ** 2))
    if ss_tot <= 1e-12:
        # All prices identical: a perfectly flat line fits perfectly. R^2 is
        # formally undefined here, but a dead-flat resistance/support is the
        # STRONGEST flat-trendline case (an ascending/descending triangle's ideal
        # ceiling/floor), so score it as a perfect fit rather than penalizing it.
        r2 = 1.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), _clamp01(r2)


def _overlaps(p_a: Pattern, p_b: Pattern) -> bool:
    """True if two patterns share >= 2 pivot positions (for mutual-exclusion)."""
    idx_a = {pt.idx for pt in p_a.key_points}
    idx_b = {pt.idx for pt in p_b.key_points}
    return len(idx_a & idx_b) >= 2


def _close_at(closes: np.ndarray, idx: int) -> float:
    """Close at positional ``idx``, clamped into the array (nearest in-range bar).

    The ``confirm`` sub-score asks whether price closed beyond a pattern's level by
    the END of the *pattern span* — so it must read the close at the span-end pivot
    index, NOT the last bar of the whole frame (which can sit well outside the span
    for an older shape). Clamps defensively so an out-of-range index never throws.
    """
    if len(closes) == 0:
        return float("nan")
    j = min(max(int(idx), 0), len(closes) - 1)
    return float(closes[j])


def _realized_vol(closes: np.ndarray, lo: int, hi: int) -> float:
    """Std of bar-to-bar % returns over ``closes[lo:hi+1]`` (the pattern span).

    The volatility-relative amplitude gate's denominator: an ATR-like, unit-free
    measure of how much this series wiggles bar-to-bar *inside the pattern span*.
    Returns ``0.0`` if the span is too short or degenerate (callers treat a
    non-positive vol as "no opinion" and skip the gate rather than divide by zero).
    """
    lo = max(int(lo), 0)
    hi = min(int(hi), len(closes) - 1)
    seg = closes[lo : hi + 1]
    if len(seg) < 3:
        return 0.0
    base = seg[:-1]
    # Guard a zero base (constant/degenerate prices) -> no usable return there.
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(seg) / base
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets))


# --- volatility-relative amplitude gates (precision over recall on noise) -----
# A textbook reversal/cup's characteristic vertical move dwarfs the series' own
# bar-to-bar noise; a random walk's incidental "shapes" do not. We gate on the
# move measured against the realized volatility over the SAME span. Two flavors:
#
#  * span-normalized (double top/bottom, cup): a random walk's cumulative
#    excursion grows ~ vol * sqrt(span), so we require the move to clear a multiple
#    of that. This cleanly separates the canonical shapes (concentrated moves) from
#    the slow drift a long random walk accumulates.
#  * raw ATR-like (head & shoulders): the head *prominence* is a LOCAL shoulder->
#    head feature, so we compare it to the raw realized vol (no sqrt(span) term).
#
# Calibrated on seeded random walks (sigma~1.2%/bar) vs the canonical fixtures:
# canonical DT/DB span-metric ~2.3-2.4 (gate 2.0), cup ~2.3 (gate 1.8), H&S raw
# ratio ~3.5-4.3 (gate 3.0). NOTE (honest limitation, see module docstring): the
# H&S gate cannot fully separate noise — a textbook H&S has large legs, so its
# prominence/vol ratio overlaps the random-walk distribution. The gate silences
# the weakest spurious H&S; some can still appear. The feature stays descriptive.
DT_DB_VOL_GATE = 2.0    # double top/bottom: move / (vol * sqrt(span))
CUP_VOL_GATE = 1.8      # cup depth / (vol * sqrt(span))
HS_VOL_GATE = 3.0       # H&S head prominence (fraction) / realized vol


def _passes_span_vol_gate(move_frac: float, closes: np.ndarray, lo: int, hi: int,
                          gate: float) -> bool:
    """True if ``move_frac`` clears ``gate * vol * sqrt(span)`` (or vol is unusable).

    ``move_frac`` is the pattern's characteristic vertical move as a fraction
    (trough depth, cup depth). A non-positive realized vol means we can't form an
    opinion, so we DON'T block (return True) — the shape's own structural gates
    already had to pass to get here.
    """
    vol = _realized_vol(closes, lo, hi)
    span = max(int(hi) - int(lo), 1)
    if vol <= 0.0:
        return True
    return move_frac >= gate * vol * math.sqrt(span)


def _passes_raw_vol_gate(move_frac: float, closes: np.ndarray, lo: int, hi: int,
                         gate: float) -> bool:
    """True if ``move_frac`` clears ``gate * vol`` (raw ATR-like, no sqrt(span))."""
    vol = _realized_vol(closes, lo, hi)
    if vol <= 0.0:
        return True
    return move_frac >= gate * vol


# --- detector tolerance constants (the precision/recall dial) ------------
# Reversal-pattern tolerances.
TOL_PEAK = 0.03      # two tops/bottoms "similar" within 3%
MIN_TROUGH = 0.04    # a real valley/peak between them is >= 4%
HEAD_PROM = 0.03     # H&S head clears each shoulder by >= 3%
TOL_SHOULDER = 0.05  # shoulders symmetric within 5%
TOL_NECK = 0.06      # neckline roughly level within 6%
TOL_RIM = 0.05       # cup rims even within 5%
CUP_MIN_BARS = {"1d": 15, "1w": 7, "1mo": 4}

# Trendline (triangle / wedge) tolerances. Normalized slopes are % of mean price
# PER BAR, so they're unit-free; the per-timeframe scale widens them for the
# coarser, fewer-bar weekly/monthly frames.
_TF_SCALE = {"1d": 1.0, "1w": 2.5, "1mo": 6.0}
FLAT_TOL = 0.0015    # |slope| this small (per bar, normalized) reads as "flat"
RISE_TOL = 0.0015    # a line rising/falling faster than this is a real trend
CONV_TOL = 0.0015    # symmetric-triangle min |slope| each side
WEDGE_TOL = 0.0015   # wedge min |slope| each side (both lines clearly sloping)

# Per-bar slope tolerances above are sized for a ~compact triangle. A long, gently
# converging triangle has the SAME total convergence spread over many more bars, so
# its per-bar slope is proportionally smaller and would miss a fixed per-bar floor.
# We scale the trend-FLOOR tolerances (RISE/CONV/WEDGE) down with span length:
# floor *= min(1, SLOPE_SPAN_REF / span). This admits slow converges WITHOUT
# loosening the channel guard — the convergence/range-shrink gate (range ratio
# <= ~0.6) still rejects a parallel channel, whose range does not shrink at all.
# The "flat" CEILING (FLAT_TOL) is NOT scaled: flatness is scale-invariant, and a
# flat line stays flat at any length.
SLOPE_SPAN_REF = 45  # spans up to this keep the base per-bar tolerances (factor 1)

MIN_CONFIDENCE = 0.45  # precision floor — weaker shapes are never reported


def _tf_of(df: pd.DataFrame) -> str:
    """Best-effort timeframe inference from the median bar spacing.

    Detectors take only ``(df, pivots)`` per the spec signature, but the trendline
    tolerances are scaled per timeframe. We infer it from the index cadence
    (daily ~1d, weekly ~7d, monthly ~30d) so the scale is right without changing
    the signature. Defaults to ``"1d"`` on anything ambiguous.
    """
    idx = getattr(df, "index", None)
    if idx is None or len(idx) < 3:
        return "1d"
    try:
        deltas = np.diff(idx.view("int64")) / 1e9 / 86400.0  # days between bars
    except Exception:
        return "1d"
    med = float(np.median(deltas)) if len(deltas) else 1.0
    if med >= 20:
        return "1mo"
    if med >= 4:
        return "1w"
    return "1d"


# --- reversal detectors --------------------------------------------------
def _detect_double_top(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """Two similar highs with a real trough between (bearish shape)."""
    closes = df["close"].to_numpy(dtype="float64")
    best: Optional[Pattern] = None
    # Scan recent consecutive triples H, L, H; prefer the latest qualifying one.
    for i in range(len(pivots) - 3, -1, -1):
        a, b, c = pivots[i], pivots[i + 1], pivots[i + 2]
        if not (a.kind == "H" and b.kind == "L" and c.kind == "H"):
            continue
        h1, l1, h2 = a.price, b.price, c.price
        if _pct_diff(h1, h2) > TOL_PEAK:
            continue
        trough_drop = (min(h1, h2) - l1) / min(h1, h2)
        if trough_drop < MIN_TROUGH or (c.idx - a.idx) < 2:
            continue
        # Volatility-relative amplitude gate: the trough must dwarf the span's noise.
        if not _passes_span_vol_gate(trough_drop, closes, a.idx, c.idx, DT_DB_VOL_GATE):
            continue
        peak_sim = _sim(h1, h2, TOL_PEAK)
        trough_score = min(1.0, trough_drop / 0.10)
        # confirm at the span END (last defining pivot), not the last bar of the frame.
        confirm = 1.0 if _close_at(closes, c.idx) < l1 else 0.5
        conf = _clamp01((peak_sim + trough_score + confirm) / 3.0)
        detail = (
            f"Two highs near {h1:.2f} and {h2:.2f} with a "
            f"{trough_drop * 100:.0f}% trough between."
        )
        cand = Pattern("double_top", "bearish", conf, a.date, c.date, detail, (a, b, c))
        if best is None or cand.confidence > best.confidence:
            best = cand
        break  # latest qualifying window only
    return best


def _detect_double_bottom(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """Two similar lows with a real peak between (bullish shape)."""
    closes = df["close"].to_numpy(dtype="float64")
    best: Optional[Pattern] = None
    for i in range(len(pivots) - 3, -1, -1):
        a, b, c = pivots[i], pivots[i + 1], pivots[i + 2]
        if not (a.kind == "L" and b.kind == "H" and c.kind == "L"):
            continue
        l1, h1, l2 = a.price, b.price, c.price
        if _pct_diff(l1, l2) > TOL_PEAK:
            continue
        peak_rise = (h1 - max(l1, l2)) / max(l1, l2)
        if peak_rise < MIN_TROUGH or (c.idx - a.idx) < 2:
            continue
        # Volatility-relative amplitude gate: the peak must dwarf the span's noise.
        if not _passes_span_vol_gate(peak_rise, closes, a.idx, c.idx, DT_DB_VOL_GATE):
            continue
        low_sim = _sim(l1, l2, TOL_PEAK)
        peak_score = min(1.0, peak_rise / 0.10)
        # confirm at the span END (last defining pivot), not the last bar of the frame.
        confirm = 1.0 if _close_at(closes, c.idx) > h1 else 0.5
        conf = _clamp01((low_sim + peak_score + confirm) / 3.0)
        detail = (
            f"Two lows near {l1:.2f} and {l2:.2f} with a "
            f"{peak_rise * 100:.0f}% peak between."
        )
        cand = Pattern("double_bottom", "bullish", conf, a.date, c.date, detail, (a, b, c))
        if best is None or cand.confidence > best.confidence:
            best = cand
        break
    return best


def _detect_head_and_shoulders(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """Three highs, the middle clearly tallest, over a roughly level neckline."""
    closes = df["close"].to_numpy(dtype="float64")
    best: Optional[Pattern] = None
    for i in range(len(pivots) - 5, -1, -1):
        s = pivots[i : i + 5]
        kinds = [p.kind for p in s]
        if kinds != ["H", "L", "H", "L", "H"]:
            continue
        h_l, t1, h_head, t2, h_r = s
        # GATES: head clearly tallest; shoulders symmetric; neckline level; both
        # troughs below both shoulders.
        if not (h_head.price - h_l.price >= HEAD_PROM * h_head.price
                and h_head.price - h_r.price >= HEAD_PROM * h_head.price):
            continue
        if _pct_diff(h_l.price, h_r.price) > TOL_SHOULDER:
            continue
        if _pct_diff(t1.price, t2.price) > TOL_NECK:
            continue
        if not (t1.price < h_l.price and t1.price < h_r.price
                and t2.price < h_l.price and t2.price < h_r.price):
            continue
        # Volatility-relative gate: head prominence (vs the smaller shoulder gap)
        # must clear a multiple of the span's realized vol. See HS_VOL_GATE note.
        head_prom_frac = min(h_head.price - h_l.price, h_head.price - h_r.price) / h_head.price
        if not _passes_raw_vol_gate(head_prom_frac, closes, h_l.idx, h_r.idx, HS_VOL_GATE):
            continue
        shoulder_sim = _sim(h_l.price, h_r.price, TOL_SHOULDER)
        head_prom_score = min(1.0, head_prom_frac / 0.08)
        neck_level = _sim(t1.price, t2.price, TOL_NECK)
        # confirm at the span END (right shoulder), not the last bar of the frame.
        confirm = 1.0 if _close_at(closes, h_r.idx) < _line_at(t1, t2, h_r.idx) else 0.5
        conf = _clamp01((shoulder_sim + head_prom_score + neck_level + confirm) / 4.0)
        detail = (
            f"Head near {h_head.price:.2f} above shoulders at "
            f"{h_l.price:.2f} and {h_r.price:.2f}."
        )
        cand = Pattern(
            "head_and_shoulders", "bearish", conf, h_l.date, h_r.date, detail, tuple(s)
        )
        if best is None or cand.confidence > best.confidence:
            best = cand
        break
    return best


def _detect_inverse_head_and_shoulders(
    df: pd.DataFrame, pivots: list[Pivot]
) -> Optional[Pattern]:
    """Three lows, the middle clearly lowest, under a roughly level neckline."""
    closes = df["close"].to_numpy(dtype="float64")
    best: Optional[Pattern] = None
    for i in range(len(pivots) - 5, -1, -1):
        s = pivots[i : i + 5]
        kinds = [p.kind for p in s]
        if kinds != ["L", "H", "L", "H", "L"]:
            continue
        l_l, t1, l_head, t2, l_r = s
        if not (l_l.price - l_head.price >= HEAD_PROM * l_head.price
                and l_r.price - l_head.price >= HEAD_PROM * l_head.price):
            continue
        if _pct_diff(l_l.price, l_r.price) > TOL_SHOULDER:
            continue
        if _pct_diff(t1.price, t2.price) > TOL_NECK:
            continue
        if not (t1.price > l_l.price and t1.price > l_r.price
                and t2.price > l_l.price and t2.price > l_r.price):
            continue
        # Volatility-relative gate (mirrors the upright H&S): see HS_VOL_GATE note.
        head_prom_frac = min(l_l.price - l_head.price, l_r.price - l_head.price) / l_head.price
        if not _passes_raw_vol_gate(head_prom_frac, closes, l_l.idx, l_r.idx, HS_VOL_GATE):
            continue
        shoulder_sim = _sim(l_l.price, l_r.price, TOL_SHOULDER)
        head_prom_score = min(1.0, head_prom_frac / 0.08)
        neck_level = _sim(t1.price, t2.price, TOL_NECK)
        # confirm at the span END (right shoulder), not the last bar of the frame.
        confirm = 1.0 if _close_at(closes, l_r.idx) > _line_at(t1, t2, l_r.idx) else 0.5
        conf = _clamp01((shoulder_sim + head_prom_score + neck_level + confirm) / 4.0)
        detail = (
            f"Low near {l_head.price:.2f} below shoulders at "
            f"{l_l.price:.2f} and {l_r.price:.2f}."
        )
        cand = Pattern(
            "inverse_head_and_shoulders", "bullish", conf, l_l.date, l_r.date, detail, tuple(s)
        )
        if best is None or cand.confidence > best.confidence:
            best = cand
        break
    return best


def _detect_cup_and_handle(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """A rounded U-shaped cup (even rims, low near the middle) then a shallow handle."""
    tf = _tf_of(df)
    cup_min = CUP_MIN_BARS[tf]
    closes = df["close"].to_numpy(dtype="float64")
    highs = _highs(pivots)
    if len(highs) < 2:
        return None
    best: Optional[Pattern] = None
    # Consider pairs of highs as the rims, latest pair first.
    for j in range(len(highs) - 1, 0, -1):
        hr = highs[j]
        for k_i in range(j - 1, -1, -1):
            hl = highs[k_i]
            width = hr.idx - hl.idx
            if width < cup_min:
                continue
            if _pct_diff(hl.price, hr.price) > TOL_RIM:
                continue
            # Cup low = lowest LOW pivot strictly between the rims.
            inner_lows = [p for p in _lows(pivots) if hl.idx < p.idx < hr.idx]
            if not inner_lows:
                continue
            cl = min(inner_lows, key=lambda p: p.price)
            cup_depth = (min(hl.price, hr.price) - cl.price) / min(hl.price, hr.price)
            if not (0.12 <= cup_depth <= 0.50):
                continue
            # Volatility-relative gate: a real cup's depth dwarfs what a random walk
            # would drift over the same (often long) rim-to-rim span; a slow noisy
            # drawdown-and-recovery does not. Measured over the cup body (rims).
            if not _passes_span_vol_gate(cup_depth, closes, hl.idx, hr.idx, CUP_VOL_GATE):
                continue
            center = (cl.idx - hl.idx) / width
            if not (0.30 <= center <= 0.70):  # U-shaped, not a V / check-mark
                continue
            # Handle: a low AFTER the right rim, shallow, staying in the upper third.
            handle_lows = [p for p in _lows(pivots) if p.idx > hr.idx]
            handle_score = 0.4
            hd = None
            for cand_hd in handle_lows:
                handle_depth = (hr.price - cand_hd.price) / hr.price
                in_upper = cand_hd.price > cl.price + 0.5 * (hr.price - cl.price)
                if 0 < handle_depth <= 0.5 * cup_depth and in_upper:
                    hd = cand_hd
                    handle_score = 1.0
                    break
            if handle_score < 0.4:  # favor precision: a cup w/o handle isn't this
                continue
            rim_sim = _sim(hl.price, hr.price, TOL_RIM)
            # Triangular depth kernel: 1.0 at ~0.25, decaying toward [0.12, 0.50].
            if cup_depth <= 0.25:
                depth_score = (cup_depth - 0.12) / (0.25 - 0.12)
            else:
                depth_score = (0.50 - cup_depth) / (0.50 - 0.25)
            depth_score = _clamp01(depth_score)
            round_score = _clamp01(1.0 - abs(center - 0.5) / 0.2)
            conf = _clamp01((rim_sim + depth_score + round_score + handle_score) / 4.0)
            if hd is not None:
                detail = (
                    f"Rounded cup ~{cup_depth * 100:.0f}% deep with a shallow handle "
                    f"near {hd.price:.2f}."
                )
                kp = (hl, cl, hr, hd)
                end_date = hd.date
            else:
                detail = f"Rounded cup ~{cup_depth * 100:.0f}% deep between even rims."
                kp = (hl, cl, hr)
                end_date = hr.date
            cand = Pattern("cup_and_handle", "bullish", conf, hl.date, end_date, detail, kp)
            if best is None or cand.confidence > best.confidence:
                best = cand
            return best  # latest rim-pair that qualifies
    return best


# --- trendline detectors (triangles + wedges) ----------------------------
def _trend_inputs(df: pd.DataFrame, pivots: list[Pivot]):
    """Shared setup for trendline detectors: recent highs/lows, fits, ranges.

    Returns ``None`` unless there are >= 3 swing highs AND >= 3 swing lows. Slopes
    are normalized by mean price (% per bar) and the per-timeframe scale; the
    start/end range is the high-line minus low-line gap measured across the span.
    """
    highs = _last_n_pivots(_highs(pivots), 5)
    lows = _last_n_pivots(_lows(pivots), 5)
    if len(highs) < 3 or len(lows) < 3:
        return None
    sh, ih, r2h = _fit_line(highs)
    sl, il, r2l = _fit_line(lows)
    all_prices = [p.price for p in highs] + [p.price for p in lows]
    mean_price = float(np.mean(all_prices)) or 1.0
    scale = _TF_SCALE[_tf_of(df)]
    sh_n = (sh / mean_price) / scale
    sl_n = (sl / mean_price) / scale
    start_idx = min(highs[0].idx, lows[0].idx)
    end_idx = max(highs[-1].idx, lows[-1].idx)
    # Span-scaled trend-floor factor: a long converge can be gentle per bar.
    span_bars = max(end_idx - start_idx, 1)
    slope_scale = min(1.0, SLOPE_SPAN_REF / span_bars)

    def line(slope, intercept, idx):
        return slope * idx + intercept

    start_range = abs(line(sh, ih, start_idx) - line(sl, il, start_idx))
    end_range = abs(line(sh, ih, end_idx) - line(sl, il, end_idx))
    span_start = min(highs[0].date, lows[0].date)
    span_end = max(highs[-1].date, lows[-1].date)
    key_points = tuple(sorted(highs + lows, key=lambda p: p.idx))
    return {
        "highs": highs, "lows": lows,
        "sh_n": sh_n, "sl_n": sl_n, "r2h": r2h, "r2l": r2l,
        "start_range": start_range, "end_range": end_range,
        "span_start": span_start, "span_end": span_end, "key_points": key_points,
        "sh": sh, "ih": ih, "sl": sl, "il": il,
        "end_idx": end_idx, "slope_scale": slope_scale,
    }


def _range_ratio(d) -> Optional[float]:
    """end/start range ratio (the convergence measure); None if start range ~0."""
    if d["start_range"] <= 1e-9:
        return None
    return d["end_range"] / d["start_range"]


def _detect_ascending_triangle(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """Flat resistance (highs) over a rising support (lows), converging (bullish)."""
    d = _trend_inputs(df, pivots)
    if d is None:
        return None
    flat_tol = FLAT_TOL  # flat CEILING: scale-invariant, not span-scaled
    rise_tol = RISE_TOL * d["slope_scale"]  # trend FLOOR: gentler over a long span
    if abs(d["sh_n"]) > flat_tol:        # highs must be ~flat
        return None
    if d["sl_n"] <= rise_tol:            # lows must be rising
        return None
    ratio = _range_ratio(d)
    if ratio is None or not (d["start_range"] > d["end_range"] and ratio <= 0.6):
        return None
    if d["r2h"] < 0.6:                   # flat resistance must be well-defined
        return None
    flatness = _sim(d["sh_n"], 0.0, max(flat_tol, 1e-9))
    rise_strength = min(1.0, d["sl_n"] / (3 * rise_tol))
    convergence = min(1.0, (1 - ratio) / 0.6)
    conf = _clamp01((flatness + rise_strength + convergence + d["r2h"]) / 4.0)
    detail = "Flat highs with rising lows converging toward the resistance."
    return Pattern(
        "ascending_triangle", "bullish", conf,
        d["span_start"], d["span_end"], detail, d["key_points"],
    )


def _detect_descending_triangle(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """Flat support (lows) under falling resistance (highs), converging (bearish)."""
    d = _trend_inputs(df, pivots)
    if d is None:
        return None
    flat_tol = FLAT_TOL  # flat CEILING: scale-invariant, not span-scaled
    rise_tol = RISE_TOL * d["slope_scale"]  # trend FLOOR: gentler over a long span
    if abs(d["sl_n"]) > flat_tol:        # lows must be ~flat
        return None
    if d["sh_n"] >= -rise_tol:           # highs must be falling
        return None
    ratio = _range_ratio(d)
    if ratio is None or not (d["start_range"] > d["end_range"] and ratio <= 0.6):
        return None
    if d["r2l"] < 0.6:                   # flat support must be well-defined
        return None
    flatness = _sim(d["sl_n"], 0.0, max(flat_tol, 1e-9))
    fall_strength = min(1.0, abs(d["sh_n"]) / (3 * rise_tol))
    convergence = min(1.0, (1 - ratio) / 0.6)
    conf = _clamp01((flatness + fall_strength + convergence + d["r2l"]) / 4.0)
    detail = "Flat lows with falling highs converging toward the support."
    return Pattern(
        "descending_triangle", "bearish", conf,
        d["span_start"], d["span_end"], detail, d["key_points"],
    )


def _detect_symmetric_triangle(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """Falling highs and rising lows converging to a future apex (neutral)."""
    d = _trend_inputs(df, pivots)
    if d is None:
        return None
    conv_tol = CONV_TOL * d["slope_scale"]  # trend FLOOR: gentler over a long span
    if not (d["sh_n"] < -conv_tol and d["sl_n"] > conv_tol):  # opposite-signed
        return None
    ratio = _range_ratio(d)
    if ratio is None or ratio > 0.6:
        return None
    if d["r2h"] < 0.5 or d["r2l"] < 0.5:
        return None
    # Apex (line intersection) should be near/after the end — still converging.
    denom = d["sh"] - d["sl"]
    if abs(denom) > 1e-12:
        apex_idx = (d["il"] - d["ih"]) / denom
        if apex_idx <= d["end_idx"]:  # already crossed -> not a forming triangle
            return None
    converge = min(1.0, (1 - ratio) / 0.6)
    symmetry = _sim(abs(d["sh_n"]), abs(d["sl_n"]), 1.0)
    conf = _clamp01((converge + d["r2h"] + d["r2l"] + symmetry) / 4.0)
    detail = "Falling highs and rising lows converging toward an apex."
    return Pattern(
        "symmetric_triangle", "neutral", conf,
        d["span_start"], d["span_end"], detail, d["key_points"],
    )


def _detect_rising_wedge(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """Both lines up, lows steeper than highs -> converging upward (bearish)."""
    d = _trend_inputs(df, pivots)
    if d is None:
        return None
    wedge_tol = WEDGE_TOL * d["slope_scale"]  # trend FLOOR: gentler over a long span
    if not (d["sh_n"] > wedge_tol and d["sl_n"] > wedge_tol):  # both up
        return None
    if d["sl_n"] <= d["sh_n"]:           # floor must rise faster (converging)
        return None
    ratio = _range_ratio(d)
    if ratio is None or ratio > 0.7:
        return None
    if d["r2h"] < 0.5 or d["r2l"] < 0.5:
        return None
    both_up = 1.0
    convergence = min(1.0, (1 - ratio) / 0.7)
    same_dir = _sim(1.0, 1.0, 1.0)  # both slopes share sign -> 1.0
    conf = _clamp01((both_up + convergence + d["r2h"] + d["r2l"] + same_dir) / 5.0)
    detail = "Both trendlines rising with the lows steeper, narrowing upward."
    return Pattern(
        "rising_wedge", "bearish", conf,
        d["span_start"], d["span_end"], detail, d["key_points"],
    )


def _detect_falling_wedge(df: pd.DataFrame, pivots: list[Pivot]) -> Optional[Pattern]:
    """Both lines down, highs steeper than lows -> converging downward (bullish)."""
    d = _trend_inputs(df, pivots)
    if d is None:
        return None
    wedge_tol = WEDGE_TOL * d["slope_scale"]  # trend FLOOR: gentler over a long span
    if not (d["sh_n"] < -wedge_tol and d["sl_n"] < -wedge_tol):  # both down
        return None
    if abs(d["sh_n"]) <= abs(d["sl_n"]):  # ceiling must fall faster (converging)
        return None
    ratio = _range_ratio(d)
    if ratio is None or ratio > 0.7:
        return None
    if d["r2h"] < 0.5 or d["r2l"] < 0.5:
        return None
    both_down = 1.0
    convergence = min(1.0, (1 - ratio) / 0.7)
    same_dir = _sim(1.0, 1.0, 1.0)
    conf = _clamp01((both_down + convergence + d["r2h"] + d["r2l"] + same_dir) / 5.0)
    detail = "Both trendlines falling with the highs steeper, narrowing downward."
    return Pattern(
        "falling_wedge", "bullish", conf,
        d["span_start"], d["span_end"], detail, d["key_points"],
    )


# --- public API ----------------------------------------------------------
# More-specific reversal patterns first, then triangles/wedges (precision).
_DISPATCH = (
    _detect_head_and_shoulders,
    _detect_inverse_head_and_shoulders,
    _detect_double_top,
    _detect_double_bottom,
    _detect_cup_and_handle,
    _detect_ascending_triangle,
    _detect_descending_triangle,
    _detect_symmetric_triangle,
    _detect_rising_wedge,
    _detect_falling_wedge,
)

# Bars needed before a pattern read is meaningful (per timeframe; safe default 10).
MIN_BARS_FOR_PATTERNS = {"1d": 40, "1w": 20, "1mo": 10}

# Pattern families for the mutual-exclusion de-dup.
_WEDGE_NAMES = {"rising_wedge", "falling_wedge"}
_HS_NAMES = {"head_and_shoulders", "inverse_head_and_shoulders"}
_DOUBLE_NAMES = {"double_top", "double_bottom"}
_REVERSAL_NAMES = _HS_NAMES | _DOUBLE_NAMES
# Multi-pivot trendline "container" shapes (triangles/wedges) plus the cup.
_RANGE_NAMES = {
    "ascending_triangle", "descending_triangle", "symmetric_triangle",
    "rising_wedge", "falling_wedge", "cup_and_handle",
}


def _exclusive_pair(pa: Pattern, pb: Pattern) -> bool:
    """True if these two pattern names are mutually exclusive when they overlap.

    Contradiction classes (see :func:`_dedup_mutual_exclusion`):
      1. symmetric-triangle vs a wedge (a clean triangle weakly resembles a wedge);
      2. a reversal (double_*/H&S) vs a triangle/wedge/cup — the textbook reading is
         one OR the other, never a bullish ascending triangle AND a bearish double
         top over the same pivots;
      3. two reversals over the same pivots where one is an H&S (e.g. an H&S whose
         two troughs also read as a double bottom).
    """
    names = {pa.name, pb.name}
    if "symmetric_triangle" in names and (names & _WEDGE_NAMES):
        return True
    if (names & _REVERSAL_NAMES) and (names & _RANGE_NAMES):
        return True
    if (names & _HS_NAMES) and (names & _DOUBLE_NAMES):
        return True
    return False


def _dedup_keep_first(pa: Pattern, pb: Pattern) -> bool:
    """For an exclusive overlapping pair, return True to keep ``pa`` (drop ``pb``).

    Precedence (in order):
      * a triangle/wedge/cup *container* beats an embedded reversal — the multi-
        pivot trendline structure is the more complete reading, and this is what
        makes a canonical ascending triangle report the TRIANGLE, not the bearish
        double top its flat-top touches incidentally satisfy;
      * an H&S beats a plain double top/bottom (the H&S is the more specific shape);
      * otherwise keep the higher-confidence pattern, breaking an exact tie toward
        the lexicographically smaller name for determinism.
    """
    a_range, b_range = pa.name in _RANGE_NAMES, pb.name in _RANGE_NAMES
    if a_range != b_range:
        return a_range  # keep whichever is the range/container shape
    a_hs, b_hs = pa.name in _HS_NAMES, pb.name in _HS_NAMES
    if a_hs != b_hs:
        return a_hs  # keep the H&S over the plain double
    if abs(pa.confidence - pb.confidence) < 1e-9:
        return pa.name <= pb.name
    return pa.confidence > pb.confidence


def _dedup_mutual_exclusion(patterns: list[Pattern]) -> list[Pattern]:
    """Drop the weaker of two overlapping contradictory shapes (shared >= 2 pivots).

    A clean symmetric triangle can weakly resemble a wedge; a textbook ascending
    triangle can incidentally satisfy a bearish double top over the same/overlapping
    pivots; an H&S's two troughs can read as a double bottom — each emitting two
    CONTRADICTORY readouts. For any such :func:`_exclusive_pair` sharing >= 2 pivots
    (overlapping span) we keep exactly ONE per :func:`_dedup_keep_first`: the
    multi-pivot triangle/wedge/cup container outranks an embedded reversal, an H&S
    outranks a plain double, and otherwise the higher-confidence shape wins. Scoped
    to overlapping spans so non-overlapping shapes elsewhere still both report.
    """
    drop: set[int] = set()
    for a in range(len(patterns)):
        if a in drop:
            continue
        for b in range(a + 1, len(patterns)):
            if b in drop:
                continue
            pa, pb = patterns[a], patterns[b]
            if not _exclusive_pair(pa, pb):
                continue
            if not _overlaps(pa, pb):
                continue
            drop.add(b if _dedup_keep_first(pa, pb) else a)
    return [p for i, p in enumerate(patterns) if i not in drop]


def detect(prices: pd.DataFrame, *, timeframe: str = "1d") -> list[Pattern]:
    """Detect chart patterns on ``prices`` at one ``timeframe`` (1d/1w/1mo).

    Pure; never raises (the whole body is wrapped in ``try/except`` -> ``[]``,
    belt-and-suspenders fail-soft matching the provider/engine ethos). Resamples
    internally, requires enough bars and >= 3 pivots, runs the dispatch list,
    applies mutual-exclusion de-dup and the :data:`MIN_CONFIDENCE` floor, stamps
    the timeframe on each result, and returns them sorted by confidence DESC then
    start ASC. The returned list is picklable (frozen dataclasses).
    """
    try:
        frame = resample_ohlc(prices, timeframe)
        min_bars = MIN_BARS_FOR_PATTERNS.get(timeframe, 10)
        if frame is None or len(frame) < min_bars:
            return []
        pivots = find_pivots(frame, timeframe=timeframe)
        if len(pivots) < 3:
            return []
        found: list[Pattern] = []
        for detector in _DISPATCH:
            pat = detector(frame, pivots)
            if pat is not None:
                found.append(pat)
        found = _dedup_mutual_exclusion(found)
        found = [p for p in found if p.confidence >= MIN_CONFIDENCE]
        # Frozen dataclass -> use replace to stamp the timeframe.
        import dataclasses

        found = [dataclasses.replace(p, timeframe=timeframe) for p in found]
        found.sort(key=lambda p: (-p.confidence, p.start))
        return found
    except Exception:
        return []


def detect_all_timeframes(daily_df: pd.DataFrame) -> dict[str, list[Pattern]]:
    """Detect across weekly / daily / monthly from one daily frame.

    Pure; resamples internally. Returns an ordered dict with EXACTLY the keys
    ``("1w", "1d", "1mo")`` in that order (weekly first as the higher-timeframe
    context, then daily, then monthly). On empty/short/None input every value is
    ``[]`` (each :func:`detect` call fail-softs) — the 3-key shape is always
    returned, so callers can iterate a stable structure. The result is picklable.
    """
    return {tf: detect(daily_df, timeframe=tf) for tf in ("1w", "1d", "1mo")}
