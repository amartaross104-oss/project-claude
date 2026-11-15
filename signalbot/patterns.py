"""Classic candlestick and chart-pattern detectors (textbook definitions, causal).

Every signal at bar t uses only bars <= t (confirmed at the bar close), +1 = bullish, -1 = bearish.
Context rules follow the textbooks: reversal patterns need a preceding 5-bar move in the opposite direction.
Only objectively definable patterns are implemented (candlesticks, double top/bottom); subjective ones
(flags, wedges, triangles, cup & handle, head & shoulders) are not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr as atr_fn

CANDLE_PATTERNS = ["hammer", "inverted_hammer", "shooting_star", "hanging_man", "bull_engulfing",
                   "bear_engulfing", "morning_star", "evening_star", "three_white_soldiers",
                   "three_black_crows", "tweezer_bottom", "tweezer_top", "bull_marubozu", "bear_marubozu"]
CHART_PATTERNS = ["double_bottom", "double_top"]
ALL_PATTERNS = CANDLE_PATTERNS + CHART_PATTERNS


def _sig(mask, direction: int) -> np.ndarray:
    return np.where(np.asarray(mask, dtype=bool), direction, 0).astype(np.int8)


def candlestick_patterns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    body, rng = (c - o).abs(), (h - l)
    top, bot = np.maximum(o, c), np.minimum(o, c)
    upper, lower = h - top, bot - l
    avg_body = body.rolling(14).mean()
    atr = atr_fn(df)
    bull, bear = c > o, c < o
    down = c.shift(1) < c.shift(6)           # prior 5-bar decline
    up = c.shift(1) > c.shift(6)
    small = body <= 0.35 * rng               # small real body
    hammer_shape = small & (lower >= 2 * body) & (upper <= 0.15 * rng) & (rng > 0)
    star_shape = small & (upper >= 2 * body) & (lower <= 0.15 * rng) & (rng > 0)

    p1o, p1c = o.shift(1), c.shift(1)
    p2o, p2c = o.shift(2), c.shift(2)
    out = {
        "hammer": _sig(hammer_shape & down, 1),
        "inverted_hammer": _sig(star_shape & down, 1),
        "shooting_star": _sig(star_shape & up, -1),
        "hanging_man": _sig(hammer_shape & up, -1),
        "bull_engulfing": _sig(bull & (p1c < p1o) & (o <= p1c) & (c >= p1o) & down, 1),
        "bear_engulfing": _sig(bear & (p1c > p1o) & (o >= p1c) & (c <= p1o) & up, -1),
        "morning_star": _sig((p2c < p2o) & (body.shift(2) > avg_body.shift(2)) & (body.shift(1) < 0.5 * avg_body.shift(1))
                             & bull & (c > (p2o + p2c) / 2), 1),
        "evening_star": _sig((p2c > p2o) & (body.shift(2) > avg_body.shift(2)) & (body.shift(1) < 0.5 * avg_body.shift(1))
                             & bear & (c < (p2o + p2c) / 2), -1),
        "three_white_soldiers": _sig(bull & (p1c > p1o) & (p2c > p2o) & (c > p1c) & (p1c > p2c)
                                     & (o > p1o) & (o < p1c) & (p1o > p2o) & (p1o < p2c) & (body > 0.5 * avg_body), 1),
        "three_black_crows": _sig(bear & (p1c < p1o) & (p2c < p2o) & (c < p1c) & (p1c < p2c)
                                  & (o < p1o) & (o > p1c) & (p1o < p2o) & (p1o > p2c) & (body > 0.5 * avg_body), -1),
        "tweezer_bottom": _sig(bull & (p1c < p1o) & ((l - l.shift(1)).abs() <= 0.05 * atr) & down, 1),
        "tweezer_top": _sig(bear & (p1c > p1o) & ((h - h.shift(1)).abs() <= 0.05 * atr) & up, -1),
        "bull_marubozu": _sig(bull & (body >= 0.9 * rng) & (body >= 1.2 * avg_body), 1),
        "bear_marubozu": _sig(bear & (body >= 0.9 * rng) & (body >= 1.2 * avg_body), -1),
    }
    for k in out:                            # warm-up bars never signal
        out[k][:20] = 0
    return out


def _swing_points(x: np.ndarray, k: int, high: bool) -> np.ndarray:
    """Index i is a swing point if x[i] is the extreme of x[i-k .. i+k] (strict on the left, non-strict on the
    right, so a flat top/bottom is counted once). It is CONFIRMED at bar i+k."""
    n = len(x)
    idx = []
    for i in range(k, n - k):
        left, right = x[i - k: i], x[i + 1: i + k + 1]
        if high and x[i] > left.max() and x[i] >= right.max():
            idx.append(i)
        elif not high and x[i] < left.min() and x[i] <= right.min():
            idx.append(i)
    return np.array(idx, dtype=int)


def double_top_bottom(df: pd.DataFrame, k: int = 5, min_gap: int = 5, max_gap: int = 60,
                      tol_atr: float = 0.5, depth_atr: float = 1.5, max_age: int = 30) -> dict[str, np.ndarray]:
    """Double top/bottom confirmed by the first CLOSE through the neckline (the trough/peak between the two extremes)."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    a = atr_fn(df).to_numpy(float)
    n = len(df)
    out = {"double_bottom": np.zeros(n, np.int8), "double_top": np.zeros(n, np.int8)}
    for name, x, high, direction in (("double_top", h, True, -1), ("double_bottom", l, False, 1)):
        sw = _swing_points(x, k, high)
        used = set()
        for t in range(n):
            # swings confirmed by bar t: index i with i + k <= t
            conf = sw[sw + k <= t]
            if len(conf) < 2 or not np.isfinite(a[t]):
                continue
            i1, i2 = conf[-2], conf[-1]
            if not (min_gap <= i2 - i1 <= max_gap) or t - (i2 + k) > max_age or (i1, i2) in used:
                continue
            if abs(x[i1] - x[i2]) > tol_atr * a[t]:
                continue
            seg = slice(i1, i2 + 1)
            neck = l[seg].min() if high else h[seg].max()
            depth = (min(x[i1], x[i2]) - neck) if high else (neck - max(x[i1], x[i2]))
            if depth < depth_atr * a[t]:
                continue
            broke = (c[t] < neck) if high else (c[t] > neck)
            was_inside = (c[t - 1] >= neck) if high else (c[t - 1] <= neck)
            if broke and was_inside:
                out[name][t] = direction
                used.add((i1, i2))
    return out


def all_patterns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {**candlestick_patterns(df), **double_top_bottom(df)}
