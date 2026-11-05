"""Triple-barrier outcomes.

Signal is generated on the close of bar t, the trade enters at the OPEN of bar t+1 and
runs for at most `horizon` bars. Barriers: entry +/- tp_mult * ATR(t).
If both barriers fall inside one candle we cannot know the order, so it is marked
ambiguous and treated as a loss for whoever is trading (conservative).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP, DOWN, TIMEOUT, AMBIGUOUS = 1, -1, 0, 2


@dataclass
class Barriers:
    outcome: np.ndarray     # int8: UP / DOWN / TIMEOUT / AMBIGUOUS
    exit_idx: np.ndarray    # bar index where the trade closes
    exit_price: np.ndarray  # price (barrier price, or close of last bar on timeout)
    entry: np.ndarray       # entry price (open of t+1)
    valid: np.ndarray       # False where there is not enough future data / ATR


def barrier_outcomes(df, atr_s, tp_mult: float, horizon: int) -> Barriers:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr_s.to_numpy(float)
    n = len(df)
    outcome = np.zeros(n, dtype=np.int8)
    exit_idx = np.zeros(n, dtype=np.int64)
    exit_price = np.full(n, np.nan)
    entry = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for t in range(n - horizon - 1):
        if not np.isfinite(a[t]):
            continue
        valid[t] = True
        e = o[t + 1]
        up, dn = e + tp_mult * a[t], e - tp_mult * a[t]
        entry[t] = e
        end = t + horizon  # last bar index of the trade
        hi, lo = h[t + 1 : end + 1], l[t + 1 : end + 1]
        hit = (hi >= up) | (lo <= dn)
        if not hit.any():
            outcome[t], exit_idx[t], exit_price[t] = TIMEOUT, end, c[end]
            continue
        j = int(hit.argmax())
        both = hi[j] >= up and lo[j] <= dn
        k = t + 1 + j
        exit_idx[t] = k
        if both:
            outcome[t], exit_price[t] = AMBIGUOUS, np.nan
        elif hi[j] >= up:
            outcome[t], exit_price[t] = UP, up
        else:
            outcome[t], exit_price[t] = DOWN, dn
    return Barriers(outcome, exit_idx, exit_price, entry, valid)


def to_class(outcome: np.ndarray) -> np.ndarray:
    """0 = DOWN hit first, 1 = nothing / ambiguous, 2 = UP hit first."""
    y = np.ones(len(outcome), dtype=np.int64)
    y[outcome == UP] = 2
    y[outcome == DOWN] = 0
    return y
