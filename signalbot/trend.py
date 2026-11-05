"""Rule-based Donchian trend-following (no fitted parameters).

Entry : close breaks above the highest high (long) / below the lowest low (short) of the previous
        `entry_n` bars -> enter at the NEXT bar's open.
Exit  : stop at `stop_mult` x ATR from entry (checked intraday; gaps fill at the open), or the close
        crosses the previous `exit_n`-bar opposite channel -> exit at the next bar's open.
Result: R-multiples (1R = initial stop distance), net of round-trip cost. One position at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import atr


@dataclass(frozen=True)
class TrendParams:
    entry_n: int = 55
    exit_n: int = 20
    atr_n: int = 20
    stop_mult: float = 2.0


TRADE_COLUMNS = ["time", "exit_time", "side", "entry", "exit", "r", "stop_frac", "bars", "open",
                 "exit_pending"]
SYSTEM_1 = TrendParams(20, 10)   # classic "Turtle" system 1
SYSTEM_2 = TrendParams(55, 20)   # classic "Turtle" system 2


def breakout_levels(df: pd.DataFrame, p: TrendParams):
    """Channels over the PREVIOUS n bars (current bar excluded -> no look-ahead)."""
    up = df["high"].shift(1).rolling(p.entry_n).max()
    dn = df["low"].shift(1).rolling(p.entry_n).min()
    ex_lo = df["low"].shift(1).rolling(p.exit_n).min()
    ex_hi = df["high"].shift(1).rolling(p.exit_n).max()
    return up, dn, ex_lo, ex_hi


def signal_at(df: pd.DataFrame, p: TrendParams, idx: int = -1) -> int:
    """+1 / -1 if the bar at `idx` closed through the entry channel, else 0."""
    up, dn, _, _ = breakout_levels(df, p)
    c = df["close"].iloc[idx]
    if np.isfinite(up.iloc[idx]) and c > up.iloc[idx]:
        return 1
    if np.isfinite(dn.iloc[idx]) and c < dn.iloc[idx]:
        return -1
    return 0


def run_trend(df: pd.DataFrame, p: TrendParams, cost_bps: float) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr(df, p.atr_n).to_numpy(float)
    up, dn, ex_lo, ex_hi = (x.to_numpy(float) for x in breakout_levels(df, p))
    n, cost = len(df), cost_bps / 1e4
    rows, t = [], 0
    while t < n - 1:
        side = 0
        if np.isfinite(up[t]) and np.isfinite(a[t]):
            side = 1 if c[t] > up[t] else (-1 if c[t] < dn[t] else 0)
        if side == 0:
            t += 1
            continue
        entry, e_idx = o[t + 1], t + 1
        stop_dist = p.stop_mult * a[t]
        stop = entry - side * stop_dist
        exit_px, x_idx, is_open, exit_pending = c[n - 1], n - 1, True, False  # open: mark to last close
        for j in range(e_idx, n):
            hit = (l[j] <= stop) if side == 1 else (h[j] >= stop)
            if hit:
                gap = (o[j] <= stop) if side == 1 else (o[j] >= stop)
                exit_px, x_idx, is_open = (o[j] if gap and j > e_idx else stop), j, False
                break
            out = (c[j] < ex_lo[j]) if side == 1 else (c[j] > ex_hi[j])
            if np.isfinite(ex_lo[j]) and out:
                if j + 1 < n:
                    exit_px, x_idx, is_open = o[j + 1], j + 1, False
                else:
                    exit_pending = True   # channel exit triggered on the last bar; fills at next open
                break
        stop_frac = stop_dist / entry
        r = (side * (exit_px - entry) / entry - cost) / stop_frac
        rows.append(dict(time=df.index[t], exit_time=df.index[x_idx], side=side, entry=entry,
                         exit=exit_px, r=r, stop_frac=stop_frac, bars=x_idx - e_idx + 1,
                         open=is_open, exit_pending=exit_pending))
        t = max(x_idx, t + 1)                          # next signal can only come after the exit
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def position_open(trades: pd.DataFrame) -> bool:
    """True if the last simulated trade is still running (so a new breakout is NOT a fresh entry)."""
    if trades.empty:
        return False
    last = trades.iloc[-1]
    return bool(last["open"]) and not bool(last["exit_pending"])


def fresh_signal(df: pd.DataFrame, p: TrendParams, cost_bps: float) -> int:
    """Entry signal on the latest closed bar, only if no simulated position is already running."""
    trades = run_trend(df, p, cost_bps)
    if position_open(trades):
        return 0
    return signal_at(df, p, -1)
