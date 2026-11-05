"""Trade SOP: explicit, causal rules that a model signal must pass before it is sent.

Rules (all use only data up to the signal bar):
  trend      long only if EMA50 > EMA200 and close > EMA50 (short: the opposite)
  rsi        no long when RSI14 > 70, no short when RSI14 < 30 (avoid chasing extremes)
  volatility ATR% must sit between the 20th and 90th percentile of the last 500 bars
  cost       round-trip cost must be <= max_cost_r of the stop distance (R)
  session    forex only: entry bar must open 08:00-17:00 UTC (London / New York liquidity)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data import INTERVAL_MS, is_crypto
from .features import rsi


@dataclass
class SOP:
    trend: bool = True
    rsi: bool = True
    volatility: bool = True
    cost: bool = True
    session: bool = True
    max_cost_r: float = 0.15
    rsi_hi: float = 70.0
    rsi_lo: float = 30.0
    vol_lo_q: float = 0.20
    vol_hi_q: float = 0.90
    vol_window: int = 500
    session_start: int = 8
    session_end: int = 17


def evaluate(df: pd.DataFrame, atr_s: pd.Series, cfg: Config, sop: SOP) -> dict:
    """rule name -> (long_ok, short_ok) boolean arrays aligned to df."""
    n, c = len(df), df["close"]
    yes = np.ones(n, dtype=bool)
    rules: dict[str, tuple] = {}

    if sop.trend:
        e50, e200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
        warm = np.arange(n) >= 200
        rules["trend"] = (((e50 > e200) & (c > e50)).to_numpy() & warm,
                          ((e50 < e200) & (c < e50)).to_numpy() & warm)
    if sop.rsi:
        r = rsi(c)
        rules["rsi"] = ((r <= sop.rsi_hi).to_numpy(), (r >= sop.rsi_lo).to_numpy())
    if sop.volatility:
        ap = atr_s / c
        lo = ap.rolling(sop.vol_window).quantile(sop.vol_lo_q)
        hi = ap.rolling(sop.vol_window).quantile(sop.vol_hi_q)
        ok = ((ap >= lo) & (ap <= hi)).to_numpy()
        rules["volatility"] = (ok, ok)
    if sop.cost:
        stop_frac = cfg.tp_mult * atr_s / c
        ok = ((cfg.cost_bps / 1e4) / stop_frac <= sop.max_cost_r).to_numpy()
        rules["cost"] = (ok, ok)
    if sop.session and not is_crypto(cfg.symbol) and cfg.interval != "1d":
        entry_hour = (df.index + pd.Timedelta(milliseconds=INTERVAL_MS[cfg.interval])).hour
        ok = np.asarray((entry_hour >= sop.session_start) & (entry_hour < sop.session_end))
        rules["session"] = (ok, ok)
    return rules or {"none": (yes, yes)}


def gate(rules: dict) -> tuple[np.ndarray, np.ndarray]:
    longs = np.logical_and.reduce([v[0] for v in rules.values()])
    shorts = np.logical_and.reduce([v[1] for v in rules.values()])
    return longs, shorts


def checklist(rules: dict, side: int, idx: int = -1) -> dict[str, bool]:
    return {name: bool(v[0 if side == 1 else 1][idx]) for name, v in rules.items()}
