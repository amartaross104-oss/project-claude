"""Trade SOP: explicit, causal rules that a model signal must pass before it is sent.

Rules (all use only data up to the signal bar):
  trend      long only if EMA50 > EMA200 and close > EMA50 (short: the opposite)
  rsi        no long when RSI14 > 70, no short when RSI14 < 30 (avoid chasing extremes)
  volatility ATR% must sit between the 20th and 90th percentile of the last 500 bars
  cost       round-trip cost must be <= max_cost_r of the stop distance (R)
  session    forex only: entry bar must open 08:00-17:00 UTC (London / New York liquidity)

Extra rules taken from a third-party SOP document (generic ideas only; the proprietary "box" indicator
is NOT reproduced; mtf is a plain EMA proxy for "higher timeframe direction"):
  mtf        long only if the last COMPLETED higher-timeframe bar closed above its EMA20 (short: below)
             (entry 5m->1h, 15m->4h, 1h->1d, 4h->1d)
  squeeze    no entry while Bollinger(20, 2) bandwidth is in the lowest 25% of the last 200 bars (sideways)
  hours      entry bar must open inside 15:00-17:00 or 21:00-23:00 Malaysia time (07-09 and 13-15 UTC)
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
    # SOP-document rules (off by default so earlier behaviour is unchanged)
    mtf: bool = False
    squeeze: bool = False
    hours: bool = False
    htf_ema: int = 20
    bb_n: int = 20
    bb_k: float = 2.0
    squeeze_q: float = 0.25
    squeeze_window: int = 200
    hours_utc: tuple = ((7, 9), (13, 15))


HTF_FOR = {"5m": "1h", "15m": "4h", "1h": "1d", "4h": "1d"}
HTF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def sop_document_preset(**over) -> "SOP":
    """Only the rules from the SOP document, plus the cost check (all the other built-in rules off)."""
    base = dict(trend=False, rsi=False, volatility=False, cost=True, session=False,
                mtf=True, squeeze=True, hours=True)
    base.update(over)
    return SOP(**base)


def htf_direction(df: pd.DataFrame, interval: str, ema_span: int) -> np.ndarray | None:
    """+1 / -1 per base bar: direction of the last COMPLETED higher-timeframe bar (0 = unknown)."""
    htf = HTF_FOR.get(interval)
    if htf is None or INTERVAL_MS[interval] >= HTF_MS[htf]:
        return None
    c = df["close"].resample(htf).last().dropna()
    ema = c.ewm(span=ema_span, adjust=False).mean()
    sign = pd.Series(np.where(c > ema, 1, -1), index=c.index + pd.Timedelta(milliseconds=HTF_MS[htf]))
    sign.iloc[: ema_span] = 0                                # EMA warm-up
    closes = df.index + pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    both = sign.index.union(closes)
    out = sign.reindex(both).ffill().reindex(closes).fillna(0)  # value available once the HTF bar has closed
    return out.to_numpy(int)


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
    if sop.mtf:
        d = htf_direction(df, cfg.interval, sop.htf_ema)
        if d is not None:
            rules["mtf"] = (d == 1, d == -1)
    if sop.squeeze:
        mid = c.rolling(sop.bb_n).mean()
        bw = (2 * sop.bb_k * c.rolling(sop.bb_n).std()) / mid
        thr = bw.rolling(sop.squeeze_window).quantile(sop.squeeze_q)
        ok = (bw > thr).to_numpy()                            # NaN warm-up compares False -> blocked
        rules["squeeze"] = (ok, ok)
    if sop.hours:
        entry_hour = (df.index + pd.Timedelta(milliseconds=INTERVAL_MS[cfg.interval])).hour
        ok = np.zeros(n, dtype=bool)
        for lo, hi in sop.hours_utc:
            ok |= np.asarray((entry_hour >= lo) & (entry_hour < hi))
        rules["hours"] = (ok, ok)
    return rules or {"none": (yes, yes)}


def gate(rules: dict) -> tuple[np.ndarray, np.ndarray]:
    longs = np.logical_and.reduce([v[0] for v in rules.values()])
    shorts = np.logical_and.reduce([v[1] for v in rules.values()])
    return longs, shorts


def checklist(rules: dict, side: int, idx: int = -1) -> dict[str, bool]:
    return {name: bool(v[0 if side == 1 else 1][idx]) for name, v in rules.items()}
