"""Backtest on out-of-sample probabilities, one position at a time, results in R-multiples."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .labels import AMBIGUOUS, DOWN, TIMEOUT, UP, Barriers


def pick_side(p: np.ndarray, cfg: Config) -> int:
    """+1 long, -1 short, 0 no trade. p = [P(down first), P(none), P(up first)]."""
    if np.isnan(p).any():
        return 0
    p_dn, p_up = p[0], p[2]
    if p_up >= cfg.thr and p_up - p_dn >= cfg.margin:
        return 1
    if p_dn >= cfg.thr and p_dn - p_up >= cfg.margin:
        return -1
    return 0


@dataclass
class Result:
    trades: pd.DataFrame
    stats: dict


def trade_r(side: int, b: Barriers, t: int, stop_frac: float, cost: float) -> float:
    """R-multiple of one trade (1R = distance to stop), net of costs."""
    out, e = b.outcome[t], b.entry[t]
    if out == AMBIGUOUS:
        gross = -stop_frac
    elif out == TIMEOUT:
        gross = side * (b.exit_price[t] - e) / e
    else:
        gross = stop_frac if out == (UP if side == 1 else DOWN) else -stop_frac
    return (gross - cost) / stop_frac


def run_backtest(df, b: Barriers, atr_s, proba: np.ndarray, cfg: Config, sides=None,
                 gate=None) -> Result:
    """gate = (allow_long, allow_short) boolean arrays (SOP); ignored when `sides` is given."""
    cost = cfg.cost_bps / 1e4
    a = atr_s.to_numpy(float)
    n = len(df)
    rows, busy_until = [], 0
    for t in range(n):
        if t < busy_until or not b.valid[t]:
            continue
        side = sides[t] if sides is not None else pick_side(proba[t], cfg)
        if sides is None and gate is not None:
            if (side == 1 and not gate[0][t]) or (side == -1 and not gate[1][t]):
                continue
        if side == 0:
            continue
        stop_frac = cfg.tp_mult * a[t] / b.entry[t]
        rows.append(
            dict(
                time=df.index[t],
                side=side,
                entry=b.entry[t],
                outcome=int(b.outcome[t]),
                r=trade_r(side, b, t, stop_frac, cost),
                stop_frac=stop_frac,
            )
        )
        busy_until = b.exit_idx[t]
    trades = pd.DataFrame(rows)
    return Result(trades, summarize(trades, cfg))


def summarize(trades: pd.DataFrame, cfg: Config) -> dict:
    if trades.empty:
        return dict(trades=0)
    r = trades["r"]
    equity = (1 + cfg.risk * r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(
        trades=len(r),
        win_rate=float((r > 0).mean()),
        expectancy_r=float(r.mean()),
        profit_factor=float(gains / losses) if losses > 0 else float("inf"),
        total_return=float(equity.iloc[-1] - 1),
        max_drawdown=float(dd),
        long_share=float((trades["side"] == 1).mean()),
    )


def random_baseline(df, b, atr_s, proba, cfg: Config, runs: int = 30, gate=None) -> float:
    """Mean expectancy (R) when the model's trade timing is kept but sides are random."""
    rng = np.random.default_rng(cfg.seed)
    base = np.array([pick_side(p, cfg) for p in proba])
    if gate is not None:
        base = np.where(((base == 1) & ~gate[0]) | ((base == -1) & ~gate[1]), 0, base)
    exp = []
    for _ in range(runs):
        sides = np.where(base != 0, rng.choice([-1, 1], size=len(base)), 0)
        res = run_backtest(df, b, atr_s, proba, cfg, sides=sides)
        if res.stats["trades"]:
            exp.append(res.stats["expectancy_r"])
    return float(np.mean(exp)) if exp else float("nan")
