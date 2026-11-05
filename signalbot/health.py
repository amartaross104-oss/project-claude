"""Strategy health gate: is the strategy currently showing an edge out-of-sample, after costs?"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import random_baseline, run_backtest, summarize
from .config import Config


@dataclass
class Criteria:
    min_trades: int = 50
    min_profit_factor: float = 1.2
    min_expectancy_r: float = 0.0
    min_edge_over_random_r: float = 0.0
    recent_days: int = 180
    min_recent_trades: int = 15
    min_recent_expectancy_r: float = 0.0


@dataclass
class Health:
    ok: bool
    reasons: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    recent: dict = field(default_factory=dict)
    baseline: float = float("nan")

    def line(self) -> str:
        s = self.stats
        if not s.get("trades"):
            return "Backtest: no trades"
        return (f"Backtest OOS: {s['trades']} trades, PF {s['profit_factor']:.2f}, "
                f"{s['expectancy_r']:+.2f} R/trade, win {s['win_rate']:.0%}")


def assess(ds, proba, cfg: Config, gate, crit: Criteria | None = None) -> Health:
    crit = crit or Criteria()
    res = run_backtest(ds.df, ds.barriers, ds.atr, proba, cfg, gate=gate)
    s = res.stats
    if not s.get("trades"):
        return Health(False, ["no out-of-sample trades"], s)
    base = random_baseline(ds.df, ds.barriers, ds.atr, proba, cfg, gate=gate)
    cut = res.trades["time"].max() - pd.Timedelta(days=crit.recent_days)
    recent = summarize(res.trades[res.trades["time"] >= cut], cfg)

    why = []
    if s["trades"] < crit.min_trades:
        why.append(f"only {s['trades']} OOS trades (need {crit.min_trades})")
    if s["profit_factor"] < crit.min_profit_factor:
        why.append(f"profit factor {s['profit_factor']:.2f} < {crit.min_profit_factor}")
    if s["expectancy_r"] <= crit.min_expectancy_r:
        why.append(f"expectancy {s['expectancy_r']:+.3f} R <= {crit.min_expectancy_r}")
    if not np.isnan(base) and s["expectancy_r"] - base <= crit.min_edge_over_random_r:
        why.append(f"no edge over random sides ({s['expectancy_r'] - base:+.3f} R)")
    if recent.get("trades", 0) < crit.min_recent_trades:
        why.append(f"only {recent.get('trades', 0)} trades in last {crit.recent_days}d")
    elif recent["expectancy_r"] <= crit.min_recent_expectancy_r:
        why.append(f"last {crit.recent_days}d expectancy {recent['expectancy_r']:+.3f} R")
    return Health(not why, why, s, recent, base)
