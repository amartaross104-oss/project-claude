"""Do textbook candlestick / double top-bottom patterns have an edge? (run: python scripts/pattern_study.py)

Each pattern is traded in its textbook direction at the NEXT bar's open with a 1.5 ATR take-profit / stop
(triple-barrier, same as the rest of the project), net of costs, one position at a time.
With many (pattern x dataset) tests some will look 'significant' by luck, so the script also prints how many
to expect by chance and a Bonferroni threshold.
"""
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signalbot import data  # noqa: E402
from signalbot.backtest import run_backtest  # noqa: E402
from signalbot.config import Config, default_cost_bps  # noqa: E402
from signalbot.features import atr  # noqa: E402
from signalbot.health import t_stat  # noqa: E402
from signalbot.labels import barrier_outcomes  # noqa: E402
from signalbot.patterns import ALL_PATTERNS, all_patterns  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT", "EURUSD", "GBPUSD"]
INTERVALS = ["1h", "4h", "1d"]
MIN_TRADES = 30

rows, pooled = [], {p: [] for p in ALL_PATTERNS}
for sym in SYMBOLS:
    for itv in INTERVALS:
        try:
            df = data.load(sym, itv)
        except Exception as e:  # missing cache / network
            print("skip", sym, itv, e)
            continue
        cfg = Config(symbol=sym, interval=itv, cost_bps=default_cost_bps(sym))
        a = atr(df)
        b = barrier_outcomes(df, a, cfg.tp_mult, cfg.horizon)
        dummy = np.zeros((len(df), 3))
        for name, sig in all_patterns(df).items():
            res = run_backtest(df, b, a, dummy, cfg, sides=sig)
            tr = res.trades
            if len(tr) < MIN_TRADES:
                continue
            r = tr["r"].to_numpy()
            rows.append(dict(sym=sym, itv=itv, pattern=name, n=len(r), win=(r > 0).mean(), exp=r.mean(), t=t_stat(r)))
            pooled[name].extend(r.tolist())

res = pd.DataFrame(rows)
N = len(res)
alpha_t = NormalDist().inv_cdf(1 - 0.05 / N)
print(f"\n{N} tests (pattern x dataset, >= {MIN_TRADES} trades each)")
print(f"positive t >= 1.96: {(res.t >= 1.96).sum()}  (about {0.025 * N:.1f} expected by pure chance)")
print(f"negative t <= -1.96: {(res.t <= -1.96).sum()}")
print(f"Bonferroni threshold t >= {alpha_t:.2f}: {(res.t >= alpha_t).sum()} pass")
print(f"mean expectancy over all tests: {res.exp.mean():+.3f} R | share with expectancy > 0: {(res.exp > 0).mean():.0%}")
print("\nBest 8 by t-stat:")
print(res.sort_values("t", ascending=False).head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nWorst 4 by t-stat:")
print(res.sort_values("t").head(4).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nPooled per pattern (all datasets):")
pp = []
for k, v in pooled.items():
    if len(v) >= 100:
        v = np.asarray(v)
        pp.append(dict(pattern=k, n=len(v), exp=v.mean(), t=t_stat(v)))
print(pd.DataFrame(pp).sort_values("t", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
