from __future__ import annotations

import argparse
import sys

import numpy as np

from . import data
from .backtest import pick_side, random_baseline, run_backtest
from .config import Config
from .model import fit_final, predict_last, walk_forward
from .notify import send_telegram
from .pipeline import build


def _cfg(a) -> Config:
    return Config(symbol=a.symbol.upper(), interval=a.interval, thr=a.thr, margin=a.margin, tp_mult=a.tp_mult,
                  horizon=a.horizon, cost_bps=a.cost_bps)


def cmd_fetch(a):
    df = data.fetch(a.symbol, a.interval)
    print(f"{a.symbol.upper()}: {len(df)} bars  {df.index[0]} -> {df.index[-1]}")


def cmd_backtest(a):
    cfg = _cfg(a)
    ds = build(cfg, refresh=a.refresh)
    proba = walk_forward(ds.X, ds.y, ds.barriers.valid, cfg)
    oos = ~np.isnan(proba[:, 0])
    res = run_backtest(ds.df, ds.barriers, ds.atr, proba, cfg)
    s = res.stats
    print(f"\n=== {cfg.symbol} {cfg.interval} | walk-forward OOS from {ds.df.index[oos][0]:%Y-%m-%d} "
          f"({oos.sum()} bars) | cost {cfg.cost_bps:g} bps/trade | thr {cfg.thr} margin {cfg.margin} ===")
    if not s["trades"]:
        print("No trades. Try lowering --margin.")
        return
    print(f"trades        {s['trades']}   (long share {s['long_share']:.0%})")
    print(f"win rate      {s['win_rate']:.1%}")
    print(f"expectancy    {s['expectancy_r']:+.3f} R / trade")
    print(f"profit factor {s['profit_factor']:.2f}")
    print(f"return        {s['total_return']:+.1%}  (risk {cfg.risk:.0%}/trade, compounded)")
    print(f"max drawdown  {s['max_drawdown']:.1%}")
    base = random_baseline(ds.df, ds.barriers, ds.atr, proba, cfg)
    print(f"random-side baseline expectancy {base:+.3f} R  "
          f"(model edge over random: {s['expectancy_r'] - base:+.3f} R)")


def cmd_signal(a):
    cfg = _cfg(a)
    ds = build(cfg, refresh=True)
    model = fit_final(ds.X, ds.y, ds.barriers.valid, cfg)
    p = predict_last(model, ds.X)
    side = pick_side(p, cfg)
    t = ds.df.index[-1]
    close = float(ds.df["close"].iloc[-1])
    dist = cfg.tp_mult * float(ds.atr.iloc[-1])
    head = f"{cfg.symbol} {cfg.interval} | bar closed {t:%Y-%m-%d %H:%M} UTC | close {close:.5g}"
    probs = f"P(down)={p[0]:.2f} P(none)={p[1]:.2f} P(up)={p[2]:.2f}"
    if side == 0:
        print(f"{head}\nNO TRADE  {probs}")
        return
    tp, sl = close + side * dist, close - side * dist
    msg = (f"{'LONG' if side == 1 else 'SHORT'} {cfg.symbol} ({cfg.interval})\n"
           f"Ref price {close:.5g}  TP {tp:.5g}  SL {sl:.5g}  max {cfg.horizon} bars\n"
           f"{probs}\nResearch signal, not financial advice. Risk max {cfg.risk:.0%} per trade.")
    print(f"{head}\n{msg}")
    if a.notify:
        send_telegram(msg)
        print("sent to Telegram")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="signalbot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fetch", cmd_fetch), ("backtest", cmd_backtest), ("signal", cmd_signal)):
        sp = sub.add_parser(name)
        sp.add_argument("symbol", help="e.g. BTCUSDT, ETHUSDT, EURUSD, GBPUSD, USDJPY")
        sp.add_argument("--interval", default="1h")
        sp.add_argument("--thr", type=float, default=0.45, help="min P(direction)")
        sp.add_argument("--margin", type=float, default=0.10,
                        help="min P(direction)-P(opposite); main trade filter")
        sp.add_argument("--tp-mult", type=float, default=1.5)
        sp.add_argument("--horizon", type=int, default=24)
        sp.add_argument("--cost-bps", type=float, default=None, help="round-trip cost in bps")
        if name == "backtest":
            sp.add_argument("--refresh", action="store_true", help="re-download data")
        if name == "signal":
            sp.add_argument("--notify", action="store_true", help="send via Telegram")
        sp.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
