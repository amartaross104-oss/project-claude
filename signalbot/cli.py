from __future__ import annotations

import argparse
import sys

import numpy as np

from . import data
from .backtest import random_baseline, run_backtest
from .config import Config
from .model import fit_final, predict_last, walk_forward
from .notify import find_chat_ids, send_telegram
from .signals import check_once, compute_signal, watch
from .pipeline import build


def _cfg(a, symbol: str | None = None) -> Config:
    return Config(symbol=(symbol or a.symbol).upper(), interval=a.interval, thr=a.thr, margin=a.margin, tp_mult=a.tp_mult,
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
    sig = compute_signal(_cfg(a))
    print(sig.message())
    if a.notify and sig.side:
        send_telegram(sig.message())
        print("sent to Telegram")


def cmd_watch(a):
    cfgs = [_cfg(a, sym) for sym in a.symbols]
    send = (lambda m: print("[dry-run]\n" + m)) if a.dry_run else send_telegram
    if a.once:
        check_once(cfgs, send, persist=not a.dry_run)
    else:
        watch(cfgs, send, persist=not a.dry_run)


def cmd_telegram_test(a):
    send_telegram("signalbot connected. You will receive trade signals here.")
    print("Test message sent. Check your Telegram.")


def cmd_chatid(a):
    chats = find_chat_ids()
    if not chats:
        print("No chats found. Open your bot in Telegram, press Start / send any message, then run again.")
    for cid, name in chats:
        print(f"TELEGRAM_CHAT_ID={cid}   ({name})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="signalbot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(sp):
        sp.add_argument("--interval", default="1h")
        sp.add_argument("--thr", type=float, default=0.45, help="min P(direction)")
        sp.add_argument("--margin", type=float, default=0.10,
                        help="min P(direction)-P(opposite); main trade filter")
        sp.add_argument("--tp-mult", type=float, default=1.5)
        sp.add_argument("--horizon", type=int, default=None,
                        help="max bars per trade (default per interval)")
        sp.add_argument("--cost-bps", type=float, default=None, help="round-trip cost in bps")

    for name, fn in (("fetch", cmd_fetch), ("backtest", cmd_backtest), ("signal", cmd_signal)):
        sp = sub.add_parser(name)
        sp.add_argument("symbol", help="e.g. BTCUSDT, ETHUSDT, EURUSD, GBPUSD, USDJPY")
        common(sp)
        if name == "backtest":
            sp.add_argument("--refresh", action="store_true", help="re-download data")
        if name == "signal":
            sp.add_argument("--notify", action="store_true", help="send via Telegram")
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("watch", help="check at every bar close and push new signals to Telegram")
    sp.add_argument("symbols", nargs="+", help="e.g. BTCUSDT ETHUSDT EURUSD")
    common(sp)
    sp.add_argument("--once", action="store_true", help="check once and exit (for cron)")
    sp.add_argument("--dry-run", action="store_true", help="print instead of sending to Telegram")
    sp.set_defaults(fn=cmd_watch)
    sub.add_parser("telegram-test", help="send a test message").set_defaults(fn=cmd_telegram_test)
    sub.add_parser("chatid", help="find your TELEGRAM_CHAT_ID").set_defaults(fn=cmd_chatid)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
