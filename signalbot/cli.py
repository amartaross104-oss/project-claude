from __future__ import annotations

import argparse
import sys

import numpy as np

from . import data
from .backtest import random_baseline, run_backtest
from .config import Config
from .model import fit_final, predict_last, walk_forward
from .notify import find_chat_ids, send_telegram
from .features import atr
from .health import assess
from .model import walk_forward as _wf  # noqa: F401
from .signals import Signal, check_once, compute_signal, watch
from .sop import SOP, evaluate as sop_evaluate, gate as sop_gate
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
    g = None if a.no_sop else sop_gate(sop_evaluate(ds.df, ds.atr, cfg, SOP()))
    res = run_backtest(ds.df, ds.barriers, ds.atr, proba, cfg, gate=g)
    s = res.stats
    print(f"\n=== {cfg.symbol} {cfg.interval} | walk-forward OOS from {ds.df.index[oos][0]:%Y-%m-%d} "
          f"({oos.sum()} bars) | cost {cfg.cost_bps:g} bps/trade | thr {cfg.thr} margin {cfg.margin} | SOP {'off' if a.no_sop else 'on'} ===")
    if not s["trades"]:
        print("No trades. Try lowering --margin.")
        return
    print(f"trades        {s['trades']}   (long share {s['long_share']:.0%})")
    print(f"win rate      {s['win_rate']:.1%}")
    print(f"expectancy    {s['expectancy_r']:+.3f} R / trade")
    print(f"profit factor {s['profit_factor']:.2f}")
    print(f"return        {s['total_return']:+.1%}  (risk {cfg.risk:.0%}/trade, compounded)")
    print(f"max drawdown  {s['max_drawdown']:.1%}")
    base = random_baseline(ds.df, ds.barriers, ds.atr, proba, cfg, gate=g)
    print(f"random-side baseline expectancy {base:+.3f} R  "
          f"(model edge over random: {s['expectancy_r'] - base:+.3f} R)")
    h = assess(ds, proba, cfg, g)
    print("HEALTH GATE:", "PASS - signals would be sent" if h.ok
          else "FAIL - signals would be blocked: " + "; ".join(h.reasons))


def cmd_signal(a):
    sig = compute_signal(_cfg(a), sop=None if a.no_sop else SOP(), health_gate=not a.no_health_gate)
    print(sig.message())
    if a.notify and sig.side:
        send_telegram(sig.message())
        print("sent to Telegram")


def cmd_watch(a):
    cfgs = [_cfg(a, sym) for sym in a.symbols]
    sop, hg = (None if a.no_sop else SOP()), not a.no_health_gate
    send = (lambda m: print("[dry-run]\n" + m)) if a.dry_run else send_telegram
    if a.once:
        check_once(cfgs, send, persist=not a.dry_run, sop=sop, health_gate=hg)
    else:
        watch(cfgs, send, persist=not a.dry_run, sop=sop, health_gate=hg)


def cmd_health(a):
    cfg = _cfg(a)
    ds = build(cfg, refresh=a.refresh)
    proba = _wf(ds.X, ds.y, ds.barriers.valid, cfg)
    g = None if a.no_sop else sop_gate(sop_evaluate(ds.df, ds.atr, cfg, SOP()))
    h = assess(ds, proba, cfg, g)
    print(f"{cfg.symbol} {cfg.interval}: {h.line()} | random-side {h.baseline:+.3f} R")
    if h.recent:
        print(f"last 180d: {h.recent.get('trades', 0)} trades, "
              f"{h.recent.get('expectancy_r', float('nan')):+.3f} R/trade")
    print("PASS - signals would be sent" if h.ok else "FAIL - signals blocked: " + "; ".join(h.reasons))


def demo_text(cfg: Config, df, atr_s, side: int) -> str:
    close = float(df["close"].iloc[-1])
    dist = cfg.tp_mult * float(atr_s.iloc[-1])
    sig = Signal(cfg.symbol, cfg.interval, df.index[-1], side, close, close + side * dist,
                 close - side * dist, None, cfg.horizon, cfg.risk, None,
                 "TEST ONLY: levels from live price + ATR. No model, no SOP, no edge checked.")
    return "TEST ONLY - NOT A REAL SIGNAL - DO NOT TRADE\n" + sig.message()


def cmd_demo(a):
    cfg = _cfg(a)
    df = data.load(cfg.symbol, cfg.interval, refresh=True)  # incremental: only new bars
    text = demo_text(cfg, df, atr(df), 1 if a.side == "long" else -1)
    print(text)
    if a.dry_run:
        return
    send_telegram(text)
    print("\nTest message sent. Check your Telegram.")


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

    def gates(sp):
        sp.add_argument("--no-sop", action="store_true", help="disable the SOP rule checklist")
        sp.add_argument("--no-health-gate", action="store_true",
                        help="send signals even if the backtest shows no edge (NOT recommended)")

    for name, fn in (("fetch", cmd_fetch), ("backtest", cmd_backtest), ("signal", cmd_signal)):
        sp = sub.add_parser(name)
        sp.add_argument("symbol", help="e.g. BTCUSDT, ETHUSDT, EURUSD, GBPUSD, USDJPY")
        common(sp)
        if name == "backtest":
            sp.add_argument("--refresh", action="store_true", help="update data (new bars only)")
            sp.add_argument("--no-sop", action="store_true", help="disable the SOP rule checklist")
        if name == "signal":
            sp.add_argument("--notify", action="store_true", help="send via Telegram")
            gates(sp)
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("watch", help="check at every bar close and push new signals to Telegram")
    sp.add_argument("symbols", nargs="+", help="e.g. BTCUSDT ETHUSDT EURUSD")
    common(sp)
    gates(sp)
    sp.add_argument("--once", action="store_true", help="check once and exit (for cron)")
    sp.add_argument("--dry-run", action="store_true", help="print instead of sending to Telegram")
    sp.set_defaults(fn=cmd_watch)
    sp = sub.add_parser("health", help="show whether the strategy currently passes the health gate")
    sp.add_argument("symbol")
    common(sp)
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--no-sop", action="store_true")
    sp.set_defaults(fn=cmd_health)
    sp = sub.add_parser("demo", help="send a clearly-labelled TEST signal to Telegram (not a real signal)")
    sp.add_argument("symbol", nargs="?", default="BTCUSDT")
    common(sp)
    sp.add_argument("--side", choices=["long", "short"], default="long")
    sp.add_argument("--dry-run", action="store_true", help="print only, do not send")
    sp.set_defaults(fn=cmd_demo)
    sub.add_parser("telegram-test", help="send a test message").set_defaults(fn=cmd_telegram_test)
    sub.add_parser("chatid", help="find your TELEGRAM_CHAT_ID").set_defaults(fn=cmd_chatid)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
