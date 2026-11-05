"""`trend` command: Donchian trend-following signals with a per-symbol, per-direction health gate."""
from __future__ import annotations

import json
import time

import numpy as np

from . import data
from .config import default_cost_bps
from .features import atr
from .health import t_stat
from .notify import send_telegram
from .signals import STATE_FILE, _load_state, next_run
from .trend import SYSTEM_1, SYSTEM_2, TrendParams, fresh_signal, run_trend

SYSTEMS = {"1": SYSTEM_1, "2": SYSTEM_2}


def gate_direction(trades, side: int, recent_days: int = 730, min_trades: int = 30):
    """Same standards as the ML health gate, per symbol and per direction."""
    t = trades[trades["side"] == side]
    if t.empty:
        return False, ["no trades"], {}
    r = t["r"].to_numpy()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    ts = t_stat(r)
    recent = t[t["time"] >= t["time"].max() - __import__("pandas").Timedelta(days=recent_days)]
    st = dict(n=len(r), pf=float(pf), exp=float(r.mean()), t=ts, win=float((r > 0).mean()),
              recent_n=len(recent), recent_exp=float(recent["r"].mean()) if len(recent) else float("nan"))
    why = []
    if st["n"] < min_trades:
        why.append(f"only {st['n']} trades (need {min_trades})")
    if pf < 1.2:
        why.append(f"profit factor {pf:.2f} < 1.2")
    if st["exp"] <= 0:
        why.append(f"expectancy {st['exp']:+.2f} R <= 0")
    if not ts >= 1.64:
        why.append(f"not statistically significant (t={ts:.2f} < 1.64)")
    if len(recent) < 5 or not st["recent_exp"] > 0:
        why.append(f"last {recent_days // 365}y: {len(recent)} trades, {st['recent_exp']:+.2f} R")
    return not why, why, st


def build_message(sym: str, p: TrendParams, df, side: int, st: dict, ok: bool, paper: bool,
                  risk: float = 0.01) -> str:
    close = float(df["close"].iloc[-1])
    a = float(atr(df, p.atr_n).iloc[-1])
    stop = close - side * p.stop_mult * a
    stop_frac = abs(close - stop) / close
    ex = (df["low"].shift(1).rolling(p.exit_n).min().iloc[-1] if side == 1
          else df["high"].shift(1).rolling(p.exit_n).max().iloc[-1])
    entry_ch = (df["high"].shift(1).rolling(p.entry_n).max().iloc[-1] if side == 1
                else df["low"].shift(1).rolling(p.entry_n).min().iloc[-1])
    side_name = "long" if side == 1 else "short"
    history = (f"History ({side_name}s on {sym}): {st['n']} trades, PF {st['pf']:.2f}, "
               f"{st['exp']:+.2f} R/trade (t={st['t']:.1f}), last 2y {st['recent_exp']:+.2f} R"
               if st else f"History: no completed {side_name} trades on {sym} to judge this signal")
    head = ("PAPER / UNVALIDATED - forward-test only, do not risk real money\n" if paper and not ok else "")
    verdict = "health gate PASSED" if ok else "health gate FAILED (paper mode)"
    return (f"{head}{'LONG' if side == 1 else 'SHORT'} {sym} (1d) TREND {p.entry_n}/{p.exit_n}\n"
            f"Close {close:.5g} broke the {p.entry_n}-day {'high' if side == 1 else 'low'} {entry_ch:.5g}\n"
            f"Enter next open (~{close:.5g}). Stop {stop:.5g} ({stop_frac:.1%} away, {p.stop_mult:g} ATR)\n"
            f"Exit: daily close {'below' if side == 1 else 'above'} the {p.exit_n}-day "
            f"{'low' if side == 1 else 'high'} (now {ex:.5g}), or the stop\n"
            f"Size: risk {risk:.0%} of equity -> position ~{risk / stop_frac:.1f}x equity\n"
            f"{history}\n"
            f"{verdict}. Research signal, not financial advice.")


def already_sent(state: dict, sym: str, system: str, bar_iso: str, side: int) -> bool:
    last = state.get(f"trend{system}_{sym}")
    return bool(last) and last["bar"] == bar_iso and last["side"] == side


def remember(state: dict, path, sym: str, system: str, bar_iso: str, side: int) -> None:
    state[f"trend{system}_{sym}"] = {"bar": bar_iso, "side": side}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state))


def check_trend(a, state_path=STATE_FILE, send=send_telegram):
    p = SYSTEMS[a.system]
    sent_any = False
    for sym in (s.upper() for s in a.symbols):
        df = data.load(sym, "1d", refresh=True)
        cost = default_cost_bps(sym)
        trades = run_trend(df, p, cost)
        side = fresh_signal(df, p, cost)
        print(f"\n{sym} 1d  bar {df.index[-1]:%Y-%m-%d}  close {df['close'].iloc[-1]:.5g}  "
              f"trend {p.entry_n}/{p.exit_n}")
        for sd, name in ((1, "long"), (-1, "short")):
            ok, why, st = gate_direction(trades, sd)
            if st:
                print(f"  {name:5s}: n={st['n']} PF {st['pf']:.2f} {st['exp']:+.2f}R t={st['t']:+.2f} "
                      f"last2y {st['recent_exp']:+.2f}R -> {'PASS' if ok else 'FAIL: ' + '; '.join(why)}")
        if side == 0:
            print("  no fresh breakout signal on the latest closed bar")
            continue
        ok, why, st = gate_direction(trades, side)
        if not ok and not a.paper:
            print(f"  signal {'LONG' if side == 1 else 'SHORT'} BLOCKED by health gate (use --paper to forward-test)")
            continue
        msg = build_message(sym, p, df, side, st, ok, a.paper)
        print("\n" + msg)
        if a.notify and not a.dry_run:
            state, bar = _load_state(state_path), df.index[-1].isoformat()
            if already_sent(state, sym, a.system, bar, side):
                print("  (already sent for this bar - skipping duplicate)")
                continue
            send(msg)
            remember(state, state_path, sym, a.system, bar, side)
            sent_any = True
            print("sent to Telegram")
    return sent_any


def cmd_trend(a):
    if not a.loop:
        check_trend(a)
        return
    print(f"watching {', '.join(a.symbols)} (daily, trend {a.system}); Ctrl+C to stop")
    while True:
        try:
            check_trend(a)
        except Exception as e:  # keep running through network/data errors
            print(f"[error] {e}")
        wake = next_run(time.time(), "1d", 120)   # daily bar closes 00:00 UTC (08:00 Malaysia time)
        print(f"\nnext check {time.strftime('%Y-%m-%d %H:%M', time.gmtime(wake))} UTC")
        time.sleep(max(1, wake - time.time()))


def add_parser(sub):
    sp = sub.add_parser("trend", help="Donchian trend-following signals (daily) with health gate")
    sp.add_argument("symbols", nargs="+", help="e.g. BTCUSDT ETHUSDT")
    sp.add_argument("--system", choices=list(SYSTEMS), default="2", help="1 = 20/10, 2 = 55/20")
    sp.add_argument("--paper", action="store_true",
                    help="also emit signals that fail the gate, clearly labelled PAPER/UNVALIDATED")
    sp.add_argument("--notify", action="store_true", help="send via Telegram")
    sp.add_argument("--dry-run", action="store_true", help="print only")
    sp.add_argument("--loop", action="store_true", help="keep running; re-check after each daily close")
    sp.set_defaults(fn=cmd_trend)
