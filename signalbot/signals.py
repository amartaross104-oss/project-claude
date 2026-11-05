"""Compute the latest signal and run the periodic watcher (with de-duplication)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .backtest import pick_side
from .config import Config
from .health import assess
from . import sop as sop_mod
from .data import DATA_DIR, INTERVAL_MS, is_crypto
from .model import fit_final, predict_last, walk_forward
from .pipeline import build

STATE_FILE = DATA_DIR / "last_signals.json"


@dataclass
class Signal:
    symbol: str
    interval: str
    bar_time: object
    side: int            # +1 long, -1 short, 0 none
    close: float
    tp: float | None
    sl: float | None
    proba: tuple
    horizon: int
    risk: float
    checks: dict | None = None     # SOP checklist for the chosen side
    health_line: str = ""
    blocked: str = ""              # why a model signal was NOT sent

    def head(self) -> str:
        return (f"{self.symbol} {self.interval} | bar closed {self.bar_time:%Y-%m-%d %H:%M} UTC"
                f" | close {self.close:.5g}")

    def probs(self) -> str:
        if self.proba is None:
            return "Model probabilities: n/a (test message)"
        return f"P(down)={self.proba[0]:.2f} P(none)={self.proba[1]:.2f} P(up)={self.proba[2]:.2f}"

    def message(self) -> str:
        if self.side == 0:
            why = f"  [{self.blocked}]" if self.blocked else ""
            return f"{self.head()}\nNO TRADE{why}  {self.probs()}"
        checks = ""
        if self.checks:
            checks = "SOP: " + " ".join(f"{k} {'OK' if v else 'X'}" for k, v in self.checks.items()) + "\n"
        health = self.health_line + "\n" if self.health_line else ""
        return (f"{'LONG' if self.side == 1 else 'SHORT'} {self.symbol} ({self.interval})\n"
                f"Ref price {self.close:.5g}  TP {self.tp:.5g}  SL {self.sl:.5g}  "
                f"max {self.horizon} bars\n{self.probs()}\n{checks}{health}"
                f"Research signal, not financial advice. Risk max {self.risk:.0%} per trade.")


def compute_signal(cfg: Config, refresh: bool = True, sop: "sop_mod.SOP | None" = None,
                   health_gate: bool = False) -> Signal:
    """Model signal -> SOP checklist -> strategy health gate. Any failure turns it into NO TRADE."""
    ds = build(cfg, refresh=refresh)
    model = fit_final(ds.X, ds.y, ds.barriers.valid, cfg)
    p = predict_last(model, ds.X)
    raw_side, side, blocked, checks, health_line = pick_side(p, cfg), 0, "", None, ""
    gate = None

    if raw_side == 0:
        blocked = "model not confident"
    else:
        side = raw_side
        if sop is not None:
            rules = sop_mod.evaluate(ds.df, ds.atr, cfg, sop)
            gate = sop_mod.gate(rules)
            checks = sop_mod.checklist(rules, raw_side)
            failed = [k for k, v in checks.items() if not v]
            if failed:
                side, blocked = 0, "SOP failed: " + ", ".join(failed)
        if side and health_gate:
            proba = walk_forward(ds.X, ds.y, ds.barriers.valid, cfg)
            h = assess(ds, proba, cfg, gate)
            health_line = h.line()
            if not h.ok:
                side, blocked = 0, "health gate: " + "; ".join(h.reasons)

    close = float(ds.df["close"].iloc[-1])
    dist = cfg.tp_mult * float(ds.atr.iloc[-1])
    return Signal(cfg.symbol, cfg.interval, ds.df.index[-1], side, close,
                  close + side * dist if side else None, close - side * dist if side else None,
                  tuple(float(x) for x in p), cfg.horizon, cfg.risk, checks, health_line, blocked)


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def should_send(sig: Signal, state: dict) -> bool:
    """Send only real signals, once per bar, and not while a same-side trade is still 'open'."""
    if sig.side == 0:
        return False
    last = state.get(f"{sig.symbol}_{sig.interval}")
    if not last:
        return True
    if last["bar"] == sig.bar_time.isoformat():
        return False
    bar_ms = INTERVAL_MS[sig.interval]
    bars_since = (sig.bar_time.value // 1_000_000 - last["bar_ms"]) // bar_ms
    return not (last["side"] == sig.side and bars_since < sig.horizon)


def record(sig: Signal, state: dict, path: Path) -> None:
    state[f"{sig.symbol}_{sig.interval}"] = {
        "bar": sig.bar_time.isoformat(), "bar_ms": sig.bar_time.value // 1_000_000, "side": sig.side}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state))


def check_once(cfgs: list[Config], send, state_path: Path = STATE_FILE, log=print,
               persist: bool = True, sop=None, health_gate: bool = False) -> None:
    for cfg in cfgs:
        try:
            sig = compute_signal(cfg, sop=sop, health_gate=health_gate)
            state = _load_state(state_path)
            if should_send(sig, state):
                send(sig.message())
                if persist:
                    record(sig, state, state_path)
                log(f"[sent] {sig.symbol} {'LONG' if sig.side == 1 else 'SHORT'} @ {sig.close:.5g}")
            else:
                log(f"[skip] {sig.head()} | {sig.blocked or 'already sent'}")
        except Exception as e:  # keep the watcher alive on network/data errors
            log(f"[error] {cfg.symbol}: {e}")


def next_run(now: float, interval: str, delay_s: int = 60) -> float:
    sec = INTERVAL_MS[interval] / 1000
    return (now // sec + 1) * sec + delay_s


def watch(cfgs: list[Config], send, delay_s: int | None = None, log=print, persist: bool = True,
          sop=None, health_gate: bool = False) -> None:
    interval = cfgs[0].interval
    if delay_s is None:  # Binance candles are final right away; Yahoo forex lags ~1 min
        delay_s = 5 if all(is_crypto(c.symbol) for c in cfgs) else 60
    log(f"watching {', '.join(c.symbol for c in cfgs)} on {interval}; Ctrl+C to stop")
    while True:
        check_once(cfgs, send, log=log, persist=persist, sop=sop, health_gate=health_gate)
        wake = next_run(time.time(), interval, delay_s)
        log(f"next check at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(wake))} UTC")
        time.sleep(max(1, wake - time.time()))
