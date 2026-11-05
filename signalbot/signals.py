"""Compute the latest signal and run the periodic watcher (with de-duplication)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .backtest import pick_side
from .config import Config
from .data import DATA_DIR, INTERVAL_MS
from .model import fit_final, predict_last
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

    def head(self) -> str:
        return (f"{self.symbol} {self.interval} | bar closed {self.bar_time:%Y-%m-%d %H:%M} UTC"
                f" | close {self.close:.5g}")

    def probs(self) -> str:
        return f"P(down)={self.proba[0]:.2f} P(none)={self.proba[1]:.2f} P(up)={self.proba[2]:.2f}"

    def message(self) -> str:
        if self.side == 0:
            return f"{self.head()}\nNO TRADE  {self.probs()}"
        return (f"{'LONG' if self.side == 1 else 'SHORT'} {self.symbol} ({self.interval})\n"
                f"Ref price {self.close:.5g}  TP {self.tp:.5g}  SL {self.sl:.5g}  "
                f"max {self.horizon} bars\n{self.probs()}\n"
                f"Research signal, not financial advice. Risk max {self.risk:.0%} per trade.")


def compute_signal(cfg: Config, refresh: bool = True) -> Signal:
    ds = build(cfg, refresh=refresh)
    model = fit_final(ds.X, ds.y, ds.barriers.valid, cfg)
    p = predict_last(model, ds.X)
    side = pick_side(p, cfg)
    close = float(ds.df["close"].iloc[-1])
    dist = cfg.tp_mult * float(ds.atr.iloc[-1])
    return Signal(cfg.symbol, cfg.interval, ds.df.index[-1], side, close,
                  close + side * dist if side else None, close - side * dist if side else None,
                  tuple(float(x) for x in p), cfg.horizon, cfg.risk)


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
               persist: bool = True) -> None:
    for cfg in cfgs:
        try:
            sig = compute_signal(cfg)
            state = _load_state(state_path)
            if should_send(sig, state):
                send(sig.message())
                if persist:
                    record(sig, state, state_path)
                log(f"[sent] {sig.symbol} {'LONG' if sig.side == 1 else 'SHORT'} @ {sig.close:.5g}")
            else:
                log(f"[skip] {sig.head()} side={sig.side}")
        except Exception as e:  # keep the watcher alive on network/data errors
            log(f"[error] {cfg.symbol}: {e}")


def next_run(now: float, interval: str, delay_s: int = 60) -> float:
    sec = INTERVAL_MS[interval] / 1000
    return (now // sec + 1) * sec + delay_s


def watch(cfgs: list[Config], send, delay_s: int = 60, log=print, persist: bool = True) -> None:
    interval = cfgs[0].interval
    log(f"watching {', '.join(c.symbol for c in cfgs)} on {interval}; Ctrl+C to stop")
    while True:
        check_once(cfgs, send, log=log, persist=persist)
        wake = next_run(time.time(), interval, delay_s)
        log(f"next check at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(wake))} UTC")
        time.sleep(max(1, wake - time.time()))
