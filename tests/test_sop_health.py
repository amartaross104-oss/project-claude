import numpy as np
import pandas as pd

from signalbot import signals
from signalbot.backtest import run_backtest
from signalbot.config import Config
from signalbot.features import atr
from signalbot.health import Criteria, assess
from signalbot.labels import barrier_outcomes, to_class
from signalbot.pipeline import Dataset
from signalbot.sop import SOP, checklist, evaluate, gate
from tests.test_signalbot import synth


def make_ds(cfg, n=2500, seed=1):
    df = synth(n, seed)
    a = atr(df)
    b = barrier_outcomes(df, a, cfg.tp_mult, cfg.horizon)
    return Dataset(df, pd.DataFrame(index=df.index), a, b, to_class(b.outcome))


def test_sop_rules_are_causal():
    cfg = Config(symbol="EURUSD", interval="1h", cost_bps=2.0)
    df = synth(900)
    a = atr(df)
    full = evaluate(df, a, cfg, SOP())
    k = 700
    part = evaluate(df.iloc[: k + 1], a.iloc[: k + 1], cfg, SOP())
    for name in full:
        for side in (0, 1):
            assert full[name][side][k] == part[name][side][k], name


def test_trend_rule_directional_and_session_only_for_forex():
    cfg_fx = Config(symbol="EURUSD", interval="4h", cost_bps=2.0)
    cfg_cr = Config(symbol="BTCUSDT", interval="4h", cost_bps=20.0)
    df = synth(900)
    a = atr(df)
    assert "session" in evaluate(df, a, cfg_fx, SOP())
    assert "session" not in evaluate(df, a, cfg_cr, SOP())
    rules = evaluate(df, a, cfg_cr, SOP())
    both = rules["trend"][0] & rules["trend"][1]
    assert not both.any()          # never "uptrend" and "downtrend" at once
    assert set(checklist(rules, 1)) == set(rules)


def test_gate_blocks_trades():
    cfg = Config(cost_bps=0.0)
    ds = make_ds(cfg)
    proba = np.tile([0.1, 0.05, 0.85], (len(ds.df), 1))
    none = (np.zeros(len(ds.df), bool), np.zeros(len(ds.df), bool))
    assert run_backtest(ds.df, ds.barriers, ds.atr, proba, cfg).stats["trades"] > 0
    assert run_backtest(ds.df, ds.barriers, ds.atr, proba, cfg, gate=none).stats["trades"] == 0


def test_health_fails_for_random_and_passes_for_oracle():
    cfg = Config(cost_bps=0.0, interval="1h")
    ds = make_ds(cfg)
    rng = np.random.default_rng(0)
    rand = rng.dirichlet([1, 0.1, 1], size=len(ds.df))
    allow = (np.ones(len(ds.df), bool),) * 2
    assert not assess(ds, rand, cfg, allow, Criteria(min_trades=20)).ok

    oracle = np.full((len(ds.df), 3), 0.05)           # perfect foresight: only for testing the gate
    oracle[np.arange(len(ds.df)), ds.y] = 0.9
    h = assess(ds, oracle, cfg, allow, Criteria(min_trades=20))
    assert h.ok and h.stats["profit_factor"] > 1.2 and "Backtest OOS" in h.line()


def test_blocked_signal_message_and_no_send(monkeypatch, tmp_path):
    blocked = signals.Signal("BTCUSDT", "4h", pd.Timestamp("2026-01-01", tz="UTC"), 0, 100.0,
                             None, None, (0.2, 0.05, 0.75), 12, 0.01, blocked="health gate: x")
    assert "NO TRADE" in blocked.message() and "health gate" in blocked.message()
    monkeypatch.setattr(signals, "compute_signal", lambda *a, **k: blocked)
    sent = []
    check_args = dict(state_path=tmp_path / "s.json", log=lambda *_: None)
    signals.check_once([Config(interval="4h")], sent.append, **check_args)
    assert sent == []


def test_sent_signal_shows_sop_checklist_and_health():
    s = signals.Signal("BTCUSDT", "4h", pd.Timestamp("2026-01-01", tz="UTC"), 1, 100.0, 101.0, 99.0,
                       (0.2, 0.05, 0.75), 12, 0.01, {"trend": True, "rsi": True},
                       "Backtest OOS: 80 trades, PF 1.40, +0.15 R/trade, win 52%")
    m = s.message()
    assert "SOP: trend OK rsi OK" in m and "PF 1.40" in m
