import numpy as np
import pandas as pd

from signalbot.trend import (TrendParams, fresh_signal, position_open, run_trend, signal_at)
from signalbot.trend_cli import build_message, gate_direction

P = TrendParams(entry_n=20, exit_n=10, atr_n=14, stop_mult=2.0)


def bars(closes, opens=None, pad=0.5):
    closes = np.asarray(closes, float)
    opens = np.r_[closes[0], closes[:-1]] if opens is None else np.asarray(opens, float)
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + pad,
                         "low": np.minimum(opens, closes) - pad, "close": closes, "volume": 1.0},
                        index=pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC"))


def ramp_then_crash():
    c = [100.0] * 60 + [100 + 1.5 * (i + 1) for i in range(20)] + [130 - 4 * (i + 1) for i in range(20)]
    return bars(c)


def test_levels_exclude_current_bar_no_lookahead():
    df = ramp_then_crash()
    full = signal_at(df, P, 60)
    part = signal_at(df.iloc[:61], P, -1)
    assert full == part == 1                      # first close above the previous 20-day high
    assert signal_at(df, P, 59) == 0              # flat before: no breakout


def test_long_trade_enters_next_open_and_exits_on_channel():
    df = ramp_then_crash()
    t = run_trend(df, P, cost_bps=0.0)
    first = t.iloc[0]
    assert first["side"] == 1 and first["time"] == df.index[60]
    assert first["entry"] == df["open"].iloc[61]
    assert first["r"] > 0 and first["exit_time"] > df.index[79] and not first["open"]


def test_gap_through_stop_fills_at_open_not_at_stop():
    c = [100.0] * 60 + [101.5, 103.0, 104.5]
    df = bars(c)
    o = df["open"].copy()
    o.iloc[62] = 80.0                              # gap far below the stop
    df = bars(c + [80.0], opens=list(o) + [80.0])
    df.loc[df.index[62], ["high", "low", "close"]] = [81.0, 79.0, 80.0]
    t = run_trend(df, P, cost_bps=0.0)
    assert t.iloc[0]["exit"] == 80.0 and t.iloc[0]["r"] < -1.0


def test_cost_reduces_r_and_position_state():
    df = ramp_then_crash()
    r0 = run_trend(df, P, 0.0).iloc[0]["r"]
    r1 = run_trend(df, P, 50.0).iloc[0]["r"]
    assert r1 < r0
    up = bars([100.0] * 60 + [100 + 1.5 * (i + 1) for i in range(15)])   # still trending, no exit yet
    assert position_open(run_trend(up, P, 0.0))
    assert fresh_signal(up, P, 0.0) == 0           # breakout continues but a trade is already open
    just = bars([100.0] * 60 + [101.5])
    assert fresh_signal(just, P, 0.0) == 1         # first breakout bar with no open trade


def test_gate_direction_blocks_small_or_insignificant_and_passes_strong():
    rng = np.random.default_rng(0)
    n = 60
    t = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=n, freq="30D", tz="UTC"),
                      "side": 1, "r": rng.normal(0.8, 1.0, n)})
    ok, why, st = gate_direction(t, 1)
    assert ok and st["n"] == n
    ok, why, _ = gate_direction(t.iloc[:10], 1)
    assert not ok and any("only 10 trades" in w for w in why)
    noisy = t.assign(r=rng.normal(0.0, 1.0, n))
    assert not gate_direction(noisy, 1)[0]
    assert gate_direction(t, -1)[0] is False       # no shorts at all


def test_message_labels_paper_mode():
    df = ramp_then_crash().iloc[:61]
    st = dict(n=20, pf=1.5, exp=0.3, t=1.0, recent_exp=0.1)
    m = build_message("BTCUSDT", P, df, 1, st, ok=False, paper=True)
    assert m.startswith("PAPER / UNVALIDATED") and "LONG BTCUSDT (1d) TREND 20/10" in m
    assert "\\n" not in m and "gate FAILED" in m
    ok_msg = build_message("BTCUSDT", P, df, 1, st, ok=True, paper=False)
    assert not ok_msg.startswith("PAPER") and "gate PASSED" in ok_msg


def test_trend_notify_dedupes_same_bar(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from signalbot import trend_cli

    df = ramp_then_crash().iloc[:61]                      # last bar = first breakout
    monkeypatch.setattr(trend_cli.data, "load", lambda *a, **k: df)
    sent = []
    a = SimpleNamespace(symbols=["BTCUSDT"], system="1", paper=True, notify=True, dry_run=False, loop=False)
    monkeypatch.setattr(trend_cli, "fresh_signal", lambda *x, **k: 1)
    state = tmp_path / "s.json"
    assert trend_cli.check_trend(a, state, sent.append) is True
    assert trend_cli.check_trend(a, state, sent.append) is False   # same bar -> no duplicate
    assert len(sent) == 1 and sent[0].startswith("PAPER / UNVALIDATED")
    a.dry_run = True
    assert trend_cli.check_trend(a, state, sent.append) is False and len(sent) == 1


def test_empty_history_does_not_crash():
    df = ramp_then_crash().iloc[:61]
    t = run_trend(df.iloc[:30], P, 0.0)                  # too short for any trade
    assert t.empty and "side" in t.columns
    ok, why, st = gate_direction(t, 1)
    assert not ok and st == {}
    m = build_message("BTCUSDT", P, df, 1, st, ok=False, paper=True)
    assert "no completed long trades" in m and m.startswith("PAPER / UNVALIDATED")
