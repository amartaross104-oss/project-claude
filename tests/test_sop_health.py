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


def test_demo_message_is_clearly_labelled_test():
    from signalbot.cli import demo_text
    cfg = Config(symbol="BTCUSDT", interval="4h", cost_bps=20.0)
    df = synth(300)
    text = demo_text(cfg, df, atr(df), 1)
    assert text.startswith("TEST ONLY - NOT A REAL SIGNAL")
    assert "LONG BTCUSDT" in text and "TP " in text and "SL " in text
    assert "n/a (test message)" in text and "No model" in text
    assert "SHORT" in demo_text(cfg, df, atr(df), -1)


def test_incremental_update_only_fetches_new_bars(tmp_path, monkeypatch):
    from signalbot import data
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    idx = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    old = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
    old.index.name = "time"
    old.to_csv(data.cache_path("BTCUSDT", "4h"))
    seen = {}

    def fake(symbol, interval, years=3.0, start_ms=None):
        seen["start"] = start_ms
        j = pd.date_range(idx[-1], periods=3, freq="4h", tz="UTC")  # last bar re-sent + 2 new
        d = pd.DataFrame({"open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 9.0}, index=j)
        d.index.name = "time"
        return d

    monkeypatch.setattr(data, "fetch_binance", fake)
    df = data.update("BTCUSDT", "4h")
    assert seen["start"] == int(idx[-1].timestamp() * 1000)   # asked only from the last cached bar
    assert len(df) == 12 and df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert df["close"].iloc[-1] == 9.0 and df["close"].iloc[0] == 1.5  # old kept, new overrides tail


def test_funding_cache_roundtrip_with_mixed_timestamps(tmp_path, monkeypatch):
    from signalbot import derivs
    monkeypatch.setattr(derivs, "DATA_DIR", tmp_path)
    path = derivs._month_file("BTCUSDT", "2024-01")
    path.parent.mkdir(parents=True)
    path.write_text("time,rate\n2024-01-02 00:00:00.002000+00:00,0.0001\n"
                    "2024-01-02 08:00:00+00:00,0.0002\n2024-01-02 16:00:00.005000+00:00,0.0003\n")
    monkeypatch.setattr(derivs, "_download_month", lambda *a: None)
    df = derivs.load_funding("BTCUSDT", pd.Timestamp("2024-01-01", tz="UTC"))
    assert len(df) == 3 and df["time"].is_monotonic_increasing


def test_funding_features_use_only_past_funding(monkeypatch):
    from signalbot import derivs
    idx = pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC")
    times = pd.to_datetime(["2024-01-02 00:00:00.003", "2024-01-02 08:00", "2024-01-02 16:00",
                            "2024-01-03 00:00:00.004", "2024-01-03 08:00", "2024-01-03 16:00"],
                           utc=True, format="ISO8601")
    fr = pd.DataFrame({"time": times, "rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    monkeypatch.setattr(derivs, "load_funding", lambda *a, **k: fr)
    f = derivs.funding_features("BTCUSDT", idx, "1d")
    # day 2024-01-02 closes 2024-01-03 00:00:00 -> last allowed funding is the 16:00 one (0.3);
    # the 00:00:00.004 settlement (0.4) is after the close and must NOT leak in.
    assert np.isclose(f["fund_last"].iloc[0], 0.3)
    assert np.isclose(f["fund_last"].iloc[1], 0.6)


def test_t_stat_and_significance_gate():
    from signalbot.health import criteria_for, t_stat
    assert np.isclose(t_stat([1.0, 1.0, 1.0, 2.0]), 1.25 / (0.5 / 2))
    assert np.isnan(t_stat([1.0]))
    assert abs(t_stat(np.random.default_rng(0).normal(0, 1, 5000))) < 3
    assert criteria_for("1d").recent_days == 365 and criteria_for("4h").recent_days == 180


def test_t_stat_zero_variance():
    from signalbot.health import t_stat
    assert t_stat([1.0, 1.0, 1.0]) == float("inf") and t_stat([-1.0, -1.0]) == float("-inf")
