import numpy as np
import pandas as pd

from signalbot import ledger, signals
from signalbot.config import Config
from signalbot.ledger import simulate_levels
from signalbot.signals import Signal, check_once

IDX = pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC")


def bars(rows):
    """rows = [(open, high, low, close)] starting at IDX[0]."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=IDX[: len(rows)])
    df["volume"] = 1.0
    return df


FLAT = (100.0, 100.2, 99.8, 100.0)
SIG = IDX[0]


def run(rows, side=1, tp=101.5, sl=98.5, horizon=8, zone=(99.75, 100.25)):
    return simulate_levels(bars(rows), SIG, side, tp, sl, horizon, *zone)


def test_long_tp_hit_with_net_r():
    r = run([FLAT, FLAT, (100.0, 101.6, 99.9, 101.0)])
    assert r["status"] == "tp" and r["exit_price"] == 101.5 and r["bars"] == 2
    assert np.isclose(r["gross_r"], 1.0) and np.isclose(r["stop_frac"], 1.5 / 100)
    # 20 bps fee on a 1.5% stop = 0.1333 R
    assert np.isclose(ledger.net_r(pd.Series(r), 20.0), 1.0 - 0.002 / 0.015)


def test_long_sl_hit_and_short_mirror():
    assert run([FLAT, FLAT, (100.0, 100.1, 98.0, 98.2)])["status"] == "sl"
    s = run([FLAT, FLAT, (100.0, 101.0, 98.4, 98.6)], side=-1, tp=98.5, sl=101.5)
    assert s["status"] == "tp" and np.isclose(s["gross_r"], 1.0)
    assert run([FLAT, FLAT, (100.0, 101.7, 99.9, 101.6)], side=-1, tp=98.5, sl=101.5)["status"] == "sl"


def test_both_barriers_in_one_bar_counts_as_sl():
    r = run([FLAT, FLAT, (100.0, 102.0, 98.0, 100.0)])
    assert r["status"] == "sl" and np.isclose(r["gross_r"], -1.0)


def test_gap_through_sl_fills_at_open():
    r = run([FLAT, FLAT, FLAT, (97.0, 97.5, 96.5, 97.0)])
    assert r["status"] == "sl" and r["exit_price"] == 97.0 and r["gross_r"] < -1.0


def test_entry_outside_zone_is_missed_and_pending_cases():
    miss = run([FLAT, (101.0, 101.2, 100.9, 101.0)])               # next open 101 > zone high
    assert miss["status"] == "missed" and miss["gross_r"] == 0.0
    assert run([FLAT]) is None                                      # entry bar not available yet
    assert run([FLAT, FLAT, FLAT]) is None                          # horizon not finished, no barrier hit
    t = run([FLAT] + [FLAT] * 8, horizon=8)
    assert t["status"] == "time" and t["bars"] == 8 and np.isclose(t["gross_r"], 0.0)


def sig(side=1):
    return Signal("BTCUSDT", "15m", IDX[0], side, 100.0, 101.5 if side == 1 else 98.5, 98.5 if side == 1 else 101.5,
                  (0.2, 0.05, 0.75), 8, 0.01, atr=1.0)


def test_check_once_records_signal_then_closes_it_and_alerts(tmp_path, monkeypatch):
    lp, sp = tmp_path / "paper.csv", tmp_path / "state.json"
    monkeypatch.setattr(signals, "compute_signal", lambda cfg, **kw: sig(1))
    monkeypatch.setattr(signals.data_mod, "load", lambda *a, **k: bars([FLAT, FLAT, (100.0, 101.6, 99.9, 101.0)]))
    sent = []
    check_once([Config(interval="15m")], sent.append, sp, log=lambda *_: None, track=True, mode="scalp",
               ledger_path=lp)
    led = ledger.load(lp)
    assert len(led) == 1 and led.loc[0, "mode"] == "scalp" and led.loc[0, "zone_lo"] == 99.75
    # next run: price data now contains the entry bar and the TP bar -> trade resolves, close alert is sent
    monkeypatch.setattr(signals, "compute_signal", lambda cfg, **kw: sig(0))
    check_once([Config(interval="15m")], sent.append, sp, log=lambda *_: None, track=True, mode="scalp",
               ledger_path=lp)
    led = ledger.load(lp)
    assert led.loc[0, "status"] == "tp" and bool(led.loc[0, "notified"])
    assert any(m.startswith("TP KENA") for m in sent) and "+1.00 R" in sent[-1]
    # a third run must not alert again
    n = len(sent)
    check_once([Config(interval="15m")], sent.append, sp, log=lambda *_: None, track=True, mode="scalp",
               ledger_path=lp)
    assert len(sent) == n


def test_dry_run_never_touches_the_ledger_and_duplicates_are_ignored(tmp_path, monkeypatch):
    lp, sp = tmp_path / "paper.csv", tmp_path / "state.json"
    monkeypatch.setattr(signals, "compute_signal", lambda cfg, **kw: sig(1))
    monkeypatch.setattr(signals.data_mod, "load", lambda *a, **k: bars([FLAT, FLAT]))
    check_once([Config(interval="15m")], lambda m: None, sp, log=lambda *_: None, persist=False, track=True,
               mode="scalp", ledger_path=lp)
    assert not lp.exists()
    assert ledger.add_signal(lp, "scalp", "BTCUSDT", "15m", IDX[0], 1, 100.0, 101.5, 98.5, 8, 99.75, 100.25)
    assert not ledger.add_signal(lp, "scalp", "BTCUSDT", "15m", IDX[0], 1, 100.0, 101.5, 98.5, 8, 99.75, 100.25)


def test_summary_and_report_text(tmp_path):
    lp = tmp_path / "paper.csv"
    rows = [("tp", 1.0), ("sl", -1.0), ("tp", 1.0), ("time", 0.2), ("sl", -1.0)]
    for i, (st, g) in enumerate(rows):
        ledger.add_signal(lp, "scalp", "BTCUSDT", "15m", IDX[i], 1, 100.0, 101.5, 98.5, 8, 99.75, 100.25)
    led = ledger.load(lp)
    led["status"], led["gross_r"], led["stop_frac"], led["bars"], led["fill"], led["exit_price"] = (
        [r[0] for r in rows], [r[1] for r in rows], 0.015, 3, 100.0, 100.0)
    ledger.save(led, lp)
    s = ledger.summarize(ledger.load(lp), 0.0)
    assert s["n"] == 5 and s["tp"] == 2 and s["sl"] == 2 and np.isclose(s["exp"], 0.04) and np.isclose(s["total_r"], 0.2)
    text = ledger.report_text(lp)
    assert "yuran 8 bps" in text and "yuran 20 bps" in text and "TP 2, SL 2" in text
    assert ledger.report_text(tmp_path / "none.csv") == "Belum ada rekod paper trade."
    text.encode("ascii")


def test_trend_trade_resolution(tmp_path):
    from tests.test_trend import P, ramp_then_crash
    from signalbot.trend import TrendParams  # noqa: F401

    df = ramp_then_crash()
    lp = tmp_path / "paper.csv"
    ledger.add_signal(lp, "trend", "BTCUSDT", "1d", df.index[60], 1, float(df["close"].iloc[60]), float("nan"), 98.0, 0,
                      float("-inf"), float("inf"), params=f"{P.entry_n}/{P.exit_n}")
    assert ledger.update(lp, "BTCUSDT", "1d", df.iloc[:70], modes=["trend"]) == []      # still running
    done = ledger.update(lp, "BTCUSDT", "1d", df, modes=["trend"])                       # exit rule triggered later
    assert len(done) == 1 and done[0]["status"] == "exit" and float(done[0]["gross_r"]) > 0
    assert "KELUAR (peraturan trend)" in ledger.close_message(done[0], 20.0)


def test_ledger_matches_backtest_barrier_outcomes_on_every_bar():
    """For identical rules (entry = next open, TP/SL = entry +/- 1.5 ATR) the ledger must agree with the backtest."""
    from signalbot.features import atr
    from signalbot.labels import AMBIGUOUS, DOWN, TIMEOUT, UP, barrier_outcomes
    from tests.test_signalbot import synth

    df = synth(900, seed=11)
    a = atr(df)
    horizon = 12
    b = barrier_outcomes(df, a, 1.5, horizon)
    checked = 0
    for t in np.flatnonzero(b.valid)[::3]:
        entry, dist = df["open"].iloc[t + 1], 1.5 * a.iloc[t]
        res = simulate_levels(df, df.index[t], 1, entry + dist, entry - dist, horizon, entry - 1, entry + 1)
        expected = {UP: "tp", DOWN: "sl", AMBIGUOUS: "sl", TIMEOUT: "time"}[int(b.outcome[t])]
        assert res is not None and res["status"] == expected, (t, res, b.outcome[t])
        if expected == "tp":
            assert np.isclose(res["gross_r"], 1.0)
        if expected == "sl" and b.outcome[t] != AMBIGUOUS:
            assert res["gross_r"] <= -1.0 + 1e-9
        assert df.index.get_loc(res["exit_time"]) == (b.exit_idx[t] if expected != "time" else t + horizon)
        checked += 1
    assert checked > 150
