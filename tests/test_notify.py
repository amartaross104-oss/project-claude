import pandas as pd
import pytest

from signalbot import notify, signals
from signalbot.config import Config
from signalbot.signals import Signal, check_once, next_run, should_send


def sig(side=1, bar="2026-01-01 00:00", interval="4h", symbol="BTCUSDT"):
    return Signal(symbol, interval, pd.Timestamp(bar, tz="UTC"), side, 100.0,
                  101.0 if side else None, 99.0 if side else None, (0.2, 0.05, 0.75), 12, 0.01)


def test_message_contains_levels_and_disclaimer():
    m = sig(1).message()
    assert "LONG BTCUSDT" in m and "TP 101" in m and "SL 99" in m and "not financial advice" in m
    assert "SHORT" in sig(-1).message()


def test_dedupe_rules(tmp_path):
    path = tmp_path / "s.json"
    s1 = sig(1, "2026-01-01 00:00")
    assert should_send(s1, {})
    signals.record(s1, {}, path)
    state = signals._load_state(path)
    assert not should_send(sig(1, "2026-01-01 00:00"), state)   # same bar
    assert not should_send(sig(1, "2026-01-01 04:00"), state)   # same side, trade still open
    assert should_send(sig(-1, "2026-01-01 04:00"), state)      # opposite side
    assert should_send(sig(1, "2026-01-03 00:00"), state)       # 12 bars (2 days) later
    assert not should_send(sig(0, "2026-01-01 04:00"), state)   # no signal


def test_check_once_sends_once_and_dry_run_does_not_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(signals, "compute_signal", lambda cfg: sig(1))
    sent, path = [], tmp_path / "s.json"
    check_once([Config(interval="4h")], sent.append, path, log=lambda *_: None, persist=False)
    check_once([Config(interval="4h")], sent.append, path, log=lambda *_: None, persist=False)
    assert len(sent) == 2 and not path.exists()                 # dry-run: nothing remembered
    sent.clear()
    for _ in range(2):
        check_once([Config(interval="4h")], sent.append, path, log=lambda *_: None)
    assert len(sent) == 1


def test_check_once_survives_errors(monkeypatch, tmp_path):
    def boom(cfg):
        raise RuntimeError("network down")
    monkeypatch.setattr(signals, "compute_signal", boom)
    logs = []
    check_once([Config(interval="4h")], lambda m: None, tmp_path / "s.json", log=logs.append)
    assert "network down" in logs[0]


def test_next_run_is_after_bar_close():
    t = pd.Timestamp("2026-01-01 05:10", tz="UTC").timestamp()
    assert next_run(t, "4h", 60) == pd.Timestamp("2026-01-01 08:01", tz="UTC").timestamp()


def test_telegram_requires_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(RuntimeError):
        notify.send_telegram("x")


def test_telegram_send_payload(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    calls = []

    class R:
        def raise_for_status(self): pass

    monkeypatch.setattr(notify.requests, "post", lambda url, json, timeout: calls.append((url, json)) or R())
    notify.send_telegram("hello")
    assert calls == [("https://api.telegram.org/botTOKEN/sendMessage", {"chat_id": "123", "text": "hello"})]
