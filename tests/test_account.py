import numpy as np
import pandas as pd

from signalbot import ledger
from signalbot.account import Account, account_text, leverage, replay

IDX = pd.date_range("2026-01-01", periods=60, freq="15min", tz="UTC")


def led_from(trades):
    """trades: list of dicts with keys i (signal bar), exit_i (exit bar), status, gross_r, stop_frac, [entry_ref, sl]."""
    rows = []
    for k, t in enumerate(trades):
        row = {c: np.nan for c in ledger.COLUMNS}
        entry_ref = t.get("entry_ref", 100.0)
        row.update(id=f"t{k}", mode="scalp", symbol="BTCUSDT", interval="15m", side=1,
                   signal_time=IDX[t["i"]].isoformat(), entry_ref=entry_ref,
                   sl=t.get("sl", entry_ref * (1 - t.get("stop_frac", 0.004))), tp=entry_ref * 1.01, horizon=16,
                   status=t.get("status", "tp"), gross_r=t.get("gross_r", 1.0), stop_frac=t.get("stop_frac", 0.004),
                   exit_time=IDX[t["exit_i"]].isoformat() if "exit_i" in t else np.nan, fill=entry_ref, notified=False)
        rows.append(row)
    return pd.DataFrame(rows, columns=ledger.COLUMNS)


def test_leverage_rule():
    a = Account()
    assert np.isclose(leverage(0.004, a), 50.0)            # 2% risk / (10% margin * 0.4% stop)
    assert np.isclose(leverage(0.055, a), 0.02 / (0.1 * 0.055))
    assert leverage(0.0001, a) == 100.0                     # capped
    assert leverage(0.5, a) == 1.0                          # floored


def test_single_win_exact_pnl_with_fee_on_notional():
    rp = replay(led_from([dict(i=0, exit_i=2, status="tp", gross_r=1.0)]))
    t = rp.trades.iloc[0]
    # margin 10, 50x -> notional 500; +0.4% move = +2.00; fee 10 bps of 500 = 0.50
    assert np.isclose(t["margin"], 10.0) and np.isclose(t["leverage"], 50.0) and np.isclose(t["notional"], 500.0)
    assert np.isclose(t["pnl"], 1.5) and np.isclose(rp.summary["balance"], 101.5)


def test_profit_is_compounded_into_the_next_trade():
    rp = replay(led_from([dict(i=0, exit_i=2, status="tp", gross_r=1.0),
                          dict(i=6, exit_i=8, status="sl", gross_r=-1.0)]))
    t2 = rp.trades.iloc[1]
    assert np.isclose(t2["margin"], 10.15)                          # 10% of 101.5, not of 100
    assert np.isclose(t2["pnl"], -(10.15 * 50 * 0.004) - 10.15 * 50 * 0.001)   # -2.03 - 0.5075
    assert np.isclose(rp.summary["balance"], 101.5 - 2.5375)


def test_liquidation_caps_loss_at_margin_and_missed_trades_are_ignored():
    rp = replay(led_from([dict(i=0, exit_i=2, status="sl", gross_r=-30.0),     # huge gap far beyond the stop
                          dict(i=10, status="missed", gross_r=0.0)]))
    assert len(rp.trades) == 1 and rp.trades.iloc[0]["liquidated"]
    assert np.isclose(rp.trades.iloc[0]["pnl"], -10.0) and np.isclose(rp.summary["balance"], 90.0)
    assert rp.summary["liquidations"] == 1


def test_open_trades_lock_margin_and_signals_without_balance_are_skipped():
    open_trades = [dict(i=k, status="open", gross_r=np.nan, stop_frac=np.nan) for k in range(12)]
    for t in open_trades:
        t["exit_i"] = None
    led = led_from([{k: v for k, v in t.items() if k != "exit_i"} for t in open_trades])
    rp = replay(led)
    assert rp.summary["open"] == 10 and np.isclose(rp.summary["locked"], 100.0)    # 10 x $10 uses the whole balance
    assert rp.summary["skipped"] == 2


def test_loss_streak_never_goes_below_zero_and_text_is_ascii():
    trades = [dict(i=3 * k, exit_i=3 * k + 1, status="sl", gross_r=-1.0) for k in range(20)]
    rp = replay(led_from(trades))
    assert rp.summary["balance"] > 0 and rp.summary["pnl"] < 0 and rp.summary["wins"] == 0
    assert rp.trades["balance"].is_monotonic_decreasing
    text = account_text(led_from(trades))
    text.encode("ascii")
    assert "AKAUN SIMULASI" in text and "Baki $" in text


def test_close_message_includes_usd_result_and_report_has_account_block(tmp_path):
    led = led_from([dict(i=0, exit_i=2, status="tp", gross_r=1.0)])
    led["bars"] = 2
    led["exit_price"] = 100.4
    row = led.iloc[0]
    acct = replay(led).trades.iloc[0]
    msg = ledger.close_message(row, 10.0, acct)
    assert "UNTUNG +1.50 USD" in msg and "Baki $101.50" in msg and "x50" in msg
    p = tmp_path / "p.csv"
    ledger.save(led, p)
    assert "AKAUN SIMULASI" in ledger.report_text(p)


def test_daily_text_window_totals_and_empty_case():
    from signalbot.account import daily_text
    now = IDX[40]
    led = led_from([dict(i=0, exit_i=2, status="tp", gross_r=1.0),          # long before the 24h window? same day -> inside
                    dict(i=6, exit_i=8, status="sl", gross_r=-1.0)])
    txt = daily_text(led, Account(), now=now)
    txt.encode("ascii")
    assert txt.startswith("RINGKASAN HARIAN (paper)")
    assert "24 jam lepas: 2 trade tamat (1 menang, 1 kalah)" in txt and "Akaun $100 -> $98.96" in txt
    assert "Terbuka: 0" in txt
    old = daily_text(led, Account(), now=IDX[40] + pd.Timedelta(days=3))
    assert "24 jam lepas: 0 trade tamat" in old                              # nothing finished in the last 24 hours
    empty = daily_text(ledger.load("/nonexistent/none.csv"), Account(), now=now)
    assert "Bot hidup" in empty
