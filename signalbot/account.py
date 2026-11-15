"""Simulated trading account built on top of the paper-trade ledger.

Rules (all configurable):
  * start with `start` USD; every signal that is actually entered uses `margin_pct` of the CURRENT balance as margin;
  * the leverage is chosen per trade so that hitting the stop loses about `risk_pct` of the balance:
        leverage = risk_pct / (margin_pct * stop_distance), clamped to [1, max_lev];
  * profit and loss use the position NOTIONAL (margin x leverage); fees = fee_bps of the notional per round trip;
  * isolated margin: a trade can never lose more than its margin (liquidation); funding and slippage are ignored;
  * profits are compounded (the next trade is sized from the new balance); margin of open trades is locked.
The account is recomputed from the ledger every time, so it can never drift from the recorded trades.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import INTERVAL_MS
from .ledger import CLOSED


@dataclass(frozen=True)
class Account:
    start: float = 100.0
    margin_pct: float = 0.10
    risk_pct: float = 0.02
    max_lev: float = 100.0
    fee_bps: float = 10.0          # round trip on the notional (about 0.05% taker per side)


@dataclass
class Replay:
    trades: pd.DataFrame
    summary: dict


def leverage(stop_frac: float, acct: Account) -> float:
    if not np.isfinite(stop_frac) or stop_frac <= 0:
        return 1.0
    return float(min(acct.max_lev, max(1.0, acct.risk_pct / (acct.margin_pct * stop_frac))))


def replay(led: pd.DataFrame, acct: Account = Account()) -> Replay:
    rows = led[led["status"].isin(CLOSED + ["open"])]
    events = []
    for _, r in rows.iterrows():
        bar = pd.Timedelta(milliseconds=INTERVAL_MS[r["interval"]])
        entry_t = pd.Timestamp(r["signal_time"]) + bar                       # entry fills at the next bar's open
        events.append((entry_t, 1, r))
        if r["status"] in CLOSED:
            events.append((pd.Timestamp(r["exit_time"]) + bar, 0, r))        # realised when the exit bar closes
    events.sort(key=lambda e: (e[0], e[1]))                                  # exits before entries at the same time

    bal, open_pos, skipped, out = acct.start, {}, set(), []
    for t, kind, r in events:
        rid = r["id"]
        if kind == 1:
            locked = sum(p["margin"] for p in open_pos.values())
            margin = min(acct.margin_pct * bal, bal - locked)
            if bal <= 0 or margin < 0.01:
                skipped.add(rid)
                continue
            stop_sig = abs(float(r["entry_ref"]) - float(r["sl"])) / float(r["entry_ref"])
            lev = leverage(stop_sig, acct)
            open_pos[rid] = dict(margin=margin, lev=lev, notional=margin * lev, t=t)
        elif rid not in skipped and rid in open_pos:
            p = open_pos.pop(rid)
            move = float(r["gross_r"]) * float(r["stop_frac"])               # price move fraction in the trade's favour
            pnl = p["notional"] * move
            liquidated = pnl <= -p["margin"]
            pnl = -p["margin"] if liquidated else max(pnl - p["notional"] * acct.fee_bps / 1e4, -p["margin"])
            bal += pnl
            out.append(dict(id=rid, mode=r["mode"], symbol=r["symbol"], side=int(r["side"]), status=r["status"],
                            entry_time=p["t"].isoformat(), exit_time=t.isoformat(), margin=p["margin"], leverage=p["lev"],
                            notional=p["notional"], pnl=pnl, balance=bal, liquidated=liquidated))
    trades = pd.DataFrame(out, columns=["id", "mode", "symbol", "side", "status", "entry_time", "exit_time", "margin",
                                        "leverage", "notional", "pnl", "balance", "liquidated"])
    bals = np.r_[acct.start, trades["balance"].to_numpy(float)] if len(trades) else np.array([acct.start])
    dd = float((bals / np.maximum.accumulate(bals) - 1).min())
    summary = dict(start=acct.start, balance=float(bal), pnl=float(bal - acct.start), ret=float(bal / acct.start - 1),
                   n=len(trades), wins=int((trades["pnl"] > 0).sum()) if len(trades) else 0,
                   losses=int((trades["pnl"] <= 0).sum()) if len(trades) else 0, drawdown=dd,
                   open=len(open_pos), locked=float(sum(p["margin"] for p in open_pos.values())),
                   skipped=len(skipped), liquidations=int(trades["liquidated"].sum()) if len(trades) else 0,
                   best=float(trades["pnl"].max()) if len(trades) else 0.0,
                   worst=float(trades["pnl"].min()) if len(trades) else 0.0)
    return Replay(trades, summary)


def account_text(led: pd.DataFrame, acct: Account = Account(), last: int = 8) -> str:
    rp = replay(led, acct)
    s = rp.summary
    lines = [f"AKAUN SIMULASI: modal ${s['start']:.0f} | margin {acct.margin_pct:.0%} daripada baki | leverage auto "
             f"(rugi ~{acct.risk_pct:.0%} jika SL, maks {acct.max_lev:g}x) | yuran {acct.fee_bps:g} bps pada nilai posisi",
             f"  Baki ${s['balance']:.2f} | Untung/Rugi {s['pnl']:+.2f} USD ({s['ret']:+.1%}) | {s['n']} trade "
             f"(menang {s['wins']}, kalah {s['losses']}) | drawdown {s['drawdown']:.1%} | terbuka {s['open']} "
             f"(margin ${s['locked']:.2f})" + (f" | dilewati {s['skipped']}" if s["skipped"] else "")
             + (f" | LIKUIDASI {s['liquidations']}" if s["liquidations"] else "")]
    tail = rp.trades.tail(last)
    if len(tail):
        lines.append("  Terakhir:")
        for _, t in tail.iterrows():
            lines.append(f"   {t['exit_time'][:16]} {t['symbol']} {'LONG' if t['side'] == 1 else 'SHORT'} {t['status']:4s} "
                         f"margin ${t['margin']:.2f} x{t['leverage']:.0f} -> {t['pnl']:+.2f} USD | baki ${t['balance']:.2f}")
    return "\n".join(lines)


def save_csv(led: pd.DataFrame, path, acct: Account = Account()) -> None:
    rp = replay(led, acct)
    path.parent.mkdir(parents=True, exist_ok=True)
    rp.trades.to_csv(path, index=False)


def daily_text(led: pd.DataFrame, acct: Account = Account(), now: pd.Timestamp | None = None) -> str:
    """Compact ASCII summary for Telegram: balance, last 24 hours, totals. Doubles as a 'bot is alive' message."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    head = f"RINGKASAN HARIAN (paper) {now:%d %b %H:%M} UTC"
    if led.empty:
        return head + "\nBot hidup. Belum ada signal direkod."
    rp = replay(led, acct)
    s, tr = rp.summary, rp.trades
    lines = [head, f"Akaun ${s['start']:.0f} -> ${s['balance']:.2f} ({s['ret']:+.1%}) | drawdown {s['drawdown']:.0%}"]
    if len(tr):
        exits = pd.to_datetime(tr["exit_time"], utc=True)
        recent = tr[(exits > now - pd.Timedelta(hours=24)) & (exits <= now)]
        w = int((recent["pnl"] > 0).sum())
        lines.append(f"24 jam lepas: {len(recent)} trade tamat ({w} menang, {len(recent) - w} kalah) -> "
                     f"{recent['pnl'].sum():+.2f} USD")
        lines.append(f"Jumlah: {s['n']} trade, menang {s['wins'] / s['n']:.0%} | terbaik {s['best']:+.2f}, "
                     f"terburuk {s['worst']:+.2f} USD")
    else:
        lines.append("Belum ada trade yang tamat.")
    lines.append(f"Terbuka: {s['open']} (margin ${s['locked']:.2f}) | terlepas: {int((led['status'] == 'missed').sum())}")
    return "\n".join(lines)
