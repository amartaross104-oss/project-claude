"""Automatic paper-trade ledger: every signal that is sent is tracked until its TP / SL / expiry is hit.

Rules (same spirit as the backtest, deliberately conservative):
  * entry fill = OPEN of the bar after the signal bar; if that price is outside the entry zone the trade is
    "missed" (the SOP says do not chase);
  * TP / SL are checked bar by bar with the high/low; if both are touched inside one bar the SL counts first;
  * a gap through the SL fills at the open (worse than the SL); TP fills at the TP level;
  * scalp trades expire after `horizon` bars at the close; trend trades follow the Donchian exit rule.
Results are stored gross (in R); fees are applied when reporting so different fee levels can be compared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DATA_DIR

LEDGER_FILE = DATA_DIR / "paper_trades.csv"
COLUMNS = ["id", "mode", "symbol", "interval", "side", "signal_time", "entry_ref", "zone_lo", "zone_hi", "tp", "sl",
           "horizon", "params", "status", "fill", "exit_time", "exit_price", "gross_r", "stop_frac", "bars", "notified"]
TEXT_COLUMNS = ["id", "mode", "symbol", "interval", "signal_time", "params", "status", "exit_time"]
CLOSED = ["tp", "sl", "time", "exit"]
FINAL = CLOSED + ["missed"]


def load(path=LEDGER_FILE) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in df:
            df[col] = np.nan
    df = df[COLUMNS].copy()
    for col in TEXT_COLUMNS:                      # empty CSV columns are read as float; they must hold text
        df[col] = df[col].astype(object)
    df["notified"] = df["notified"].astype(object)
    return df


def save(df: pd.DataFrame, path=LEDGER_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_id(mode, symbol, interval, signal_time, side) -> str:
    return f"{mode}|{symbol}|{interval}|{pd.Timestamp(signal_time).isoformat()}|{int(side)}"


def add_signal(path, mode, symbol, interval, signal_time, side, entry_ref, tp, sl, horizon, zone_lo, zone_hi,
               params="") -> bool:
    """Record a sent signal as open. Returns False if it was already recorded."""
    led = load(path)
    sid = make_id(mode, symbol, interval, signal_time, side)
    if (led["id"] == sid).any():
        return False
    row = {c: np.nan for c in COLUMNS}
    row.update(id=sid, mode=mode, symbol=symbol, interval=interval, side=int(side),
               signal_time=pd.Timestamp(signal_time).isoformat(), entry_ref=entry_ref, zone_lo=zone_lo,
               zone_hi=zone_hi, tp=tp, sl=sl, horizon=horizon, params=params, status="open", notified=False)
    save(pd.concat([led, pd.DataFrame([row])], ignore_index=True), path)
    return True


def simulate_levels(df: pd.DataFrame, signal_time, side: int, tp: float, sl: float, horizon: int,
                    zone_lo: float, zone_hi: float) -> dict | None:
    """Resolve one TP/SL trade against price data. None = still pending."""
    signal_time = pd.Timestamp(signal_time)
    if signal_time not in df.index:
        return None
    i = df.index.get_loc(signal_time)
    if i + 1 >= len(df):
        return None                                           # entry bar has not closed yet
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    entry = o[i + 1]
    if not (zone_lo <= entry <= zone_hi):
        return dict(status="missed", fill=entry, exit_time=df.index[i + 1], exit_price=entry, gross_r=0.0,
                    stop_frac=np.nan, bars=0)
    risk = abs(entry - sl)

    def done(status, j, px):
        return dict(status=status, fill=entry, exit_time=df.index[j], exit_price=px,
                    gross_r=side * (px - entry) / risk, stop_frac=risk / entry, bars=j - i)

    last = min(i + horizon, len(df) - 1)
    for j in range(i + 1, last + 1):
        hit_sl = (l[j] <= sl) if side == 1 else (h[j] >= sl)
        hit_tp = (h[j] >= tp) if side == 1 else (l[j] <= tp)
        if hit_sl:                                            # SL first when both are touched
            gap = j > i + 1 and ((o[j] < sl) if side == 1 else (o[j] > sl))
            return done("sl", j, o[j] if gap else sl)
        if hit_tp:
            return done("tp", j, tp)
    if i + horizon <= len(df) - 1:
        return done("time", i + horizon, c[i + horizon])
    return None


def resolve_trend(row, df: pd.DataFrame) -> dict | None:
    from .trend import TrendParams, run_trend

    entry_n, exit_n = (int(x) for x in str(row["params"]).split("/"))
    trades = run_trend(df, TrendParams(entry_n, exit_n), 0.0)
    m = trades[pd.to_datetime(trades["time"], utc=True) == pd.Timestamp(row["signal_time"])]
    if m.empty:
        return None
    t = m.iloc[0]
    if bool(t["open"]) or bool(t["exit_pending"]):
        return None
    return dict(status="exit", fill=t["entry"], exit_time=t["exit_time"], exit_price=t["exit"], gross_r=t["r"],
                stop_frac=t["stop_frac"], bars=int(t["bars"]))


def update(path, symbol: str, interval: str, df: pd.DataFrame, modes=None) -> list[dict]:
    """Resolve open trades of this symbol/interval. Returns the rows that just closed (or were missed)."""
    led = load(path)
    sel = (led["symbol"] == symbol) & (led["interval"] == interval) & (led["status"] == "open")
    if modes:
        sel &= led["mode"].isin(modes)
    newly = []
    for i in led.index[sel]:
        row = led.loc[i]
        res = (resolve_trend(row, df) if row["mode"] == "trend" else
               simulate_levels(df, row["signal_time"], int(row["side"]), float(row["tp"]), float(row["sl"]),
                               int(row["horizon"]), float(row["zone_lo"]), float(row["zone_hi"])))
        if res is None:
            continue
        for k, v in res.items():
            led.loc[i, k] = v if not isinstance(v, pd.Timestamp) else v.isoformat()
        newly.append(led.loc[i].to_dict())
    if newly:
        save(led, path)
    return newly


def mark_notified(path, ids) -> None:
    led = load(path)
    led.loc[led["id"].isin(ids), "notified"] = True
    save(led, path)


def net_r(row_or_df, cost_bps: float):
    """Net R after round-trip fees: gross R minus (fee fraction / stop fraction)."""
    return row_or_df["gross_r"] - (cost_bps / 1e4) / row_or_df["stop_frac"]


def close_message(row, cost_bps: float = 8.0) -> str:
    """Short ASCII Telegram message for a trade that just finished."""
    d = "LONG" if int(row["side"]) == 1 else "SHORT"
    head = f"{d} {row['symbol']} {row['interval']} ({row['mode']})"
    if row["status"] == "missed":
        return f"TERLEPAS: {head}\nHarga buka {float(row['fill']):.5g} di luar julat masuk, tiada trade (paper)."
    label = {"tp": "TP KENA", "sl": "SL KENA", "time": "TAMAT MASA", "exit": "KELUAR (peraturan trend)"}[row["status"]]
    n = float(net_r(row, cost_bps))
    return (f"{label}: {head}\nEntry {float(row['fill']):.5g} -> Keluar {float(row['exit_price']):.5g} "
            f"({int(row['bars'])} bar)\nHasil: {float(row['gross_r']):+.2f} R (bersih {n:+.2f} R @ {cost_bps:g} bps)")


def summarize(led: pd.DataFrame, cost_bps: float) -> dict:
    c = led[led["status"].isin(CLOSED)].copy()
    out = dict(open=int((led["status"] == "open").sum()), missed=int((led["status"] == "missed").sum()), n=len(c))
    if c.empty:
        return out
    r = net_r(c.astype({"gross_r": float, "stop_frac": float}), cost_bps).to_numpy(float)
    eq = np.cumprod(1 + 0.01 * r)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float("nan")
    out.update(win=float((r > 0).mean()), exp=float(r.mean()), pf=float(gains / losses) if losses > 0 else float("inf"),
               total_r=float(r.sum()), ret=float(eq[-1] - 1), dd=float((eq / np.maximum.accumulate(eq) - 1).min()),
               ci=(float(r.mean() - 1.96 * se), float(r.mean() + 1.96 * se)) if len(r) > 1 else (float("nan"),) * 2,
               tp=int((c["status"] == "tp").sum()), sl=int((c["status"] == "sl").sum()),
               other=int((~c["status"].isin(["tp", "sl"])).sum()))
    return out


def report_text(path=LEDGER_FILE, mode: str = "all", costs=(8.0, 20.0), last: int = 8) -> str:
    led = load(path)
    if mode != "all":
        led = led[led["mode"] == mode]
    if led.empty:
        return "Belum ada rekod paper trade."
    lines = [f"PAPER TRADE ({mode}) | jumlah signal {len(led)} | masih terbuka {int((led.status == 'open').sum())} "
             f"| terlepas (di luar julat) {int((led.status == 'missed').sum())}"]
    for cost in costs:
        s = summarize(led, cost)
        if not s["n"]:
            lines.append(f"  yuran {cost:g} bps: belum ada trade tamat")
            continue
        lines.append(f"  yuran {cost:g} bps: {s['n']} trade (TP {s['tp']}, SL {s['sl']}, lain {s['other']}) | menang {s['win']:.0%} | "
                     f"purata {s['exp']:+.2f} R [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] | PF {s['pf']:.2f} | jumlah {s['total_r']:+.1f} R "
                     f"| akaun (risiko 1%) {s['ret']:+.1%}, drawdown {s['dd']:.1%}")
    closed = led[led["status"].isin(FINAL)].tail(last)
    if not closed.empty:
        lines.append("Terakhir:")
        for _, r in closed.iterrows():
            lines.append(f"  {str(r['signal_time'])[:16]} {r['symbol']} {'LONG' if int(r['side']) == 1 else 'SHORT'} "
                         f"{r['status']:6s} {float(r['gross_r']):+.2f} R")
    return "\n".join(lines)
