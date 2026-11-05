"""Technical features. Every feature at bar t uses only data up to and including bar t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    a = atr(df)
    ret1 = c.pct_change()
    f = pd.DataFrame(index=df.index)

    f["atr_pct"] = a / c
    for n in (1, 3, 6, 12, 24, 72):
        f[f"ret_{n}"] = c.pct_change(n)
    f["rsi14"] = rsi(c)

    ema = {n: c.ewm(span=n, adjust=False).mean() for n in (12, 20, 26, 50, 200)}
    f["c_ema20"] = c / ema[20] - 1
    f["ema20_50"] = ema[20] / ema[50] - 1
    f["ema50_200"] = ema[50] / ema[200] - 1
    macd = ema[12] - ema[26]
    f["macd_hist_atr"] = (macd - macd.ewm(span=9, adjust=False).mean()) / a

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    f["bb_width"] = 4 * sd / mid

    vol24, vol168 = ret1.rolling(24).std(), ret1.rolling(168).std()
    f["vol_24"] = vol24
    f["vol_ratio"] = vol24 / vol168
    f["range_atr"] = (h - l) / a
    f["body_atr"] = (c - o) / a

    hour, dow = df.index.hour, df.index.dayofweek
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    if (df["volume"] > 0).mean() > 0.5:  # forex feeds from Yahoo have no real volume
        v = df["volume"]
        f["vol_z"] = (v - v.rolling(48).mean()) / v.rolling(48).std()

    return f.replace([np.inf, -np.inf], np.nan)
