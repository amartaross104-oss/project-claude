import numpy as np
import pandas as pd

from signalbot.patterns import ALL_PATTERNS, all_patterns, candlestick_patterns, double_top_bottom
from tests.test_signalbot import synth


def frame(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def test_all_patterns_causal():
    df = synth(700, seed=5)
    full = all_patterns(df)
    assert set(full) == set(ALL_PATTERNS)
    for k in (300, 450, 650):
        part = all_patterns(df.iloc[: k + 1])
        for name in ALL_PATTERNS:
            assert full[name][k] == part[name][k], (name, k)


def test_hammer_and_engulfing_on_handmade_candles():
    base = [[100 - i, 100.5 - i, 99.2 - i, 99.6 - i] for i in range(30)]      # steady decline
    hammer = [70.0, 70.3, 66.0, 70.2]                                          # long lower shadow, tiny body
    df = frame(base + [hammer])
    assert candlestick_patterns(df)["hammer"][-1] == 1
    engulf = frame(base + [[69.8, 70.0, 68.9, 69.0], [68.8, 71.5, 68.7, 71.2]])  # bearish then bigger bullish
    assert candlestick_patterns(engulf)["bull_engulfing"][-1] == 1
    # same shape after an UP-trend is a hanging man (bearish), not a hammer
    up = [[100 + i, 100.8 + i, 99.5 + i, 100.5 + i] for i in range(30)]
    df2 = frame(up + [[130.0, 130.3, 126.0, 130.2]])
    p = candlestick_patterns(df2)
    assert p["hammer"][-1] == 0 and p["hanging_man"][-1] == -1


def test_double_bottom_fires_once_on_neckline_break():
    x = np.r_[np.linspace(120, 100, 25), np.linspace(100, 112, 10), np.linspace(112, 100.3, 10),
              np.linspace(100.3, 113.5, 10), np.full(10, 114.0)]
    df = frame([[v, v + 0.4, v - 0.4, v] for v in x])
    # give the series some range so ATR is non-zero and the bottoms are real swing lows
    sig = double_top_bottom(df, k=4, min_gap=5, depth_atr=1.0, tol_atr=1.0)["double_bottom"]
    assert (sig == 1).sum() == 1
    first = int(np.flatnonzero(sig)[0])
    assert df["close"].iloc[first] > 112 and df["close"].iloc[first - 1] <= 112.5   # first close above the neckline
