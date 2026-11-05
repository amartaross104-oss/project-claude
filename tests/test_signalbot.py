import numpy as np
import pandas as pd

from signalbot.backtest import pick_side, run_backtest, trade_r
from signalbot.config import Config
from signalbot.features import atr, make_features
from signalbot.labels import AMBIGUOUS, DOWN, TIMEOUT, UP, barrier_outcomes, to_class


def synth(n=600, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.002, n)) * close
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + spread,
         "low": np.minimum(open_, close) - spread, "close": close,
         "volume": rng.uniform(1, 2, n)},
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


def test_features_are_causal():
    df = synth()
    full = make_features(df)
    k = 400
    part = make_features(df.iloc[: k + 1])
    pd.testing.assert_series_equal(full.iloc[k], part.iloc[k], check_names=False)


def flat(n, price=100.0):
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 1.0},
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


def test_barrier_up_down_timeout_ambiguous():
    df = flat(40)
    a = pd.Series(1.0, index=df.index)   # barriers at 100 +/- 1.5
    df.loc[df.index[5], "high"] = 102.0  # up hit on bar 5 for signal at t=3
    b = barrier_outcomes(df, a, tp_mult=1.5, horizon=10)
    assert b.outcome[3] == UP and b.exit_idx[3] == 5 and b.exit_price[3] == 101.5
    assert b.entry[3] == 100.0

    df = flat(40); df.loc[df.index[5], "low"] = 98.0
    assert barrier_outcomes(df, a, 1.5, 10).outcome[3] == DOWN

    assert barrier_outcomes(flat(40), a, 1.5, 10).outcome[3] == TIMEOUT

    df = flat(40); df.loc[df.index[5], ["high", "low"]] = [102.0, 98.0]
    assert barrier_outcomes(df, a, 1.5, 10).outcome[3] == AMBIGUOUS


def test_label_has_no_future_beyond_horizon():
    df = flat(30); a = pd.Series(1.0, index=df.index)
    b = barrier_outcomes(df, a, 1.5, 10)
    assert not b.valid[-11:].any() and b.valid[0]
    assert set(to_class(b.outcome)) == {1}


def test_trade_r_accounting():
    df = flat(40); a = pd.Series(1.0, index=df.index)
    df.loc[df.index[5], "high"] = 102.0
    b = barrier_outcomes(df, a, 1.5, 10)
    stop = 1.5 / 100
    assert np.isclose(trade_r(1, b, 3, stop, 0.0), 1.0)    # long wins +1R
    assert np.isclose(trade_r(-1, b, 3, stop, 0.0), -1.0)  # short loses -1R
    assert np.isclose(trade_r(1, b, 3, stop, 0.0015), 0.9)  # cost = 0.1R


def test_pick_side_and_one_position_at_a_time():
    cfg = Config(thr=0.45, margin=0.10)
    assert pick_side(np.array([0.2, 0.05, 0.75]), cfg) == 1
    assert pick_side(np.array([0.75, 0.05, 0.2]), cfg) == -1
    assert pick_side(np.array([0.48, 0.04, 0.48]), cfg) == 0
    assert pick_side(np.array([np.nan] * 3), cfg) == 0

    df = flat(60); a = pd.Series(1.0, index=df.index)
    b = barrier_outcomes(df, a, 1.5, 10)
    proba = np.tile([0.1, 0.05, 0.85], (60, 1))
    res = run_backtest(df, b, a, proba, Config(cost_bps=0.0))
    idx = list(df.index)
    pos = [idx.index(t) for t in res.trades["time"]]
    assert all(y - x >= 10 for x, y in zip(pos, pos[1:]))  # no overlapping trades
