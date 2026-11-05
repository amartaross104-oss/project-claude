from dataclasses import dataclass


BARS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6}
HORIZON_DEFAULT = {"15m": 48, "1h": 24, "4h": 12}


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    # Triple-barrier: TP and SL are both tp_mult * ATR away from entry.
    tp_mult: float = 1.5
    horizon: int | None = None   # max bars a trade may stay open; None -> per-interval default
    # Signal filter on model probabilities.
    thr: float = 0.45            # min probability of the chosen direction
    margin: float = 0.10         # min gap between chosen and opposite direction (main filter)
    risk: float = 0.01           # fraction of equity risked per trade
    cost_bps: float | None = None  # round-trip cost in bps; None -> asset default
    # Walk-forward.
    min_train_bars: int | None = None  # None -> 180 days of bars
    step_bars: int | None = None       # None -> 30 days of bars
    seed: int = 42

    def __post_init__(self):
        bpd = BARS_PER_DAY[self.interval]
        if self.horizon is None:
            self.horizon = HORIZON_DEFAULT[self.interval]
        if self.min_train_bars is None:
            self.min_train_bars = 180 * bpd
        if self.step_bars is None:
            self.step_bars = 30 * bpd


def default_cost_bps(symbol: str) -> float:
    """Conservative round-trip cost (fees + spread + slippage) in basis points."""
    from .data import is_crypto

    return 20.0 if is_crypto(symbol) else 2.0
