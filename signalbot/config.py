from dataclasses import dataclass


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    # Triple-barrier: TP and SL are both tp_mult * ATR away from entry.
    tp_mult: float = 1.5
    horizon: int = 24            # max bars a trade may stay open
    # Signal filter on model probabilities.
    thr: float = 0.45            # min probability of the chosen direction
    margin: float = 0.10         # min gap between chosen and opposite direction (main filter)
    risk: float = 0.01           # fraction of equity risked per trade
    cost_bps: float | None = None  # round-trip cost in bps; None -> asset default
    # Walk-forward.
    min_train_bars: int = 24 * 180
    step_bars: int = 24 * 30
    seed: int = 42


def default_cost_bps(symbol: str) -> float:
    """Conservative round-trip cost (fees + spread + slippage) in basis points."""
    from .data import is_crypto

    return 20.0 if is_crypto(symbol) else 2.0
