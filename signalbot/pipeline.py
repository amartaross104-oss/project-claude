from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import data
from .config import Config, default_cost_bps
from .features import atr, make_features
from .labels import Barriers, barrier_outcomes, to_class


@dataclass
class Dataset:
    df: pd.DataFrame
    X: pd.DataFrame
    atr: pd.Series
    barriers: Barriers
    y: "object"


def build(cfg: Config, refresh: bool = False) -> Dataset:
    if cfg.cost_bps is None:
        cfg.cost_bps = default_cost_bps(cfg.symbol)
    df = data.load(cfg.symbol, cfg.interval, refresh=refresh)
    a = atr(df)
    b = barrier_outcomes(df, a, cfg.tp_mult, cfg.horizon)
    return Dataset(df, make_features(df), a, b, to_class(b.outcome))
