"""LightGBM classifier + purged expanding-window walk-forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .config import Config


def new_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=seed,
        verbose=-1,
    )


def _proba3(model: LGBMClassifier, X: np.ndarray) -> np.ndarray:
    """predict_proba as an (n, 3) array in class order 0,1,2 even if a class was unseen."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, col]
    return out


def walk_forward(X: pd.DataFrame, y: np.ndarray, valid: np.ndarray, cfg: Config) -> np.ndarray:
    """Out-of-sample class probabilities (NaN until enough training data).

    Training for a test block starting at `start` only uses bars < start - horizon, so no
    training label depends on prices inside the test block (purging).
    """
    n = len(X)
    Xv = X.to_numpy(float)
    proba = np.full((n, 3), np.nan)
    start = cfg.min_train_bars
    while start < n:
        end = min(start + cfg.step_bars, n)
        tr = valid & ~np.isnan(Xv).all(axis=1) & (np.arange(n) < start - cfg.horizon)
        if tr.sum() >= 500:
            m = new_model(cfg.seed).fit(Xv[tr], y[tr])
            proba[start:end] = _proba3(m, Xv[start:end])
        start = end
    return proba


def fit_final(X: pd.DataFrame, y: np.ndarray, valid: np.ndarray, cfg: Config) -> LGBMClassifier:
    Xv = X.to_numpy(float)
    mask = valid & ~np.isnan(Xv).all(axis=1)
    return new_model(cfg.seed).fit(Xv[mask], y[mask])


def predict_last(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return _proba3(model, X.to_numpy(float)[-1:])[0]
