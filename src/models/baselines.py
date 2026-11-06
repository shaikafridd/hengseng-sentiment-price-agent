"""Naive baseline benchmark models for time-series evaluation."""

import numpy as np


class NaiveLastPriceModel:
    """Predicts yesterday's close price as tomorrow's price (Random Walk / Martingale null hypothesis).
    Implies predicted next return R_{t+1} = 0.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveLastPriceModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class NaiveMeanReturnModel:
    """Predicts historical mean return of the training set."""

    def __init__(self):
        self.mean_return_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveMeanReturnModel":
        self.mean_return_ = float(np.mean(y)) if len(y) > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_return_, dtype=float)
