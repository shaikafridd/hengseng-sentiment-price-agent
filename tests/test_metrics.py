"""Test metric calculations, Wilson score confidence intervals, and baseline evaluations."""

import numpy as np
from src.models.metrics import compute_prediction_metrics, wilson_score_interval
from src.models.baselines import NaiveLastPriceModel, NaiveMeanReturnModel


def test_wilson_score_interval_properties():
    """Verify Wilson score confidence interval bounds."""
    # 50 out of 100 successes (50% hit rate)
    lower, upper = wilson_score_interval(50, 100, confidence=0.95)
    assert 40.0 < lower < 50.0
    assert 50.0 < upper < 60.0
    assert lower < upper

    # Perfect score 10 out of 10
    low_p, up_p = wilson_score_interval(10, 10)
    assert up_p == 100.0
    assert low_p > 65.0

    # 0 out of 10
    low_0, up_0 = wilson_score_interval(0, 10)
    assert low_0 == 0.0
    assert up_0 < 35.0


def test_prediction_metrics_computation():
    """Verify compute_prediction_metrics on known synthetic values."""
    y_true = np.array([0.02, -0.01, 0.03, -0.02])
    y_pred = np.array([0.01, -0.01, 0.02, 0.01]) # 3 out of 4 correct sign
    prices = np.array([100.0, 100.0, 100.0, 100.0])

    metrics = compute_prediction_metrics(y_true, y_pred, prices)
    assert metrics["directional_accuracy"] == 75.0
    assert metrics["mae_price"] > 0.0
    assert metrics["rmse_price"] >= metrics["mae_price"]
    assert metrics["ci_lower"] < 75.0 < metrics["ci_upper"]


def test_naive_baselines():
    """Verify NaiveLastPriceModel always predicts zero return."""
    X = np.ones((5, 3))
    y = np.array([0.01, -0.02, 0.03, -0.01, 0.02])

    naive_last = NaiveLastPriceModel().fit(X, y)
    preds = naive_last.predict(X)
    assert np.all(preds == 0.0)

    naive_mean = NaiveMeanReturnModel().fit(X, y)
    mean_preds = naive_mean.predict(X)
    assert np.allclose(mean_preds, np.mean(y))
