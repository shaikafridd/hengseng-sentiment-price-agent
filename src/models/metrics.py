"""Statistical and financial metrics evaluation including Wilson score confidence intervals."""

import math
from typing import Dict, Tuple
import numpy as np


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion (hit rate).

    Returns:
        Tuple of (lower_bound_pct, upper_bound_pct) in percentage [0, 100]
    """
    if total == 0:
        return 0.0, 0.0

    p_hat = successes / total
    # z for 95% confidence is 1.95996
    z = 1.95996 if confidence == 0.95 else 1.64485
    z_sq = z * z

    denominator = 1.0 + z_sq / total
    center = (p_hat + z_sq / (2.0 * total)) / denominator
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z_sq / (4.0 * total * total)))) / denominator

    lower = max(0.0, center - spread) * 100.0
    upper = min(1.0, center + spread) * 100.0

    return round(lower, 2), round(upper, 2)


def compute_prediction_metrics(
    y_true_returns: np.ndarray,
    y_pred_returns: np.ndarray,
    current_prices: np.ndarray
) -> Dict[str, float]:
    """Calculate MAE, RMSE, MAPE on implied price and return, plus directional accuracy with 95% CI.

    Args:
        y_true_returns: Actual next-bar returns
        y_pred_returns: Predicted next-bar returns
        current_prices: Spot price at time t

    Returns:
        Dict of standardized evaluation metrics.
    """
    n = len(y_true_returns)
    if n == 0:
        return {
            "mae_price": 0.0, "rmse_price": 0.0, "mape_price": 0.0,
            "mae_return": 0.0, "directional_accuracy": 0.0,
            "ci_lower": 0.0, "ci_upper": 0.0, "sample_count": 0
        }

    # Implied actual and predicted prices
    p_true = current_prices * (1.0 + y_true_returns)
    p_pred = current_prices * (1.0 + y_pred_returns)

    # Price errors
    price_err = p_true - p_pred
    mae_price = float(np.mean(np.abs(price_err)))
    rmse_price = float(np.sqrt(np.mean(price_err ** 2)))
    mape_price = float(np.mean(np.abs(price_err / (p_true + 1e-9)))) * 100.0

    # Return errors
    ret_err = y_true_returns - y_pred_returns
    mae_return = float(np.mean(np.abs(ret_err)))

    # Directional Accuracy (Hit Rate)
    # Correct if sign matches or both are zero
    true_dir = np.sign(y_true_returns)
    pred_dir = np.sign(y_pred_returns)
    correct_hits = int(np.sum(true_dir == pred_dir))
    directional_acc = float((correct_hits / n) * 100.0)

    # 95% Wilson Score Confidence Interval
    ci_lower, ci_upper = wilson_score_interval(correct_hits, n, confidence=0.95)

    return {
        "mae_price": round(mae_price, 4),
        "rmse_price": round(rmse_price, 4),
        "mape_price": round(mape_price, 2),
        "mae_return": round(mae_return, 5),
        "directional_accuracy": round(directional_acc, 2),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "sample_count": n
    }
