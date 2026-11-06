"""Technical analysis indicators (RSI, MACD, Bollinger Bands, ATR, Momentum, Volatility)."""

import numpy as np
import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard quantitative technical indicators on an OHLCV DataFrame.

    Requires columns: ['Open', 'High', 'Low', 'Close', 'Volume'].
    Returns a copy with added technical feature columns.
    """
    res = df.copy()
    close = res["Close"]
    high = res["High"]
    low = res["Low"]
    volume = res["Volume"]

    # 1. Lagged Returns
    res["ret_1d"] = close.pct_change(1)
    res["ret_2d"] = close.pct_change(2)
    res["ret_5d"] = close.pct_change(5)
    res["ret_10d"] = close.pct_change(10)

    # 2. Rolling Volatility
    res["vol_5d"] = res["ret_1d"].rolling(window=5).std()
    res["vol_20d"] = res["ret_1d"].rolling(window=20).std()

    # 3. RSI (14-period standard Relative Strength Index)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.ewm(com=13, adjust=False).mean()
    roll_down = loss.ewm(com=13, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    res["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 4. MACD (12, 26, 9 EMA)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    res["macd"] = (ema_12 - ema_26) / (close + 1e-9) # Normalized by close
    res["macd_signal"] = res["macd"].ewm(span=9, adjust=False).mean()
    res["macd_hist"] = res["macd"] - res["macd_signal"]

    # 5. Bollinger Bands (20-period, 2-std)
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    res["bb_position"] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-9)

    # 6. Normalized ATR (Average True Range / Close)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res["natr_14"] = tr.rolling(window=14).mean() / (close + 1e-9)

    # 7. Volume Momentum (Z-score vs 20-period rolling average)
    vol_mean = volume.rolling(window=20).mean()
    vol_std = volume.rolling(window=20).std()
    res["volume_zscore"] = (volume - vol_mean) / (vol_std + 1e-9)

    return res
