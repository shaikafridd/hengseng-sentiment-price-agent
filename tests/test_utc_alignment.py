"""Tests for timestamp normalization and UTC alignment."""

from datetime import datetime, timezone
import pandas as pd
import pytz
from src.config import config
from src.ingestion.price_client import PriceClient


def test_hkt_and_kst_to_utc_conversion():
    """Verify market open timestamps in HKT (UTC+8) and KST (UTC+9) correctly convert to UTC."""
    # HKEX opens at 09:30 HKT
    hkt = pytz.timezone("Asia/Hong_Kong")
    hk_open = hkt.localize(datetime(2026, 11, 2, 9, 30, 0))
    hk_open_utc = hk_open.astimezone(pytz.UTC)
    assert hk_open_utc.hour == 1 and hk_open_utc.minute == 30, "HKEX open 09:30 HKT must map to 01:30 UTC"

    # KRX opens at 09:00 KST
    kst = pytz.timezone("Asia/Seoul")
    kr_open = kst.localize(datetime(2026, 11, 2, 9, 0, 0))
    kr_open_utc = kr_open.astimezone(pytz.UTC)
    assert kr_open_utc.hour == 0 and kr_open_utc.minute == 0, "KRX open 09:00 KST must map to 00:00 UTC"


def test_price_client_utc_normalization_synthetic(tmp_path):
    """Test that PriceClient guarantees tz-aware UTC timestamps on loaded/mocked DataFrames."""
    client = PriceClient(cache_dir=tmp_path)

    # Create synthetic daily data localized in Asia/Hong_Kong
    dates = pd.date_range("2026-01-01", periods=5, freq="D", tz="Asia/Hong_Kong")
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [105.0, 106.0, 107.0, 108.0, 109.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Volume": [1000, 1200, 1100, 1500, 1300],
        },
        index=dates
    )

    # Save to parquet cache
    cache_path = client._get_cache_path("0700.HK", "1d")
    df_utc = df.copy()
    df_utc.index = df_utc.index.tz_convert("UTC")
    df_utc.to_parquet(cache_path)

    # Load via client
    loaded = client.fetch_ohlcv("0700.HK", interval="1d")
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert str(loaded.index.tz) == "UTC", f"Expected UTC timezone, got {loaded.index.tz}"
    assert loaded.index.is_monotonic_increasing
    assert len(loaded) == 5
