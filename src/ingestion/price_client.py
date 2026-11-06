"""Price ingestion client fetching daily and hourly OHLCV data via yfinance, aligned strictly in UTC."""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import yfinance as yf
from src.config import config, AssetSpec

logger = logging.getLogger(__name__)


class PriceClient:
    """Client for downloading, standardizing, and caching OHLCV market data."""

    def __init__(self, cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.cache_dir = cache_dir or (config.raw_dir / "prices")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    def _get_cache_path(self, ticker: str, interval: str) -> Path:
        clean_ticker = ticker.replace(".", "_")
        return self.cache_dir / f"{clean_ticker}_{interval}.parquet"

    def fetch_ohlcv(
        self,
        ticker: str,
        interval: str = "1d",
        period: Optional[str] = None,
        days_back: int = 365,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """Fetch OHLCV price bars for a given ticker and interval, standardized to UTC.

        Args:
            ticker: Yahoo finance symbol (e.g. '0700.HK', '005930.KS')
            interval: Bar resolution ('1d', '1h', etc.)
            period: Pre-defined period string (e.g. '1y', '1mo')
            days_back: Number of historical days to fetch if period is not set
            force_refresh: Whether to ignore local cache

        Returns:
            DataFrame with UTC datetime index and columns [Open, High, Low, Close, Volume]
        """
        cache_file = self._get_cache_path(ticker, interval)

        # 1. Return from cache if available and not stale
        if self.use_cache and not force_refresh and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info("Loaded cached price data for %s (%s): %d bars", ticker, interval, len(df))
                return df
            except Exception as e:
                logger.warning("Failed reading cache for %s (%s): %s. Refetching.", ticker, interval, e)

        # 2. Determine download parameters
        if period is None:
            if interval == "1h":
                # Yahoo Finance hourly limit is max 730 days, recent 10-60 days typical
                period = f"{min(days_back, 60)}d"
            else:
                period = f"{days_back}d"

        logger.info("Downloading %s OHLCV (interval=%s, period=%s) via yfinance...", ticker, interval, period)
        try:
            yf_ticker = yf.Ticker(ticker)
            raw_df = yf_ticker.history(period=period, interval=interval, auto_adjust=True)
        except Exception as e:
            logger.error("Error downloading price data for %s: %s", ticker, e)
            raw_df = pd.DataFrame()

        if raw_df.empty:
            logger.warning("Empty price dataframe returned for %s (%s)", ticker, interval)
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # 3. Clean columns & normalize timezone strictly to UTC
        df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Handle index timezone: convert to UTC
        if df.index.tz is None:
            # Determine native exchange timezone from config
            asset = config.get_asset_by_ticker(ticker)
            if asset:
                exch_name = "HKEX" if ".HK" in ticker.upper() else "KRX"
                native_tz = config.get_exchanges().get(exch_name, {}).timezone or "UTC"
            else:
                native_tz = "Asia/Hong_Kong" if ".HK" in ticker.upper() else "Asia/Seoul"

            df.index = df.index.tz_localize(native_tz).tz_convert("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "timestamp_utc"
        df = df.sort_index()

        # Deduplicate and remove bars with NaN close or 0 volume if daily
        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna(subset=["Close"])

        # 4. Save to cache
        if self.use_cache and not df.empty:
            try:
                df.to_parquet(cache_file)
                logger.info("Cached %d price bars for %s (%s) to %s", len(df), ticker, interval, cache_file)
            except Exception as e:
                logger.warning("Failed caching price data for %s: %s", ticker, e)

        return df

    def fetch_universe(
        self,
        daily_days: int = 365,
        hourly_days: int = 10,
        force_refresh: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch both daily (long history) and hourly (recent) data for all configured assets."""
        assets = config.get_all_assets()
        daily_frames = []
        hourly_frames = []

        for asset in assets:
            logger.info("Fetching prices for %s (%s)...", asset.ticker, asset.name)
            d_df = self.fetch_ohlcv(asset.ticker, interval="1d", days_back=daily_days, force_refresh=force_refresh)
            if not d_df.empty:
                d_df = d_df.copy()
                d_df["ticker"] = asset.ticker
                daily_frames.append(d_df)

            h_df = self.fetch_ohlcv(asset.ticker, interval="1h", days_back=hourly_days, force_refresh=force_refresh)
            if not h_df.empty:
                h_df = h_df.copy()
                h_df["ticker"] = asset.ticker
                hourly_frames.append(h_df)

        all_daily = pd.concat(daily_frames).reset_index() if daily_frames else pd.DataFrame()
        all_hourly = pd.concat(hourly_frames).reset_index() if hourly_frames else pd.DataFrame()

        return all_daily, all_hourly
