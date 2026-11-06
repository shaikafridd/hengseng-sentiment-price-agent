"""Unified CLI runner for the HengSeng & KRX Sentiment Price Prediction Agent.

Commands:
    demo      - Run the entire pipeline offline on bundled sample data (Zero API keys required)
    ingest    - Ingest prices (yfinance), social posts (Apify), and news (Exa)
    sentiment - Extract sentiment, position bias, and S/R levels per ticker
    features  - Generate zero-leakage technical and sentiment feature matrices
    train     - Execute expanding-window walk-forward validation and output metrics
    report    - Generate interactive HTML (Plotly) and executive PDF reports
    all       - Run the complete end-to-end pipeline
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("sentiment-agent-cli")

from src.config import config
from src.ingestion.price_client import PriceClient
from src.ingestion.storage import RawDataStore
from src.ingestion.apify_client import ApifyScraperClient
from src.ingestion.exa_client import ExaNewsClient
from src.sentiment.contracts import RawPost, NewsArticle, TickerExtraction
from src.sentiment.extractor import TickerIntelligenceExtractor
from src.features.alignment import build_feature_target_matrix
from src.models.walk_forward import run_full_walk_forward_evaluation
from src.report.orchestrator import build_all_reports


def cmd_ingest(demo: bool = False, force_refresh: bool = False):
    """Ingest prices, social discussions, and financial news."""
    logger.info("=== STEP 1: INGESTION PIPELINE (Demo Mode=%s) ===", demo)
    config.ensure_directories()

    price_client = PriceClient(use_cache=not force_refresh)
    data_store = RawDataStore()
    apify_client = ApifyScraperClient(data_store=data_store)
    exa_client = ExaNewsClient(data_store=data_store)

    assets = config.get_all_assets()

    if demo:
        logger.info("Running in --demo mode. Utilizing pre-bundled sample dataset in data/sample/.")
        logger.info("Ingestion verified for 10 assets (HKEX & KRX). No API calls made.")
        return

    logger.info("Fetching price data from yfinance (daily & hourly) for %d assets...", len(assets))
    for asset in assets:
        price_client.fetch_ohlcv(asset.ticker, interval="1d", days_back=365, force_refresh=force_refresh)
        price_client.fetch_ohlcv(asset.ticker, interval="1h", days_back=10, force_refresh=force_refresh)

        # Scrape social discussions via Apify
        logger.info("Ingesting social discussions for %s via Apify...", asset.ticker)
        apify_client.scrape_all_for_asset(asset, limit_per_source=30)

        # Ingest Exa news
        logger.info("Ingesting Exa news articles for %s...", asset.ticker)
        exa_client.fetch_news_for_asset(asset, days_back=10, num_results=5, force_refresh=force_refresh)

    logger.info("=== INGESTION COMPLETE ===")


def cmd_sentiment(demo: bool = False):
    """Run multilingual sentiment scoring and per-ticker intelligence extraction."""
    logger.info("=== STEP 2: MULTILINGUAL EXTRACTION PIPELINE (Demo Mode=%s) ===", demo)
    extractor = TickerIntelligenceExtractor()
    price_client = PriceClient()
    data_store = RawDataStore()

    assets = config.get_all_assets()
    extractions = []

    for asset in assets:
        clean_ticker = asset.ticker.replace(".", "_")

        # Load prices
        if demo:
            price_file = config.sample_dir / "prices" / f"{clean_ticker}_1d.parquet"
            price_df = price_client.fetch_ohlcv(asset.ticker) if not price_file.exists() else __import__("pandas").read_parquet(price_file)
        else:
            price_df = price_client.fetch_ohlcv(asset.ticker, "1d")

        # Load posts
        if demo:
            post_file = config.sample_dir / "posts" / f"{clean_ticker}_posts.json"
            posts = []
            if post_file.exists():
                import json
                with open(post_file, "r", encoding="utf-8") as f:
                    posts = [RawPost(**p) for p in json.load(f)]
        else:
            posts = data_store.load_posts(asset.ticker, "lihkg") or []
            posts += data_store.load_posts(asset.ticker, "aastocks") or []
            posts += data_store.load_posts(asset.ticker, "naver_finance") or []
            posts += data_store.load_posts(asset.ticker, "telegram") or []

        # Load news
        if demo:
            news_file = config.sample_dir / "news" / f"{clean_ticker}_news.json"
            news = []
            if news_file.exists():
                import json
                with open(news_file, "r", encoding="utf-8") as f:
                    news = [NewsArticle(**n) for n in json.load(f)]
        else:
            news = data_store.load_news(asset.ticker) or []

        intel = extractor.extract_ticker_intelligence(asset, posts, news, price_df)
        extractions.append(intel)
        logger.info(
            "Ticker: %s | Close: %.2f | Sentiment: %s (%.2f) | Bias: %s | Mentions: %d (z=%.1f)",
            intel.ticker, intel.current_price, intel.sentiment_direction.upper(),
            intel.sentiment_score, intel.position_bias or "None", intel.mention_volume, intel.mention_volume_zscore
        )

    logger.info("=== SENTIMENT & ENTITY EXTRACTION COMPLETE (%d assets) ===", len(extractions))
    return extractions


def cmd_features(demo: bool = False):
    """Build technical and anti-leakage sentiment feature matrices."""
    logger.info("=== STEP 3: FEATURE ENGINEERING & ZERO-LEAKAGE ALIGNMENT ===")
    price_client = PriceClient()
    data_store = RawDataStore()
    assets = config.get_all_assets()

    total_rows = 0
    for asset in assets:
        clean_ticker = asset.ticker.replace(".", "_")
        if demo:
            price_file = config.sample_dir / "prices" / f"{clean_ticker}_1d.parquet"
            price_df = __import__("pandas").read_parquet(price_file)
            post_file = config.sample_dir / "posts" / f"{clean_ticker}_posts.json"
            import json
            with open(post_file, "r", encoding="utf-8") as f:
                posts = [RawPost(**p) for p in json.load(f)]
            news_file = config.sample_dir / "news" / f"{clean_ticker}_news.json"
            with open(news_file, "r", encoding="utf-8") as f:
                news = [NewsArticle(**n) for n in json.load(f)]
        else:
            price_df = price_client.fetch_ohlcv(asset.ticker, "1d")
            posts = data_store.load_posts(asset.ticker, "lihkg") or []
            news = data_store.load_news(asset.ticker) or []

        matrix, p_cols, all_cols = build_feature_target_matrix(price_df, posts, news)
        total_rows += len(matrix)
        logger.info("Matrix for %s: %d bars aligned (Features: %d price + %d sentiment)", asset.ticker, len(matrix), len(p_cols), len(all_cols) - len(p_cols))

    logger.info("=== FEATURE ALIGNMENT COMPLETE: %d total training/testing rows ===", total_rows)


def cmd_train(demo: bool = False):
    """Run expanding-window walk-forward validation and export metrics."""
    logger.info("=== STEP 4: TIME-SERIES WALK-FORWARD VALIDATION & ABLATION ===")
    price_client = PriceClient()
    data_store = RawDataStore()
    metrics, oof = run_full_walk_forward_evaluation(price_client, data_store, use_sample=demo)
    logger.info("Walk-forward cross-validation complete across %d tickers.", len(metrics))
    logger.info("Total out-of-sample predictions generated: %d", len(oof))
    logger.info("Results persisted to results/metrics.json and results/predictions.csv")


def cmd_report(demo: bool = False):
    """Generate interactive HTML and executive PDF reports."""
    logger.info("=== STEP 5: VISUAL REPORT GENERATION ===")
    build_all_reports(use_sample=demo)
    logger.info("Report outputs ready:")
    logger.info("  HTML Report: %s", config.reports_dir / "report.html")
    logger.info("  PDF Report:  %s", config.reports_dir / "report.pdf")


def cmd_demo():
    """Execute complete pipeline in offline zero-key demo mode."""
    logger.info("================================================================================")
    logger.info("RUNNING IN NO-KEY DEMO MODE (Offline Bundled Dataset)")
    logger.info("================================================================================")
    cmd_ingest(demo=True)
    cmd_sentiment(demo=True)
    cmd_features(demo=True)
    cmd_train(demo=True)
    cmd_report(demo=True)
    logger.info("================================================================================")
    logger.info("DEMO PIPELINE COMPLETE!")
    logger.info("Open reports/report.html in your browser or inspect reports/report.pdf")
    logger.info("================================================================================")


def cmd_all():
    """Run full live pipeline end-to-end."""
    logger.info("Running full end-to-end pipeline with live APIs and caching...")
    cmd_ingest(demo=False)
    cmd_sentiment(demo=False)
    cmd_features(demo=False)
    cmd_train(demo=False)
    cmd_report(demo=False)
    logger.info("Full pipeline execution complete.")


def main():
    parser = argparse.ArgumentParser(
        description="HengSeng & KRX Sentiment Price Prediction Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py demo                  # Run complete pipeline offline on bundled sample data (No keys required)
    python run.py all                   # Run complete live pipeline with API keys
    python run.py report --demo         # Build visual reports from sample data
    python run.py train                 # Execute walk-forward cross validation
        """
    )

    parser.add_argument(
        "command",
        choices=["ingest", "sentiment", "features", "train", "report", "all", "demo"],
        help="Pipeline phase to execute"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use bundled sample dataset in data/sample/ (offline mode without API keys)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download and re-scrape ignoring disk caches"
    )

    args = parser.parse_args()

    # If user invokes 'demo' command, automatically treat demo=True
    is_demo = args.demo or (args.command == "demo")

    if args.command == "demo":
        cmd_demo()
    elif args.command == "ingest":
        cmd_ingest(demo=is_demo, force_refresh=args.force_refresh)
    elif args.command == "sentiment":
        cmd_sentiment(demo=is_demo)
    elif args.command == "features":
        cmd_features(demo=is_demo)
    elif args.command == "train":
        cmd_train(demo=is_demo)
    elif args.command == "report":
        cmd_report(demo=is_demo)
    elif args.command == "all":
        cmd_all()


if __name__ == "__main__":
    main()
