"""End-to-end report generation pipeline producing both HTML and PDF deliverables."""

import json
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.config import config
from src.ingestion.price_client import PriceClient
from src.ingestion.storage import RawDataStore
from src.sentiment.contracts import RawPost, NewsArticle, TickerExtraction
from src.sentiment.extractor import TickerIntelligenceExtractor
from src.models.walk_forward import run_full_walk_forward_evaluation
from src.report.generator import generate_html_report
from src.report.pdf_export import generate_pdf_report

logger = logging.getLogger(__name__)


def build_all_reports(use_sample: bool = True) -> None:
    """Execute end-to-end extraction and reporting across the universe."""
    config.ensure_directories()
    price_client = PriceClient(use_cache=True)
    data_store = RawDataStore()
    extractor = TickerIntelligenceExtractor()

    assets = config.get_all_assets()
    extractions: List[TickerExtraction] = []

    # 1. Ensure metrics and predictions exist
    metrics_path = config.results_dir / "metrics.json"
    preds_path = config.results_dir / "predictions.csv"

    if not metrics_path.exists() or not preds_path.exists():
        logger.info("Metrics not found on disk. Running walk-forward evaluation...")
        metrics_data, preds_df = run_full_walk_forward_evaluation(price_client, data_store, use_sample=use_sample)
    else:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        preds_df = pd.read_csv(preds_path)

    # 2. Extract Per-Ticker Intelligence
    logger.info("Generating per-ticker intelligence across 10 assets...")
    for asset in assets:
        clean_ticker = asset.ticker.replace(".", "_")

        # Load prices
        if use_sample:
            price_file = config.sample_dir / "prices" / f"{clean_ticker}_1d.parquet"
            price_df = pd.read_parquet(price_file) if price_file.exists() else price_client.fetch_ohlcv(asset.ticker, "1d")
        else:
            price_df = price_client.fetch_ohlcv(asset.ticker, "1d")

        # Load posts
        if use_sample:
            post_file = config.sample_dir / "posts" / f"{clean_ticker}_posts.json"
            posts = []
            if post_file.exists():
                with open(post_file, "r", encoding="utf-8") as f:
                    posts = [RawPost(**p) for p in json.load(f)]
        else:
            posts = data_store.load_posts(asset.ticker, "lihkg") or []
            posts += data_store.load_posts(asset.ticker, "aastocks") or []
            posts += data_store.load_posts(asset.ticker, "naver_finance") or []
            posts += data_store.load_posts(asset.ticker, "telegram") or []

        # Load news
        if use_sample:
            news_file = config.sample_dir / "news" / f"{clean_ticker}_news.json"
            news = []
            if news_file.exists():
                with open(news_file, "r", encoding="utf-8") as f:
                    news = [NewsArticle(**n) for n in json.load(f)]
        else:
            news = data_store.load_news(asset.ticker) or []

        intel = extractor.extract_ticker_intelligence(asset, posts, news, price_df)
        extractions.append(intel)

    # 3. Generate HTML Report
    html_path = config.reports_dir / "report.html"
    generate_html_report(extractions, metrics_data, preds_df, output_path=html_path)

    # 4. Generate PDF Report
    pdf_path = config.reports_dir / "report.pdf"
    generate_pdf_report(extractions, metrics_data, output_path=pdf_path)

    # 5. Mirror to docs/ for instant GitHub Pages deployment
    docs_dir = config.base_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(html_path, docs_dir / "index.html")
    shutil.copy2(pdf_path, docs_dir / "report.pdf")

    logger.info("Report generation complete: %s, %s, and docs/ (GitHub Pages ready)", html_path, pdf_path)


if __name__ == "__main__":
    build_all_reports()
