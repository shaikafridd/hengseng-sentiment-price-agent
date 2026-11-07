"""Test visual HTML and PDF report generation."""

from pathlib import Path
from src.config import config
from src.report.orchestrator import build_all_reports


def test_report_files_generation():
    """Verify that build_all_reports successfully produces report.html and report.pdf."""
    build_all_reports(use_sample=True)

    html_file = config.reports_dir / "report.html"
    pdf_file = config.reports_dir / "report.pdf"
    docs_html = config.base_dir / "docs" / "index.html"
    docs_pdf = config.base_dir / "docs" / "report.pdf"

    assert html_file.exists(), "HTML report was not generated"
    assert pdf_file.exists(), "PDF report was not generated"
    assert docs_html.exists(), "GitHub Pages docs/index.html was not mirrored"
    assert docs_pdf.exists(), "GitHub Pages docs/report.pdf was not mirrored"
    assert html_file.stat().st_size > 10000, "HTML report appears truncated or incomplete"
    assert pdf_file.stat().st_size > 2000, "PDF report appears truncated"

    # Verify content strings inside the generated HTML report
    with open(html_file, "r", encoding="utf-8") as f:
        html_text = f.read()

    assert "HengSeng & KRX Sentiment-Driven Price Prediction Agent" in html_text
    assert "Shaik Afrid" in html_text
    assert "0700.HK" in html_text
    assert "005930.KS" in html_text
    assert "Walk-Forward Out-of-Sample Predictions" in html_text
