"""HTML visual report generator with interactive Plotly charts, rankings, and ablation summary."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import config
from src.sentiment.contracts import TickerExtraction

logger = logging.getLogger(__name__)


def generate_html_report(
    extractions: List[TickerExtraction],
    metrics_data: Dict[str, Any],
    predictions_df: pd.DataFrame,
    output_path: Path = config.reports_dir / "report.html"
) -> Path:
    """Generate a self-contained, responsive HTML report with embedded Plotly charts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Split extractions by exchange
    hk_extractions = [e for e in extractions if ".HK" in e.ticker]
    kr_extractions = [e for e in extractions if ".KS" in e.ticker]

    # Sort rankings: primary by sentiment score desc, secondary by mention volume desc
    hk_extractions.sort(key=lambda x: (x.sentiment_score, x.mention_volume), reverse=True)
    kr_extractions.sort(key=lambda x: (x.sentiment_score, x.mention_volume), reverse=True)

    # 2. Build Plotly Chart 1: Bubble Chart of Sentiment vs Mention Volume
    fig_bubble = go.Figure()
    all_ext = hk_extractions + kr_extractions
    tickers = [e.ticker for e in all_ext]
    names = [e.asset_name for e in all_ext]
    sent_scores = [e.sentiment_score for e in all_ext]
    mentions = [e.mention_volume for e in all_ext]
    confidences = [e.confidence_score * 35 + 15 for e in all_ext]
    colors = [
        "#10b981" if e.sentiment_direction == "bullish" else ("#ef4444" if e.sentiment_direction == "bearish" else "#6b7280")
        for e in all_ext
    ]

    fig_bubble.add_trace(go.Scatter(
        x=sent_scores,
        y=mentions,
        mode="markers+text",
        text=tickers,
        textposition="top center",
        marker=dict(
            size=confidences,
            color=colors,
            opacity=0.85,
            line=dict(width=1.5, color="#1e293b")
        ),
        hovertext=[
            f"<b>{t} ({n})</b><br>Sentiment: {s:+.2f}<br>Mentions: {m}<br>Position Bias: {e.position_bias or 'None'}<br>Confidence: {e.confidence_score:.2f}"
            for t, n, s, m, e in zip(tickers, names, sent_scores, mentions, all_ext)
        ],
        hoverinfo="text"
    ))
    fig_bubble.update_layout(
        title="<b>Asset Universe: Sentiment Score vs. Social Mention Volume</b>",
        xaxis_title="Sentiment Score (-1 Bearish to +1 Bullish)",
        yaxis_title="10-Day Social Mentions",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(range=[-1.05, 1.05], zeroline=True, zerolinewidth=1.5, zerolinecolor="#94a3b8"),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff"
    )
    bubble_chart_html = fig_bubble.to_html(full_html=False, include_plotlyjs="cdn")

    # 3. Build Plotly Chart 2: Sample Walk-Forward Actual vs Predicted Prices (0700.HK & 005930.KS)
    fig_pred = make_subplots(
        rows=1, cols=2,
        subplot_titles=("0700.HK (Tencent) Walk-Forward Predictions", "005930.KS (Samsung) Walk-Forward Predictions")
    )

    for idx, sample_ticker in enumerate(["0700.HK", "005930.KS"]):
        sub_df = predictions_df[predictions_df["ticker"] == sample_ticker]
        if not sub_df.empty:
            timestamps = pd.to_datetime(sub_df["timestamp_utc"])
            fig_pred.add_trace(
                go.Scatter(x=timestamps, y=sub_df["actual_target_price"], mode="lines+markers", name=f"{sample_ticker} Actual", line=dict(color="#0f172a", width=2)),
                row=1, col=idx+1
            )
            fig_pred.add_trace(
                go.Scatter(x=timestamps, y=sub_df["pred_price_gbr_all"], mode="lines", name=f"{sample_ticker} GBR (Price+Sent)", line=dict(color="#2563eb", width=1.8, dash="dot")),
                row=1, col=idx+1
            )
            fig_pred.add_trace(
                go.Scatter(x=timestamps, y=sub_df["pred_price_naive"], mode="lines", name=f"{sample_ticker} Naive Baseline", line=dict(color="#94a3b8", width=1.5, dash="dash")),
                row=1, col=idx+1
            )

    fig_pred.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        showlegend=True
    )
    pred_chart_html = fig_pred.to_html(full_html=False, include_plotlyjs=False)

    # 4. Helper HTML table row generator
    def render_ranking_rows(ext_list: List[TickerExtraction]) -> str:
        rows = []
        for rank, e in enumerate(ext_list, start=1):
            badge_color = "#dcfce7; color: #166534" if e.sentiment_direction == "bullish" else ("#fee2e2; color: #991b1b" if e.sentiment_direction == "bearish" else "#f3f4f6; color: #374151")
            pos_badge = f"<span class='badge' style='background: #e0e7ff; color: #3730a3;'>{e.position_bias.upper()}</span>" if e.position_bias else "<span style='color:#9ca3af;'>None</span>"
            supports_str = ", ".join(f"{s:,.1f}" for s in e.support_levels[:2]) if e.support_levels else "N/A"
            resist_str = ", ".join(f"{r:,.1f}" for r in e.resistance_levels[:2]) if e.resistance_levels else "N/A"

            rows.append(f"""
            <tr>
                <td style="font-weight: 600; text-align: center;">#{rank}</td>
                <td><b>{e.ticker}</b><br><small style="color: #64748b;">{e.asset_name}</small></td>
                <td style="font-weight: 600;">{e.current_price:,.2f}</td>
                <td><span class="badge" style="background: {badge_color};">{e.sentiment_direction.upper()} ({e.sentiment_score:+.2f})</span></td>
                <td>{pos_badge}</td>
                <td style="text-align: center;">{e.mention_volume} <small style="color: #64748b;">(z={e.mention_volume_zscore:+.1f})</small></td>
                <td style="text-align: center;">{e.bull_bear_ratio:.2f}</td>
                <td style="font-size: 0.85rem;"><span style="color: #16a34a;">S:</span> {supports_str}<br><span style="color: #dc2626;">R:</span> {resist_str}</td>
                <td style="text-align: center;"><b>{int(e.confidence_score * 100)}%</b></td>
            </tr>
            """)
        return "\n".join(rows)

    hk_table_rows = render_ranking_rows(hk_extractions)
    kr_table_rows = render_ranking_rows(kr_extractions)

    # 5. Model Ablation & Performance Table
    ablation_rows = []
    for ticker, data in metrics_data.items():
        naive = data.get("naive_baseline", {})
        gbr_p = data.get("gbr_price_only", {})
        gbr_all = data.get("gbr_price_sentiment", {})
        ablation = data.get("ablation", {})

        improved_icon = "✅ Improved" if ablation.get("gbr_sentiment_improved_rmse") else "⚠️ Neutral/Par"
        commentary = ablation.get("honest_commentary", "")

        ablation_rows.append(f"""
        <tr>
            <td><b>{ticker}</b></td>
            <td>{naive.get('rmse_price', 0):,.2f}</td>
            <td>{gbr_p.get('rmse_price', 0):,.2f} ({gbr_p.get('directional_accuracy', 0):.1f}%)</td>
            <td><b>{gbr_all.get('rmse_price', 0):,.2f}</b> ({gbr_all.get('directional_accuracy', 0):.1f}%)</td>
            <td>[{gbr_all.get('ci_lower', 0):.1f}%, {gbr_all.get('ci_upper', 0):.1f}%]</td>
            <td><span class="badge" style="background: #f1f5f9; color: #0f172a;">{improved_icon}</span></td>
            <td style="font-size: 0.85rem; color: #475569;">{commentary}</td>
        </tr>
        """)
    ablation_table_rows = "\n".join(ablation_rows)

    # 6. Top Influential Social Posts Cards
    top_post_cards = []
    for e in all_ext[:6]:
        if e.top_posts:
            p = e.top_posts[0]
            top_post_cards.append(f"""
            <div class="card post-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                    <b>{e.ticker} &bull; {p.get('source', 'social').upper()}</b>
                    <small style="color: #64748b;">Likes: {p.get('likes', 0)} | Views: {p.get('views', 0)}</small>
                </div>
                <p style="font-size: 0.9rem; color: #1e293b; margin: 0.2rem 0;">"{p.get('text', '')}"</p>
                <div style="font-size: 0.8rem; color: #475569; margin-top: 0.4rem;">
                    Score: <b>{p.get('score', 0):+.2f}</b> &bull; Weight: <b>{p.get('weight', 1.0)}x</b>
                </div>
            </div>
            """)
    post_cards_html = "\n".join(top_post_cards)

    # Assemble Full HTML Document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HengSeng Sentiment Price Prediction Agent - Executive Intelligence Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --primary: #0f172a;
            --accent: #2563eb;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 2rem 1.5rem;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            padding: 2.2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        }}
        .header h1 {{ margin: 0 0 0.5rem 0; font-size: 1.85rem; letter-spacing: -0.02em; }}
        .header p {{ margin: 0; color: #94a3b8; font-size: 0.95rem; }}
        .meta-strip {{
            display: flex;
            gap: 2rem;
            margin-top: 1.2rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            font-size: 0.85rem;
            color: #cbd5e1;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 1.25rem;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 0.6rem;
            color: #1e293b;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        th, td {{
            padding: 0.75rem 0.85rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        tr:hover {{ background: #f8fafc; }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 1.2rem;
        }}
        .post-card {{
            background: #f8fafc;
            border-left: 4px solid var(--accent);
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }}
        .callout-box {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 1rem 1.25rem;
            border-radius: 6px;
            font-size: 0.9rem;
            color: #1e3a8a;
            margin: 1rem 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <h1>HengSeng & KRX Sentiment-Driven Price Prediction Agent</h1>
                    <p>Hiring Assessment for AI Agents Developer Internship &bull; CrowdWisdomTrading &bull; Evaluator: Gilad</p>
                </div>
                <div>
                    <a href="report.pdf" download style="display: inline-flex; align-items: center; gap: 0.5rem; background: #2563eb; color: white; padding: 0.65rem 1.2rem; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 0.85rem; box-shadow: 0 4px 6px -1px rgba(37,99,235,0.3);">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        Download Executive PDF
                    </a>
                </div>
            </div>
            <div class="meta-strip">
                <div>Candidate: <b>Shaik Afrid</b></div>
                <div>Generated: <b>{generated_at}</b></div>
                <div>Coverage: <b>HKEX (5 Tickers) & KRX (5 Tickers)</b></div>
                <div>Validation: <b>Expanding-Window Walk-Forward (Zero-Leakage)</b></div>
            </div>
        </div>

        <!-- Bubble Chart Matrix -->
        <div class="card">
            <h2>Market Sentiment & Discussion Velocity Overview</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                Bubble size reflects confidence score (sample volume + source diversity + agreement). Color indicates direction (Green = Bullish, Red = Bearish, Grey = Neutral).
            </p>
            {bubble_chart_html}
        </div>

        <!-- Exchange Rankings -->
        <div class="card">
            <h2>1. Hong Kong Stock Exchange (HKEX) Social Sentiment Leaderboard</h2>
            <p style="color: var(--text-muted); font-size: 0.85rem;">Sources: LIHKG (連登討論區) + AAStocks Stock Forum + Exa AI Curated Financial News</p>
            <table>
                <thead>
                    <tr>
                        <th style="text-align: center;">Rank</th>
                        <th>Ticker / Asset</th>
                        <th>Price (HKD)</th>
                        <th>Sentiment</th>
                        <th>Trader Bias</th>
                        <th style="text-align: center;">Mentions (z)</th>
                        <th style="text-align: center;">Bull/Bear</th>
                        <th>Support / Resistance</th>
                        <th style="text-align: center;">Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {hk_table_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>2. Korea Exchange (KRX) Social Sentiment Leaderboard</h2>
            <p style="color: var(--text-muted); font-size: 0.85rem;">Sources: Naver Finance Jongmok Toronsil (종목토론실) + Financial Telegram Channels + Exa AI News</p>
            <table>
                <thead>
                    <tr>
                        <th style="text-align: center;">Rank</th>
                        <th>Ticker / Asset</th>
                        <th>Price (KRW)</th>
                        <th>Sentiment</th>
                        <th>Trader Bias</th>
                        <th style="text-align: center;">Mentions (z)</th>
                        <th style="text-align: center;">Bull/Bear</th>
                        <th>Support / Resistance</th>
                        <th style="text-align: center;">Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {kr_table_rows}
                </tbody>
            </table>
        </div>

        <!-- Out-of-Sample Predictions Chart -->
        <div class="card">
            <h2>3. Walk-Forward Out-of-Sample Predictions vs. Actuals</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                Out-of-sample prediction trajectories from expanding-window evaluation. Features at time <i>t</i> use strictly historical information available before prediction time.
            </p>
            {pred_chart_html}
        </div>

        <!-- Model Ablation & Results -->
        <div class="card">
            <h2>4. Rigorous Out-of-Sample Performance & Ablation Analysis</h2>
            <div class="callout-box">
                <b>Honest Evaluation Standard:</b> A 10-day social sentiment dataset provides high-frequency retail sentiment, but mega-cap price formation is largely dominated by institutional flows and macro news. Models must beat the Martingale Naive Baseline (R_(t+1) = 0) to prove statistical value.
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Naive Baseline RMSE</th>
                        <th>Price-Only GBR (Hit %)</th>
                        <th>Price + Sentiment GBR (Hit %)</th>
                        <th>95% Wilson CI (Hit Rate)</th>
                        <th>Ablation Status</th>
                        <th>Engineering Commentary</th>
                    </tr>
                </thead>
                <tbody>
                    {ablation_table_rows}
                </tbody>
            </table>
        </div>

        <!-- Influential Social Discussions -->
        <div class="card">
            <h2>5. High-Engagement Retail Discussion Intelligence</h2>
            <p style="color: var(--text-muted); font-size: 0.85rem;">Top weighted social posts across Cantonese, Korean, and English communities.</p>
            <div class="grid-2">
                {post_cards_html}
            </div>
        </div>

        <!-- Footer -->
        <div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 1.5rem 0;">
            Sentiment Price Prediction Agent &bull; CrowdWisdomTrading AI Agents Intern Assessment &bull; Candidate: Shaik Afrid
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Successfully generated interactive HTML report at %s", output_path)
    return output_path
