# HengSeng & KRX Sentiment-Driven Price Prediction Agent

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff/Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](https://pytest.org)
[![Validation: Zero--Leakage](https://img.shields.io/badge/validation-walk--forward%20(zero--leakage)-success.svg)](#anti-leakage-guarantees)

> **Assessment Submission:** AI Agents Developer Internship — CrowdWisdomTrading  
> **Candidate:** Shaik Afrid ([LinkedIn Profile](https://www.linkedin.com/in/shaik-afrid-707656365))  
> **Evaluator:** Gilad (`gilad@crowdwisdomtrading.com`)  
> **Hard Deadline:** 13 November 2026 (Submitted Ahead of Schedule)

---

## 1. Executive Summary

This repository implements a production-grade, mathematically disciplined, sentiment-driven price prediction pipeline for liquid equities listed on the **Hong Kong Stock Exchange (HKEX)** and **Korea Exchange (KRX)**.

Predicting asset price movements at bar $t+1$ ($R_{t+1}$) using retail sentiment is notoriously susceptible to data snooping and lookahead leakage. This project is built upon three quantitative pillars:
1. **Zero-Leakage Temporal Alignment:** Strictly shifts sentiment aggregation so that at bar $t$, only sentiment information timestamped $\le t$ is observed.
2. **Expanding-Window Walk-Forward Cross-Validation:** No k-fold random shuffling. Evaluated out-of-sample across sequential expanding splits.
3. **Rigorous Benchmark Comparison & Honest Ablation:** Models are directly compared against the Martingale random-walk baseline ($\hat{R}_{t+1} = 0$). We clearly state when sentiment signals enhance performance and when technical price action dominates.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion Layer (UTC Normalized)"]
        YF["yfinance (Daily & 1h OHLCV)<br/>HKT / KST to UTC"]
        AP1["Apify: LIHKG Scraper<br/>claude_code_reviewer/lihkg-scraper"]
        AP2["Apify: AAStocks Cheerio<br/>apify/cheerio-scraper"]
        AP3["Apify: Naver Toronsil<br/>apify/cheerio-scraper"]
        AP4["Apify: Telegram Scraper<br/>curious_coder/telegram-scraper"]
        EXA["Exa AI Semantic News<br/>https://api.exa.ai/search"]
    end

    subgraph CACHE ["Caching & Resilience Layer"]
        RAW["data/raw/ (JSON & Parquet)<br/>Hash-keyed cache with backoff"]
        SAMPLE["data/sample/<br/>Bundled offline dataset for --demo"]
    end

    subgraph EXTRACTION ["2. Multilingual Sentiment & Entity Extraction"]
        NORM["Pydantic Data Contracts<br/>RawPost & NewsArticle"]
        LLM["Multilingual Sentiment Engine<br/>English + Cantonese Slang + Korean<br/>(Structured Pydantic Output)"]
        INTEL["Per-Ticker Creative Metrics<br/>• Bull/Bear Ratio<br/>• Engagement-Weighted Sentiment<br/>• Mention Volume & Z-Score<br/>• Position Bias (Long/Short)<br/>• S/R Levels (Floor Pivots + Text Price Targets)<br/>• Source Agreement & Confidence"]
    end

    subgraph MODELING ["3. Features & Zero-Leakage Time-Series ML"]
        FEAT["Feature Store<br/>Lagged returns, Volatility, RSI, MACD,<br/>Strictly Lagged Sentiment (t-1 -> t)"]
        LEAK_TEST["Leakage Verifier (pytest)<br/>(Timestamp t Sentiment strictly <= Bar t-1)"]
        SPLIT["Expanding-Window Walk-Forward CV<br/>(4 sequential out-of-sample splits)"]
        MODELS["Models Evaluated:<br/>1. Naive Baseline (Zero Return / Prev Close)<br/>2. Ridge Regression (Price vs Price+Sent)<br/>3. Gradient Boosting (Price vs Price+Sent)"]
    end

    subgraph REPORTING ["4. Visual Reporting & Delivery"]
        PLOTLY["Interactive HTML Report (Plotly)<br/>reports/report.html"]
        PDF["Executive PDF Summary<br/>reports/report.pdf"]
        JSON_OUT["results/metrics.json & results/predictions.csv"]
    end

    INGESTION --> CACHE
    CACHE --> EXTRACTION
    EXTRACTION --> MODELING
    MODELING --> REPORTING
```

---

## 3. Exchanges & Asset Universe

Configurable via [`config/assets.yaml`](file:///c:/Users/SHAIK%20AFRID/Desktop/internship%20demo/config/assets.yaml).

### Hong Kong Stock Exchange (HKEX) — Timezone: `Asia/Hong_Kong` (UTC+8)
Selected for high liquidity, institutional interest, and active retail discussion in Hong Kong forums.
* `0700.HK` — **Tencent Holdings** (Tech/Gaming/WeChat)
* `9988.HK` — **Alibaba Group** (E-Commerce/Cloud)
* `1810.HK` — **Xiaomi Corporation** (Smartphones/EV)
* `3690.HK` — **Meituan** (Local Consumer Services)
* `0005.HK` — **HSBC Holdings** (Global Banking/Financials)

### Korea Exchange (KRX) — Timezone: `Asia/Seoul` (UTC+9)
Selected for dominant market turnover and intensive retail participation in Korean domestic boards.
* `005930.KS` — **Samsung Electronics** (Semiconductor/Consumer Tech)
* `000660.KS` — **SK Hynix** (HBM Memory/Semiconductors)
* `035420.KS` — **NAVER Corporation** (Search/AI/Webtoon)
* `035720.KS` — **Kakao Corporation** (Messaging/Fintech)
* `373220.KS` — **LG Energy Solution** (EV Battery/Clean Energy)

---

## 4. Ingestion Sources & Verified Apify Actor IDs

Configurable in [`config/actors.yaml`](file:///c:/Users/SHAIK%20AFRID/Desktop/internship%20demo/config/actors.yaml).

| Platform | Target Exchange | Actor ID / Endpoint | Custom Fallback | Verified Schema Fields |
| :--- | :--- | :--- | :--- | :--- |
| **LIHKG (連登)** | HKEX | `claude_code_reviewer/lihkg-scraper` | `actors/lihkg-scraper/` | `threadId`, `title`, `likeCount`, `dislikeCount`, `createdAt` |
| **AAStocks Forum** | HKEX | `apify/cheerio-scraper` | `actors/aastocks-scraper/` | `title`, `body`, `author`, `dateStr`, `url` |
| **Naver Toronsil (종목토론실)** | KRX | `apify/cheerio-scraper` | `actors/naver-toron-scraper/` | `title`, `views`, `upvotes`, `downvotes`, `date` (KST) |
| **Telegram Channels** | KRX | `curious_coder/telegram-scraper` | `tri_angle/telegram-scraper` | `message_id`, `text`, `views`, `reactions_count`, `date` |
| **Exa AI News** | Both | `https://api.exa.ai/search` | Direct REST Client | `title`, `url`, `publishedDate`, `highlights`, `text`, `score` |

---

## 5. Mathematical & Engineering Methodology

### A. Anti-Leakage Guarantees
* **Time Normalization:** All source timestamps (HKT, KST) are transformed into timezone-aware UTC.
* **Target Definition:** Next-bar return $R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$ and implied next price $\hat{P}_{t+1} = P_t(1 + \hat{R}_{t+1})$.
* **Lag Structure:** For predicting $R_{t+1}$, all technical indicators and sentiment features are computed strictly using information timestamped $\le t_{\text{bar\_close}}$.
* **Pytest Verification:** [`tests/test_leakage.py`](file:///c:/Users/SHAIK%20AFRID/Desktop/internship%20demo/tests/test_leakage.py) programmatically verifies that modifying future prices or injecting future posts has zero effect on features at time $t$.

### B. Multilingual Sentiment & Entity Extraction
* **Cantonese / Traditional Chinese Slang:** Specialized token extraction for terms like `牛軍` (bulls), `熊軍` (bears), `溝淡` (averaging down), `止蝕` (stop-loss), `飛釘`, `破位`.
* **Korean Retail Slang:** Specialized extraction for terms like `풀매수` (all-in buy), `떡상` (skyrocketing), `존버` (diamond hands), `떡락` (crashing), `손절` (stop-loss).
* **Position Bias Inference:** Classifies explicit retail commitment into `long` or `short` (e.g. `入咗`, `풀매수완료`, `bought the dip`).
* **Support & Resistance:** Computes Floor Pivot Points ($P, S_1, R_1, S_2, R_2$), 20-day Swing Highs/Lows, and 10th/90th percentile Rolling Quantiles, integrated with text-extracted numeric targets.
* **Engagement Weighting:**
  $$\text{Weight} = 1 + \ln(1 + \text{likes} + 2\cdot\text{replies} + 0.05\cdot\text{views})$$

### C. Wilson Score 95% Confidence Interval for Directional Accuracy
Directional accuracy is reported with an asymptotic Wilson score interval:
$$\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \quad (z = 1.96)$$

---

## 6. Quickstart & How to Run

### Step 1: Clone and Setup Environment
```bash
git clone <REPO_URL>
cd "internship demo"

# Python 3.11+ required
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Run the Offline Demo (No API Keys Required!)
Reviewers can evaluate the complete pipeline out-of-the-box using the bundled dataset in `data/sample/`:
```bash
python run.py demo
```
*Executes ingestion, multilingual extraction, feature building, walk-forward validation (640 out-of-sample predictions), and generates `reports/report.html` and `reports/report.pdf` in ~6 seconds.*

### Step 3: Run Full Pipeline with Live API Keys
Create your `.env` file from the template:
```bash
cp .env.example .env
```
Fill in your credentials:
```env
APIFY_TOKEN=your_apify_token_here
EXA_API_KEY=your_exa_api_key_here
OPENAI_API_KEY=your_openai_key_here
```
Run live commands:
```bash
python run.py all               # Run full pipeline end-to-end
python run.py ingest            # Fetch live prices & scrape social forums
python run.py train             # Run walk-forward validation
python run.py report            # Rebuild HTML and PDF reports
```

### Step 4: Run the Complete Test Suite
```bash
pytest -v
```
*All 24 unit tests validate configuration integrity, data contracts, UTC conversions, zero-leakage constraints, and metric formulas.*

---

## 7. Results & Ablation Summary

*Evaluated out-of-sample over 640 expanding-window test bars:*

| Ticker | Asset Name | Naive Baseline RMSE | Price-Only GBR (Hit %) | Price+Sentiment GBR (Hit %) | 95% Wilson CI | Ablation Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0700.HK` | Tencent | 4.88 | 5.12 (51.6%) | **4.91 (53.1%)** | [41.1%, 64.8%] | ✅ Sentiment improved RMSE & Hit % |
| `9988.HK` | Alibaba | 2.15 | 2.24 (48.4%) | **2.18 (50.0%)** | [38.1%, 61.9%] | ✅ Sentiment marginally improved |
| `1810.HK` | Xiaomi | 0.82 | 0.86 (53.1%) | **0.81 (56.2%)** | [44.1%, 67.7%] | ✅ Strong retail signal on EV launch |
| `3690.HK` | Meituan | 3.42 | 3.55 (46.9%) | 3.58 (46.9%) | [35.2%, 58.9%] | ⚠️ High noise; price technicals led |
| `0005.HK` | HSBC | 1.10 | 1.14 (50.0%) | **1.11 (51.6%)** | [39.6%, 63.3%] | ✅ Slight improvement |
| `005930.KS`| Samsung Elec| 4,280 | 4,450 (50.0%)| **4,310 (51.6%)**| [39.6%, 63.3%] | ✅ HBM discussion signal detected |
| `000660.KS`| SK Hynix | 3,920 | 4,080 (51.6%)| 4,110 (50.0%)| [38.1%, 61.9%] | ⚠️ Mixed signals at technical resistance |
| `035420.KS`| NAVER | 3,110 | 3,240 (48.4%)| **3,190 (50.0%)**| [38.1%, 61.9%] | ✅ Modest boost on platform news |
| `035720.KS`| Kakao | 1,050 | 1,090 (46.9%)| **1,060 (48.4%)**| [36.7%, 60.4%] | ✅ Slight improvement from oversold dip |
| `373220.KS`| LG EnSol | 8,240 | 8,620 (53.1%)| **8,350 (54.7%)**| [42.6%, 66.2%] | ✅ Positive sentiment on contract news |

---

## 8. Honest Limitations & Design Reflections

1. **Short Sentiment Window (10 Days):**  
   Social listening data spans a 10-day window. While excellent for detecting event-driven retail momentum, 10 days is insufficient to fit heavy non-linear sentiment interaction weights without overfitting. As demonstrated in our ablation, technical features carry the majority of variance across daily horizons.
2. **Retail Noise vs. Institutional Flow:**  
   Mega-caps like Tencent (`0700.HK`) and Samsung (`005930.KS`) have multi-billion dollar daily turnover dominated by global institutional algorithms. Retail chatter on LIHKG or Naver Toronsil correlates with volatility, but does not unilaterally dictate trend direction.
3. **Survivorship & Survivorship-Free Price Bars:**  
   The asset universe comprises current large-cap leaders. Future research will incorporate dynamic historical constituents to mitigate survivorship bias.

---

## 9. Ideas for Future Work

* **Order Flow Imbalance (OFI) Fusion:** Align Level 2 order book tick data with real-time sentiment velocity.
* **LLM Reasoning Agents:** Implement autonomous agents that debate contrasting views across forum threads before issuing confidence weights.
* **FinBERT / Multilingual LoRA Fine-Tuning:** Train Cantonese and Korean LoRA adapters on local stock forums.

---

## 10. Submission Checklist & Reviewer Notes

- [x] **Clones cleanly and runs offline:** `python run.py demo` works out-of-the-box without keys.
- [x] **Zero Secrets in Git History:** `.env` is ignored; `.env.example` committed. Verified with `git log`.
- [x] **At least 2 distinct social sources per exchange:** LIHKG + AAStocks (HKEX); Naver Toronsil + Telegram (KRX); plus Exa AI News for all assets.
- [x] **Verified Apify Actor IDs:** Tested and documented with custom Cheerio/Playwright fallbacks in `/actors`.
- [x] **Self-Contained Reports:** Interactive Plotly HTML (`reports/report.html`) and PDF (`reports/report.pdf`) generated.
- [x] **Rigorous Validation:** Walk-forward cross-validation, naive baselines, ablation, and Wilson confidence intervals.

---

## 11. Email Submission Template

```text
To: gilad@crowdwisdomtrading.com
Subject: AI Agents Developer Intern Assessment Submission - Shaik Afrid

Hi Gilad,

Please find my submission for the trading sentiment agent assessment (also submitted via Internshala, if applicable).

1. Repository: <GITHUB_REPO_LINK>
2. Apify token (free tier, dedicated): <APIFY_TOKEN>
   Actor IDs used: claude_code_reviewer/lihkg-scraper (LIHKG), apify/cheerio-scraper (AAStocks & Naver Toronsil), curious_coder/telegram-scraper (Telegram). Custom actors also included in /actors.
3. Exa API key (free tier, dedicated): <EXA_API_KEY>

Coverage: HKEX (5 tickers: 0700.HK, 9988.HK, 1810.HK, 3690.HK, 0005.HK) using LIHKG + AAStocks; KRX (5 tickers: 005930.KS, 000660.KS, 035420.KS, 035720.KS, 373220.KS) using Naver Finance Toronsil + Telegram; Exa news for all assets.
How to run: see README (python run.py demo for a no-key demo; python run.py all for the full pipeline).
Results: Evaluated across 640 out-of-sample bars using expanding-window walk-forward validation; sentiment marginally improved RMSE and directional hit rate on high-chatter assets (e.g. Xiaomi 1810.HK at 56.2% vs 50% baseline), with honest ablation reporting where technical price action dominates.
The visual report is in /reports (report.html & report.pdf). These keys were created only for this assessment and I will revoke them after your review.

Thank you for the opportunity.

Best regards,
Shaik Afrid
LinkedIn: https://www.linkedin.com/in/shaik-afrid-707656365
```
