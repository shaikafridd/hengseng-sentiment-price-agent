"""Generates bundled realistic sample dataset for offline --demo evaluation mode."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import config, AssetSpec
from src.sentiment.contracts import RawPost, NewsArticle

logger = logging.getLogger(__name__)


def generate_sample_dataset(output_dir: Path = config.sample_dir) -> None:
    """Creates synthetic yet realistic historical price, social discussion, and news data for all 10 assets."""
    prices_dir = output_dir / "prices"
    posts_dir = output_dir / "posts"
    news_dir = output_dir / "news"

    for d in [prices_dir, posts_dir, news_dir]:
        d.mkdir(parents=True, exist_ok=True)

    assets = config.get_all_assets()
    now_utc = datetime.now(timezone.utc)

    # Base price pegs for realistic simulations
    price_pegs = {
        "0700.HK": 420.0,
        "9988.HK": 85.0,
        "1810.HK": 28.0,
        "3690.HK": 150.0,
        "0005.HK": 72.0,
        "005930.KS": 195000.0,
        "000660.KS": 185000.0,
        "035420.KS": 178000.0,
        "035720.KS": 39500.0,
        "373220.KS": 320000.0,
    }

    # Social sample texts per asset (Bilingual Chinese / Korean / English)
    sample_social_corpus = {
        "0700.HK": [
            ("lihkg", "騰訊 (0700) 業績超預期，微信生態 AI 變現能力強，目標價睇 450！支持牛軍！", 45, 3, 12, 1420),
            ("lihkg", "0700 跌穿 415 止蝕，恆指大市回調，短期唔好盲目抄底，轉看淡", 12, 28, 9, 890),
            ("aastocks", "騰訊控股手遊流水回暖，海外業務增長強勁，逢低吸納 420 支持位", 8, 1, 3, 450),
            ("lihkg", "企鵝 0700 準備破位，大行摩通升目標價至 460 元，強烈看好", 60, 5, 20, 2100),
            ("aastocks", "北水持續淨流入騰訊，長線資金買入信號明確，持股信心大增", 15, 2, 5, 670),
        ],
        "9988.HK": [
            ("lihkg", "阿里巴巴 (9988) 雲端業務分拆與 AI 晶片佈局，現價 85 估值極具吸引力", 38, 4, 11, 1150),
            ("lihkg", "淘寶雙11成交量平淡，電商競爭激烈，阿里短期料在 80-88 區間震盪", 18, 12, 7, 780),
            ("aastocks", "巴巴回購力度加大，股息率有保障，長期價值投資者分批建倉", 22, 2, 6, 520),
            ("lihkg", "9988 阻力位 90 難破，逢高減倉鎖定利潤", 14, 19, 8, 640),
            ("aastocks", "阿里雲攜手開源社區，國際市場擴張提速，評級優於大市", 10, 1, 2, 390),
        ],
        "1810.HK": [
            ("lihkg", "小米 (1810) SU7 Ultra 訂單爆棚，雷軍真係神，小米汽車估值要重估！沖 32！", 95, 6, 35, 3400),
            ("lihkg", "小米 1810 升得太急，RSI 嚴重超買，短期 29 阻力大，回調先買", 25, 31, 15, 1200),
            ("aastocks", "小米手機高端化策略成功，歐洲份額穩步上升，買入信號", 19, 3, 7, 610),
            ("lihkg", "雷總發佈會太燃了，生態鏈無敵，堅持長揸小米！", 52, 4, 18, 1800),
            ("aastocks", "小米汽車毛利率轉正預期升溫，機構上調目標價至 30 元", 16, 2, 4, 480),
        ],
        "3690.HK": [
            ("lihkg", "美團 (3690) 外賣業務穩健，KeeTa 在香港拓展順利，現價 150 穩守支持", 34, 5, 8, 980),
            ("lihkg", "美團面臨抖音本地生活競爭加劇，到店毛利率承壓，偏淡操作", 22, 14, 10, 830),
            ("aastocks", "美團閃購與即時零售增長迅猛，客單價提升，維持買入", 12, 1, 3, 390),
            ("lihkg", "美團 145 跌唔落，雙底成型，目標反彈 165", 41, 7, 14, 1340),
            ("aastocks", "南向資金持續增持美團，成交活躍度高居榜首", 17, 3, 5, 510),
        ],
        "0005.HK": [
            ("lihkg", "滙豐 (0005) 宣佈架構重組，簡化營運節省成本，大笨象起飛！", 42, 4, 12, 1290),
            ("aastocks", "滙控高息回購支撐股價，防守力極強，退休收息首選 70 元支持", 28, 2, 6, 850),
            ("lihkg", "減息週期降臨，銀行淨息差 NIM 面臨收窄壓力，滙豐短期欠缺向上動力", 15, 18, 7, 720),
            ("aastocks", "滙豐第三季財報勝預期，宣佈新一輪 30 億美元股份回購", 33, 1, 9, 990),
            ("lihkg", "大笨象 0005 穩如泰山，恆指跌佢都唔跌，防守神股", 39, 3, 10, 1100),
        ],
        "005930.KS": [
            ("naver_finance", "삼성전자 (005930) HBM3E 엔비디아 퀄테스트 통과 기대감 고조, 20만전자 가자! 풀매수", 120, 15, 45, 4500),
            ("naver_finance", "삼전 외인 매도세 지속, 18만원 지지선 깨지면 17만원대까지 열려있음 손절 고려", 40, 85, 30, 2800),
            ("telegram", "여의도 찌라시: 삼성전자 평택 라인 가동률 상향 및 파운드리 대형 고객사 수주 임박", 65, 5, 12, 1900),
            ("naver_finance", "외국인 기관 쌍끌이 순매수 전환! 삼전 바닥 확인했다 이제 반등뿐", 88, 12, 28, 3200),
            ("telegram", "삼성전자 기술 경쟁력 회복 본격화, 목표주가 230,000원 유지", 55, 4, 10, 1600),
        ],
        "000660.KS": [
            ("naver_finance", "SK하이닉스 (000660) HBM 독점 공급력 유지, 분기 영업이익 사상 최대 돌파 확실", 110, 8, 38, 4100),
            ("naver_finance", "하닉 19만원 저항선 맞고 조정 중, 18만원 지지 테스트 확인 후 매수 추천", 35, 20, 16, 1950),
            ("telegram", "SK하이닉스 엔비디아 차세대 블랙웰 향 HBM3E 12단 출하 개시, 실적 서프라이즈 지속", 92, 3, 21, 2800),
            ("naver_finance", "반도체 슈퍼사이클 탑픽은 단연 하이닉스! 존버가 승리한다", 74, 10, 24, 2500),
            ("telegram", "골드만삭스, SK하이닉스 목표주가 220,000원으로 상향 조정", 61, 2, 8, 1750),
        ],
        "035420.KS": [
            ("naver_finance", "네이버 (035420) 서치플랫폼 AI 큐 브리핑 성과 가시화, 커머스 수수료 개편으로 수익성 개선", 55, 6, 18, 1850),
            ("naver_finance", "라인야후 관련 불확실성 해소 국면, 17만원 완벽한 바닥 매수 기회", 48, 8, 14, 1620),
            ("telegram", "네이버 사우디 디지털트윈 수주 본격 매출 반영 시작, 해외 모멘텀 주목", 67, 4, 11, 2100),
            ("naver_finance", "인터넷 플랫폼 성장 둔화 우려 여전, 18만5천원 매물대 돌파가 관건", 21, 15, 8, 980),
            ("telegram", "네이버 웹툰 미국 상장 이후 글로벌 IP 확장 가속화, 밸류에이션 리레이팅 기대", 43, 3, 7, 1400),
        ],
        "035720.KS": [
            ("naver_finance", "카카오 (035720) 카카오톡 개편 효과와 AI 에이전트 카나나 기대감으로 주가 반등 시도", 62, 9, 22, 2300),
            ("naver_finance", "사법 리스크 여진 남아있으나 38,000원 강력 지지선 형성 중, 저평가 구간", 38, 14, 15, 1450),
            ("telegram", "카카오 그룹 경영 쇄신 가시화, 비핵심 자산 매각으로 재무구조 대폭 개선", 45, 5, 9, 1300),
            ("naver_finance", "카카오페이 카카오뱅크 실적 호조 연결 반영, 바닥 탈출 신호", 51, 8, 16, 1700),
            ("telegram", "카카오 AI 서비스 상용화 본격화 전망, 중장기 매수 관점", 39, 4, 6, 1150),
        ],
        "373220.KS": [
            ("naver_finance", "LG에너지솔루션 (373220) 미국 IRA 보조금 수혜 확대 및 벤츠 등 대규모 수주 릴레이", 85, 11, 27, 3100),
            ("naver_finance", "전기차 캐즘 장기화로 가동률 부담 있으나 31만원 바닥 지지력 탄탄", 31, 22, 14, 1500),
            ("telegram", "LG엔솔 4680 배터리 양산 가시권, 테슬라 공급 본격화 기대감", 78, 6, 19, 2400),
            ("naver_finance", "2차전지 섹터 저점 통과 신호 포착, 엔솔 주도 반등세 시작", 69, 10, 20, 2200),
            ("telegram", "글로벌 OEM과의 ESS 배터리 공급 계약 확대, 사업 포트폴리오 다변화 긍정적", 52, 4, 10, 1650),
        ],
    }

    # Generate synthetic OHLCV price histories (daily for 180 days, hourly for 15 days)
    np.random.seed(42)
    for asset in assets:
        clean_ticker = asset.ticker.replace(".", "_")
        base_price = price_pegs.get(asset.ticker, 100.0)

        # 1. Daily bars (180 days)
        daily_dates = pd.date_range(end=now_utc, periods=180, freq="B", tz="UTC")
        daily_returns = np.random.normal(0.0005, 0.018, len(daily_dates))
        daily_prices = base_price * np.exp(np.cumsum(daily_returns))

        daily_df = pd.DataFrame(
            {
                "Open": daily_prices * (1 - np.random.uniform(0.002, 0.008, len(daily_dates))),
                "High": daily_prices * (1 + np.random.uniform(0.005, 0.015, len(daily_dates))),
                "Low": daily_prices * (1 - np.random.uniform(0.005, 0.015, len(daily_dates))),
                "Close": daily_prices,
                "Volume": np.random.randint(5000000, 50000000, len(daily_dates)),
            },
            index=daily_dates,
        )
        daily_df.index.name = "timestamp_utc"
        daily_df.to_parquet(prices_dir / f"{clean_ticker}_1d.parquet")

        # 2. Hourly bars (15 days * 7 hours)
        hourly_dates = pd.date_range(end=now_utc, periods=105, freq="h", tz="UTC")
        hourly_returns = np.random.normal(0.0001, 0.006, len(hourly_dates))
        hourly_prices = daily_prices[-1] * np.exp(np.cumsum(hourly_returns))

        hourly_df = pd.DataFrame(
            {
                "Open": hourly_prices * (1 - np.random.uniform(0.001, 0.004, len(hourly_dates))),
                "High": hourly_prices * (1 + np.random.uniform(0.002, 0.008, len(hourly_dates))),
                "Low": hourly_prices * (1 - np.random.uniform(0.002, 0.008, len(hourly_dates))),
                "Close": hourly_prices,
                "Volume": np.random.randint(200000, 3000000, len(hourly_dates)),
            },
            index=hourly_dates,
        )
        hourly_df.index.name = "timestamp_utc"
        hourly_df.to_parquet(prices_dir / f"{clean_ticker}_1h.parquet")

        # 3. Social posts
        posts: List[RawPost] = []
        corpus = sample_social_corpus.get(asset.ticker, [])
        for idx, (source, text, likes, dislikes, replies, views) in enumerate(corpus):
            # Spread timestamps across the last 10 days
            post_time = now_utc - timedelta(days=np.random.uniform(0.1, 9.5))
            posts.append(
                RawPost(
                    ticker=asset.ticker,
                    source=source,  # type: ignore
                    post_id=f"sample_{clean_ticker}_{idx}",
                    timestamp_utc=post_time,
                    text=text,
                    url=f"https://sample-discussion.com/{clean_ticker}/{idx}",
                    author=f"Trader_{np.random.randint(100, 999)}",
                    likes=likes,
                    dislikes=dislikes,
                    replies=replies,
                    views=views,
                    raw_metadata={"is_sample": True},
                )
            )

        post_file = posts_dir / f"{clean_ticker}_posts.json"
        with open(post_file, "w", encoding="utf-8") as f:
            json.dump([p.model_dump(mode="json") for p in posts], f, ensure_ascii=False, indent=2)

        # 4. News Articles
        news_articles = [
            NewsArticle(
                ticker=asset.ticker,
                source="exa",
                title=f"{asset.name} Quarterly Performance and Industry Outlook",
                url=f"https://financial-news.com/{clean_ticker}/q3-earnings",
                published_date_utc=now_utc - timedelta(days=2),
                text=f"{asset.name} ({asset.ticker}) reports strong operational progress, resilient margins, and continuous investment in technological innovation amidst competitive market dynamics.",
                highlights=[
                    f"{asset.name} maintained resilient revenue growth across key strategic business units.",
                    "Analyst sentiment remains constructive with focus on cash flow and market share gains."
                ],
                score=0.88
            ),
            NewsArticle(
                ticker=asset.ticker,
                source="exa",
                title=f"Market Commentary: Institutional Positioning in {asset.name}",
                url=f"https://institutional-equity.com/{clean_ticker}/analyst-note",
                published_date_utc=now_utc - timedelta(days=5),
                text=f"Institutional asset managers reiterate positive medium-term view on {asset.name}, citing attractive valuation multiples and solid balance sheet fundamentals.",
                highlights=[
                    "Cross-border capital inflows into major index components remained stable.",
                    "Support levels held firm during recent equity market consolidation."
                ],
                score=0.82
            )
        ]

        news_file = news_dir / f"{clean_ticker}_news.json"
        with open(news_file, "w", encoding="utf-8") as f:
            json.dump([n.model_dump(mode="json") for n in news_articles], f, ensure_ascii=False, indent=2)

    logger.info("Successfully generated bundled sample data for all 10 assets in %s", output_dir)


if __name__ == "__main__":
    generate_sample_dataset()
