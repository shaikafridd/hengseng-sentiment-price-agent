"""Test multilingual sentiment scoring, slang heuristics, position inference, and S/R extraction."""

from datetime import datetime, timezone
import pandas as pd
from src.config import config
from src.sentiment.contracts import RawPost, NewsArticle
from src.sentiment.heuristics import MultilingualLexicon
from src.sentiment.extractor import TickerIntelligenceExtractor


def test_cantonese_slang_scoring():
    """Verify Cantonese / Traditional Chinese financial slang classifications."""
    bull_text = "騰訊 0700 業績超預期，牛軍大勝，目標價 450 破位狂升！"
    score, direction = MultilingualLexicon.score_text(bull_text)
    assert score > 0.3
    assert direction == "bullish"

    bear_text = "恆指大跌，0700 跌穿支持，熊軍做淡，快啲止蝕割肉！"
    b_score, b_direction = MultilingualLexicon.score_text(bear_text)
    assert b_score < -0.3
    assert b_direction == "bearish"


def test_korean_slang_scoring():
    """Verify Korean retail investor slang classifications."""
    bull_text = "삼전 엔비디아 퀄테스트 통과 호재! 떡상 가즈아 풀매수 탑승!"
    score, direction = MultilingualLexicon.score_text(bull_text)
    assert score > 0.3
    assert direction == "bullish"

    bear_text = "외인 대량 순매도 악재 발생... 지지선 무너지고 떡락중 손절했다"
    b_score, b_direction = MultilingualLexicon.score_text(bear_text)
    assert b_score < -0.3
    assert b_direction == "bearish"


def test_position_bias_inference():
    """Verify intent extraction for long and short trade positions."""
    assert MultilingualLexicon.infer_position_bias("我已經入咗兩手騰訊長揸") == "long"
    assert MultilingualLexicon.infer_position_bias("今日全部止蝕咗清倉走人") == "short"
    assert MultilingualLexicon.infer_position_bias("삼성전자 500주 풀매수완료") == "long"
    assert MultilingualLexicon.infer_position_bias("너무 불안해서 전량매도 손절침") == "short"
    assert MultilingualLexicon.infer_position_bias("I bought the dip and added more calls") == "long"
    assert MultilingualLexicon.infer_position_bias("Stopped out and cut loss today") == "short"
    assert MultilingualLexicon.infer_position_bias("What do you think of this company?") is None


def test_technical_sr_levels_math():
    """Verify Pivot Points math satisfies S2 < S1 < Close < R1 < R2."""
    dates = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [100.0] * 20,
            "High": [110.0] * 20,
            "Low": [90.0] * 20,
            "Close": [100.0] * 20,
            "Volume": [100000] * 20,
        },
        index=dates
    )

    extractor = TickerIntelligenceExtractor()
    supports, resistances = extractor.compute_technical_levels(df)

    assert len(supports) > 0
    assert len(resistances) > 0
    assert all(s < 100.0 for s in supports)
    assert all(r > 100.0 for r in resistances)


def test_full_ticker_intelligence_extraction():
    """Test full extraction pipeline on an asset with posts and news."""
    asset = config.get_asset_by_ticker("0700.HK")
    assert asset is not None

    dates = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
    price_df = pd.DataFrame(
        {
            "Open": [415.0] * 20,
            "High": [430.0] * 20,
            "Low": [410.0] * 20,
            "Close": [420.0] * 20,
            "Volume": [1000000] * 20,
        },
        index=dates
    )

    posts = [
        RawPost(
            ticker="0700.HK",
            source="lihkg",
            post_id="p1",
            timestamp_utc=datetime.now(timezone.utc),
            text="騰訊 0700 業績勁，目標價 450，我入咗一手好倉",
            likes=50,
            views=1000
        ),
        RawPost(
            ticker="0700.HK",
            source="aastocks",
            post_id="p2",
            timestamp_utc=datetime.now(timezone.utc),
            text="420 支持位好穩，微信收入持續增長，買入",
            likes=20,
            views=400
        )
    ]

    news = [
        NewsArticle(
            ticker="0700.HK",
            title="Tencent Cloud introduces new AI models",
            url="https://news.com/tencent",
            published_date_utc=datetime.now(timezone.utc),
            text="Strong enterprise customer adoption."
        )
    ]

    extractor = TickerIntelligenceExtractor()
    intel = extractor.extract_ticker_intelligence(asset, posts, news, price_df)

    assert intel.ticker == "0700.HK"
    assert intel.current_price == 420.0
    assert intel.sentiment_direction == "bullish"
    assert intel.position_bias == "long"
    assert 450.0 in intel.text_extracted_price_levels or 420.0 in intel.text_extracted_price_levels
    assert len(intel.support_levels) > 0
    assert len(intel.resistance_levels) > 0
    assert intel.confidence_score > 0.0
