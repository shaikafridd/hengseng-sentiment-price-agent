"""Multilingual financial sentiment lexicon, slang dictionary, and entity extractor.
Supports Traditional Chinese / Cantonese forum slang, Korean retail slang, and English financial jargon.
"""

import re
from typing import Dict, List, Optional, Set, Tuple


class MultilingualLexicon:
    """Rule-based multilingual sentiment and position intent classifier."""

    # Traditional Chinese / Cantonese Forum Terms (LIHKG, AAStocks, Futu)
    ZH_BULLISH = [
        "牛軍", "好倉", "升", "破位", "抄底", "溝淡", "大升", "牛證", "狂升",
        "目標價", "買入", "增持", "衝", "飛釘", "大陽燭", "上車", "爆升",
        "起飛", "好消息", "睇好", "追入", "加倉", "超預期", "強勁", "利好", "支持位"
    ]
    ZH_BEARISH = [
        "熊軍", "淡倉", "跌", "跌穿", "止蝕", "暴跌", "熊證", "減持", "沽",
        "割肉", "做淡", "走佬", "破產", "陰燭", "落車", "大跌", "看淡", "沽空",
        "踩雷", "跌停", "沽盤", "阻力位", "虧損", "弱勢", "利空", "爆煲"
    ]
    ZH_LONG_INTENT = [
        "入咗", "買咗", "加倉", "溝淡", "建倉", "開好倉", "上車", "追入", "持有",
        "坐貨", "買入", "掃貨", "買多兩手"
    ]
    ZH_SHORT_INTENT = [
        "沽咗", "止蝕咗", "平倉", "開淡倉", "清倉", "割咗", "落車", "走咗",
        "沽空", "止損", "止賺"
    ]

    # Korean Retail Terms (Naver Toronsil, Telegram, Daum)
    KR_BULLISH = [
        "풀매수", "떡상", "존버", "상승", "반등", "급등", "호재", "매수",
        "목표가", "슈퍼사이클", "바닥확인", "가즈아", "신고가", "순매수", "흑자전환",
        "수주", "실적서프라이즈", "추천", "기대감", "상방", "대박", "지지력"
    ]
    KR_BEARISH = [
        "떡락", "폭락", "하락", "손절", "악재", "매도", "하방", "지지선이탈",
        "물림", "탈출", "공매도", "신저가", "순매도", "적자", "어닝쇼크", "망했다",
        "비중축소", "붕괴", "위험", "털렸다", "지하실"
    ]
    KR_LONG_INTENT = [
        "풀매수했다", "매수완료", "탑승", "담았다", "존버중", "추매", "매수했음",
        "장기투자", "모아가는중"
    ]
    KR_SHORT_INTENT = [
        "손절했다", "익절", "전량매도", "탈출했다", "숏", "던졌다", "손절침",
        "매도완료", "손 뗐다"
    ]

    # English Financial Jargon
    EN_BULLISH = [
        "bullish", "moon", "calls", "breakout", "buy", "dip", "accumulate",
        "undervalued", "rally", "long", "outperform", "upgrade", "growth",
        "surge", "beat", "strong", "upside", "rebound"
    ]
    EN_BEARISH = [
        "bearish", "dump", "puts", "breakdown", "sell", "short", "overvalued",
        "crash", "stop loss", "downgrade", "weak", "miss", "plunge", "drop",
        "underperform", "downside", "correction", "risk"
    ]
    EN_LONG_INTENT = [
        "bought", "long", "added", "holding", "bought the dip", "accumulating",
        "entered long", "all in"
    ]
    EN_SHORT_INTENT = [
        "sold", "cut loss", "shorted", "stopped out", "cashed out", "exited",
        "taking profit", "short position"
    ]

    @classmethod
    def score_text(cls, text: str) -> Tuple[float, str]:
        """Compute sentiment score in range [-1.0, 1.0] and direction ('bullish', 'bearish', 'neutral')."""
        if not text:
            return 0.0, "neutral"

        text_lower = text.lower()

        bull_hits = 0
        bear_hits = 0

        # Chinese matches
        for word in cls.ZH_BULLISH:
            bull_hits += text.count(word)
        for word in cls.ZH_BEARISH:
            bear_hits += text.count(word)

        # Korean matches
        for word in cls.KR_BULLISH:
            bull_hits += text.count(word)
        for word in cls.KR_BEARISH:
            bear_hits += text.count(word)

        # English matches
        for word in cls.EN_BULLISH:
            bull_hits += text_lower.count(word)
        for word in cls.EN_BEARISH:
            bear_hits += text_lower.count(word)

        total_hits = bull_hits + bear_hits
        if total_hits == 0:
            return 0.0, "neutral"

        # Bounded score
        score = (bull_hits - bear_hits) / total_hits
        score = max(-1.0, min(1.0, score))

        if score > 0.15:
            direction = "bullish"
        elif score < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        return float(round(score, 3)), direction

    @classmethod
    def infer_position_bias(cls, text: str) -> Optional[str]:
        """Infer whether user explicitly expressed a Long or Short position intent."""
        text_lower = text.lower()

        long_count = sum(text.count(w) for w in cls.ZH_LONG_INTENT) + \
                     sum(text.count(w) for w in cls.KR_LONG_INTENT) + \
                     sum(text_lower.count(w) for w in cls.EN_LONG_INTENT)

        short_count = sum(text.count(w) for w in cls.ZH_SHORT_INTENT) + \
                      sum(text.count(w) for w in cls.KR_SHORT_INTENT) + \
                      sum(text_lower.count(w) for w in cls.EN_SHORT_INTENT)

        if long_count > short_count and long_count > 0:
            return "long"
        elif short_count > long_count and short_count > 0:
            return "short"
        return None

    @classmethod
    def extract_price_mentions(cls, text: str, reference_price: float) -> List[float]:
        """Extract plausible price targets / levels mentioned in social text near the reference price."""
        levels = []

        # 1. Matches for Korean 'X만원' or 'X만' (e.g. 20만 -> 200,000, 18만원 -> 180,000)
        kr_man_matches = re.findall(r"(\d+(?:\.\d+)?)\s*만\s*원?", text)
        for m in kr_man_matches:
            try:
                val = float(m) * 10000.0
                if 0.3 * reference_price <= val <= 3.0 * reference_price:
                    levels.append(round(val, 2))
            except ValueError:
                pass

        # 2. General numbers with commas or decimals (e.g. 180,000 or 420.5 or 85元)
        general_matches = re.findall(r"(?:[$¥₩HKD]?\s*)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:元|蚊|원|usd|hkd|krw)?", text, re.IGNORECASE)
        for m in general_matches:
            clean_str = m.replace(",", "")
            try:
                val = float(clean_str)
                # Filter out obvious dates, codes, or unrelated small numbers
                if 0.5 * reference_price <= val <= 2.0 * reference_price:
                    levels.append(round(val, 2))
            except ValueError:
                pass

        return sorted(list(set(levels)))
