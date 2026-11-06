"""Test configuration loading, validation, and asset metadata integrity."""

from src.config import config


def test_asset_universe_counts():
    """Verify that both HKEX and KRX each have exactly 5 liquid tickers configured."""
    exchanges = config.get_exchanges()
    assert "HKEX" in exchanges, "HKEX exchange definition missing"
    assert "KRX" in exchanges, "KRX exchange definition missing"

    hk_assets = exchanges["HKEX"].assets
    kr_assets = exchanges["KRX"].assets

    assert len(hk_assets) == 5, f"Expected 5 HKEX assets, got {len(hk_assets)}"
    assert len(kr_assets) == 5, f"Expected 5 KRX assets, got {len(kr_assets)}"

    all_assets = config.get_all_assets()
    assert len(all_assets) == 10, f"Expected 10 total assets, got {len(all_assets)}"


def test_asset_attributes_completeness():
    """Verify that every asset has ticker, raw_code, names, keywords, and cashtags."""
    for asset in config.get_all_assets():
        assert asset.ticker, "Ticker must not be empty"
        assert asset.raw_code, "Raw code must not be empty"
        assert asset.name, "Asset English name must not be empty"
        assert asset.name_local, "Asset local name must not be empty"
        assert len(asset.keywords_en) > 0, f"Keywords EN missing for {asset.ticker}"
        assert len(asset.keywords_local) > 0, f"Keywords local missing for {asset.ticker}"
        assert len(asset.cashtags) > 0, f"Cashtags missing for {asset.ticker}"


def test_actors_config():
    """Verify that all scrapers and Exa news endpoints are defined."""
    actors = config.get_actors_config()
    required_keys = ["hkex_social_1", "hkex_social_2", "krx_social_1", "krx_social_2", "news_source"]
    for key in required_keys:
        assert key in actors, f"Missing actor configuration: {key}"
