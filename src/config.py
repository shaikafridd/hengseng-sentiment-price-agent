"""Configuration management module using Pydantic Settings and YAML specs."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AssetSpec(BaseModel):
    ticker: str
    raw_code: str
    name: str
    name_local: str
    cashtags: List[str] = Field(default_factory=list)
    keywords_en: List[str] = Field(default_factory=list)
    keywords_local: List[str] = Field(default_factory=list)
    sector: str = "General"


class ExchangeSpec(BaseModel):
    name: str
    timezone: str
    currency: str
    trading_hours: Dict[str, str] = Field(default_factory=dict)
    assets: List[AssetSpec] = Field(default_factory=list)


class AppConfig(BaseSettings):
    """Application configuration and credentials loaded from environment and YAML."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys (Never hardcoded!)
    apify_token: Optional[str] = Field(default=None, alias="APIFY_TOKEN")
    exa_api_key: Optional[str] = Field(default=None, alias="EXA_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")

    # Pipeline settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cache_dir: str = Field(default="data/raw", alias="CACHE_DIR")
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")

    # Project directories
    base_dir: Path = Path(__file__).resolve().parent.parent
    config_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "config")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    sample_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "sample")
    raw_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw")
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "reports")
    results_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results")

    # Exchange and Actor configuration caches
    _exchanges: Optional[Dict[str, ExchangeSpec]] = None
    _actors: Optional[Dict[str, Any]] = None

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for p in [self.data_dir, self.sample_dir, self.raw_dir, self.reports_dir, self.results_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def get_exchanges(self) -> Dict[str, ExchangeSpec]:
        """Load and cache exchange and asset configurations."""
        if self._exchanges is None:
            assets_path = self.config_dir / "assets.yaml"
            if not assets_path.exists():
                raise FileNotFoundError(f"Assets configuration not found at {assets_path}")
            with open(assets_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._exchanges = {
                exch: ExchangeSpec(**spec)
                for exch, spec in data.get("exchanges", {}).items()
            }
        return self._exchanges

    def get_all_assets(self) -> List[AssetSpec]:
        """Return a flattened list of all configured assets across exchanges."""
        exchanges = self.get_exchanges()
        assets: List[AssetSpec] = []
        for exch in exchanges.values():
            assets.extend(exch.assets)
        return assets

    def get_asset_by_ticker(self, ticker: str) -> Optional[AssetSpec]:
        """Lookup an asset by its ticker symbol (e.g. '0700.HK')."""
        for asset in self.get_all_assets():
            if asset.ticker.lower() == ticker.lower():
                return asset
        return None

    def get_actors_config(self) -> Dict[str, Any]:
        """Load actors and scraper configurations."""
        if self._actors is None:
            actors_path = self.config_dir / "actors.yaml"
            if not actors_path.exists():
                raise FileNotFoundError(f"Actors configuration not found at {actors_path}")
            with open(actors_path, "r", encoding="utf-8") as f:
                self._actors = yaml.safe_load(f).get("sources", {})
        return self._actors


# Global configuration singleton
config = AppConfig()
config.ensure_directories()
