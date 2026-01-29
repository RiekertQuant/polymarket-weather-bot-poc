"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Trading mode
    trading_mode: Literal["paper", "live"] = Field(
        default="paper",
        description="Trading mode: paper (simulated) or live (not implemented)",
    )

    # Bet sizing
    min_bet_usd: float = Field(default=2.0, ge=0.01, le=100.0)
    max_bet_usd: float = Field(default=5.0, ge=0.01, le=100.0)
    max_trades_per_run: int = Field(default=10, ge=1, le=100)
    max_daily_risk_usd: float = Field(default=50.0, ge=1.0, le=1000.0)
    max_risk_per_city_usd: float = Field(default=20.0, ge=1.0, le=500.0)

    # Strategy filters
    price_min: float = Field(default=0.001, description="Minimum price to consider")
    price_max: float = Field(default=0.25, description="Maximum price for cheap shares")
    skip_price_min: float = Field(default=0.40, description="Skip 50/50 range lower bound")
    skip_price_max: float = Field(default=0.60, description="Skip 50/50 range upper bound")
    min_edge_absolute: float = Field(default=0.15, description="Minimum p_model required")
    min_edge_relative: float = Field(default=0.10, description="p_model must exceed price by this")

    # Weather forecast
    forecast_sigma: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Standard deviation for forecast uncertainty (Celsius)",
    )

    # Database
    db_path: Path = Field(default=Path("data/trading.db"))

    # Polymarket client
    polymarket_client: Literal["mock", "real"] = Field(default="mock")
    polymarket_api_url: str = Field(default="https://clob.polymarket.com")

    # Weather forecast source
    weather_source: Literal["open_meteo", "nws"] = Field(
        default="open_meteo",
        description="Weather forecast source: open_meteo or nws (NWS matches Polymarket resolution)",
    )

    # Correlated bets limit (max bets per city/date)
    max_bets_per_city_date: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum bets allowed per city/date to avoid correlated losses",
    )

    # ML calibrator
    use_ml_calibrator: bool = Field(
        default=False,
        description="Use ML calibrator for probability adjustment (experimental)",
    )

    # Enhanced probability engine features
    use_enhanced_engine: bool = Field(
        default=True,
        description="Use enhanced probability engine with all improvements",
    )
    enable_ensemble: bool = Field(
        default=True,
        description="Combine forecasts from multiple sources",
    )
    enable_bias_correction: bool = Field(
        default=True,
        description="Apply bias correction to forecasts",
    )
    enable_dynamic_sigma: bool = Field(
        default=True,
        description="Scale uncertainty by forecast horizon",
    )
    enable_regime_detection: bool = Field(
        default=True,
        description="Detect stable vs transitional weather patterns",
    )

    # Notifications
    discord_webhook_url: str = Field(
        default="", description="Discord webhook URL for alerts"
    )

    # Logging
    log_level: str = Field(default="INFO")

    # Cities to track
    cities: list[str] = Field(
        default=["New York City", "London", "Seoul"],
        description="Cities to monitor for weather markets",
    )


# Global settings instance
settings = Settings()
