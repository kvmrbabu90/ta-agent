"""Central configuration. Resolves project paths and loads environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root once. This module lives at packages/common/config.py,
# so the project root is two parents up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = DATA_DIR / "models"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure directories exist.
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Runtime settings, loaded from .env file or environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # IB credentials / connection
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # 7497 paper, 7496 live
    ib_client_id: int = 1

    # Kite credentials
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""

    # Storage
    duckdb_path: str = str(PROCESSED_DIR / "market.duckdb")

    # Logging
    log_level: str = "INFO"


settings = Settings()
