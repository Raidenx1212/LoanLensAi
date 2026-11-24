import os
import sys
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine possible locations for .env
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
PROJECT_ROOT = BASE_DIR.parent                     # loanlens/

env_files = [
    PROJECT_ROOT / ".env",
    BASE_DIR / ".env",
    Path.cwd() / ".env",
    Path.cwd() / "backend" / ".env",
]

# Pick the first existing .env file
active_env_file = str(BASE_DIR / ".env")
for ef in env_files:
    if ef.is_file() and ef.stat().st_size > 0:
        active_env_file = str(ef)
        break


class Settings(BaseSettings):
    PROJECT_NAME: str = "LoanLens AI Auditor"
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.5-flash")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./loanlens.db"
    )

    model_config = SettingsConfigDict(
        env_file=active_env_file,
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
