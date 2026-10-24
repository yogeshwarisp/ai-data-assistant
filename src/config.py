"""Central configuration, loaded from environment variables / a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    max_upload_mb: int
    max_rows: int
    max_result_rows: int
    query_timeout_seconds: int

    @property
    def ai_enabled(self) -> bool:
        """AI mode needs both an API key and a model name."""
        return bool(self.openai_api_key and self.openai_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "").strip(),
        max_upload_mb=_int_env("MAX_UPLOAD_MB", 20),
        max_rows=_int_env("MAX_ROWS", 500_000),
        max_result_rows=_int_env("MAX_RESULT_ROWS", 200),
        query_timeout_seconds=_int_env("QUERY_TIMEOUT_SECONDS", 10),
    )
