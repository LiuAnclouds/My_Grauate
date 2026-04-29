from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


SYSTEM_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = SYSTEM_ROOT / "backend"
DATA_ROOT = BACKEND_ROOT / "data"
UPLOAD_ROOT = DATA_ROOT / "uploads"


class Settings(BaseSettings):
    app_name: str = "DyRIFT Fraud Detection System"
    api_prefix: str = "/api"
    database_url: str = f"sqlite:///{(DATA_ROOT / 'dyrift_system.db').as_posix()}"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False

    verification_code_ttl_minutes: int = 10
    demo_mode: bool = True
    inference_device: str = "cpu"
    inference_timeout_seconds: int = 900
    demo_admin_email: str = "root"
    demo_admin_password: str = "root"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_prefix="DYRIFT_SYSTEM_",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


settings = Settings()


def ensure_runtime_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
