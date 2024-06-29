from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MISTRAL_API_KEY: str
    WANDB_API_KEY: Optional[str] = None
    WANDB_PROJECT: Optional[str] = None
    LLM_BASE_MODEL_FOR_FINETUNING: str
    LEARNING_RATE: float
    LLM_TEMPERATURE: float
    LLM_MAX_TOKEN: int
    FINETUNED_MODEL_NAME: str
    FRONT_END_ORIGIN: str

    DB_PATH: str = "./sql_app.db"
    USE_FIRESTORE_DB: bool = False
    GCP_PROJECT: Optional[str] = None
    GCS_BUCKET: Optional[str] = None

    APP_VERSION: str = "24.05.1"

    model_config = SettingsConfigDict(env_file="local.env")


@lru_cache
def get_settings():
    return Settings()
