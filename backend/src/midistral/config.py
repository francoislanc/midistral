from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

MIDISTRAL_FOLDER = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    MISTRAL_API_KEY: str
    WANDB_API_KEY: Optional[str] = None
    WANDB_PROJECT: Optional[str] = None
    LLM_BASE_MODEL_FOR_FINETUNING: str
    LEARNING_RATE: float
    LLM_TEMPERATURE: float
    LLM_MAX_TOKEN: int
    PROMPT_MODEL_NAME: str
    FINETUNED_MODEL_NAME_1: str
    FINETUNED_MODEL_NAME_2: str
    RAG_MODEL_NAME: str
    FRONT_END_ORIGIN: str

    DB_PATH: str = str(MIDISTRAL_FOLDER / "sql_app.db")
    USE_FIRESTORE_DB: bool = False
    WITH_RAG: bool = True
    DB_LIMIT: int = 5
    RETRIEVED_LIMIT: int = 2
    GCP_PROJECT: Optional[str] = None
    GCS_BUCKET: Optional[str] = None

    APP_VERSION: str = "31.08.1"

    model_config = SettingsConfigDict(env_file=str(MIDISTRAL_FOLDER / "local.env"))


@lru_cache
def get_settings():
    return Settings()
