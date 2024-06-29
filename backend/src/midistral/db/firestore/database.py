from functools import lru_cache

from google.cloud import firestore

from midistral.config import get_settings


@lru_cache
def get_firestore_db():
    return firestore.Client(project=get_settings().GCP_PROJECT)
