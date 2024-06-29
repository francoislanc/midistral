from functools import lru_cache
from pathlib import Path

from google.cloud import storage

from midistral.config import get_settings


@lru_cache
def get_gcs_client():
    return storage.Client(project=get_settings().GCP_PROJECT)


def upload_file(fp: Path):
    bucket = get_gcs_client().get_bucket(get_settings().GCS_BUCKET)
    blob = bucket.blob(fp.name)
    blob.upload_from_filename(fp)


def download_file(fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    bucket = get_gcs_client().get_bucket(get_settings().GCS_BUCKET)
    blob = bucket.blob(fp.name)
    blob.download_to_filename(fp)
