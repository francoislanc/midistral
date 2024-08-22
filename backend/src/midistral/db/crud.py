from midistral.config import get_settings
from midistral.db.firestore import crud as firestore_crud
from midistral.db.sqlite import crud as sqlite_crud


def get_crud():
    if get_settings().USE_FIRESTORE_DB:
        return firestore_crud
    else:
        return sqlite_crud
