import datetime
import uuid
from typing import Optional
from uuid import UUID

from google.cloud import firestore

from midistral.db import schemas
from midistral.db.firestore.database import get_firestore_db


def create_abc_generation(
    abc_generation: schemas.AbcGenerationCreate,
) -> schemas.AbcGeneration:
    db: firestore.Client = get_firestore_db()
    doc_id = str(uuid.uuid4())
    doc_ref = db.collection("abc_generations").document(doc_id)
    val = {
        **abc_generation.model_dump(),
        "id": doc_id,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    print(val)
    doc_ref.set(val)
    return schemas.AbcGeneration.model_validate(doc_ref.get().to_dict())


def update_feedback_abc_generation(abc_generation_id: UUID, liked: bool):
    db: firestore.Client = get_firestore_db()
    doc_ref = db.collection("abc_generations").document(str(abc_generation_id))
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.update({"liked": liked})


def get_abc_generation(abc_generation_id: UUID) -> Optional[schemas.AbcGeneration]:
    db: firestore.Client = get_firestore_db()
    doc_ref = db.collection("abc_generations").document(str(abc_generation_id))
    doc = doc_ref.get()
    if doc.exists:
        return schemas.AbcGeneration.model_validate(doc.to_dict())
    else:
        return None
