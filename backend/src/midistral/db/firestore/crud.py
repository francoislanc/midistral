import datetime
import uuid
from typing import List, Optional
from uuid import UUID

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import BaseCompositeFilter, FieldFilter

from midistral.db import schemas
from midistral.db.firestore.database import get_firestore_db
from midistral.db.firestore.utils import reformat_key
from midistral.types import AudioTextDescription


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


def create_annotated_abc(
    annotated_abc: schemas.AnnotatedAbcCreate,
) -> schemas.AnnotatedAbc:

    db: firestore.Client = get_firestore_db()
    doc_id = str(uuid.uuid4())
    doc_ref = db.collection("annotated_abcs").document(doc_id)

    # need to use a map structure instead of array because we cannot currently have multiple arrayContains conditions in a single query
    # https://stackoverflow.com/questions/54987399/firestore-search-array-contains-for-multiple-values

    val = {**annotated_abc.model_dump(), "id": doc_id}
    val["description"] = {
        k: {reformat_key(ev, True): True for ev in v}
        for k, v in val["description"].items()
    }
    doc_ref.set(val)
    annotated_abc_firestore = doc_ref.get().to_dict()
    annotated_abc_firestore["description"] = {
        k: [reformat_key(ek, False) for ek, ev in v.items()]
        for k, v in annotated_abc_firestore["description"].items()
    }
    return schemas.AnnotatedAbc.model_validate(annotated_abc_firestore)


def _get_annotated_abcs_from_description(
    description: AudioTextDescription, limit: int
) -> List[schemas.AnnotatedAbc]:
    db: firestore.Client = get_firestore_db()
    query = db.collection("annotated_abcs")

    if len(description.instruments) > 0:
        query = query.where(
            filter=(
                BaseCompositeFilter(
                    "OR",
                    [
                        FieldFilter(
                            f"description.instruments.{reformat_key(i, True)}",
                            "==",
                            True,
                        )
                        for i in description.instruments
                    ],
                )
            )
        )
    if len(description.mood) > 0:
        query = query.where(
            filter=(
                BaseCompositeFilter(
                    "OR",
                    [
                        FieldFilter(
                            f"description.mood.{reformat_key(i, True)}", "==", True
                        )
                        for i in description.mood
                    ],
                )
            )
        )

    if len(description.genre) > 0:
        query = query.where(
            filter=(
                BaseCompositeFilter(
                    "OR",
                    [
                        FieldFilter(
                            f"description.genre.{reformat_key(i, True)}", "==", True
                        )
                        for i in description.genre
                    ],
                )
            )
        )
    query = query.limit(limit)

    res = []
    for d in query.stream():
        reformatted_d = d.to_dict()
        reformatted_d["description"] = {
            k: [reformat_key(ek, False) for ek, ev in v.items()]
            for k, v in reformatted_d["description"].items()
        }
        res.append(schemas.AnnotatedAbc.model_validate(reformatted_d))

    return res


def get_annotated_abcs_from_description(
    description: AudioTextDescription,
    limit: int,
    with_constraints_relaxation: bool = True,
) -> List[schemas.AnnotatedAbc]:
    res = _get_annotated_abcs_from_description(description, limit)
    if with_constraints_relaxation:
        if len(res) == 0:
            # removing mood constraints
            description_without_mood = AudioTextDescription(
                instruments=description.instruments, genre=description.genre, mood=[]
            )
            res = _get_annotated_abcs_from_description(description_without_mood, limit)
            if len(res) == 0:
                # removing genre constraints
                description_without_genre = AudioTextDescription(
                    instruments=description.instruments, mood=description.mood, genre=[]
                )
                res = _get_annotated_abcs_from_description(
                    description_without_genre, limit
                )

                if len(res) == 0:
                    # removing both mood and genre constraints
                    description_without_mood_without_genre = AudioTextDescription(
                        instruments=description.instruments, mood=[], genre=[]
                    )
                    res = _get_annotated_abcs_from_description(
                        description_without_mood_without_genre, limit
                    )
    return res
