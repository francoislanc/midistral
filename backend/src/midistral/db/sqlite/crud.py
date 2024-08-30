from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from midistral.db import schemas
from midistral.db.sqlite import models
from midistral.db.sqlite.database import get_sqlite_db
from midistral.types import AudioTextDescription


def create_abc_generation(
    abc_generation: schemas.AbcGenerationCreate,
) -> schemas.AbcGeneration:

    db: Session = get_sqlite_db()
    db_abc_generation = models.AbcGeneration(**abc_generation.model_dump())
    db.add(db_abc_generation)
    db.commit()
    db.refresh(db_abc_generation)
    return db_abc_generation


def update_feedback_abc_generation(abc_generation_id: UUID, liked: bool):
    db: Session = get_sqlite_db()
    abc_generation = (
        db.query(models.AbcGeneration)
        .filter(models.AbcGeneration.id == abc_generation_id)
        .first()
    )
    if abc_generation:
        abc_generation.liked = liked
        db.add(abc_generation)
        db.commit()


def get_abc_generation(abc_generation_id: UUID) -> Optional[schemas.AbcGeneration]:
    db: Session = get_sqlite_db()
    return (
        db.query(models.AbcGeneration)
        .filter(models.AbcGeneration.id == abc_generation_id)
        .first()
    )


def create_annotated_abc(
    annotated_abc: schemas.AnnotatedAbcCreate,
) -> schemas.AnnotatedAbcBase:

    db: Session = get_sqlite_db()
    db_annotated_abc = models.AnnotatedAbc(**annotated_abc.model_dump())
    db.add(db_annotated_abc)
    db.commit()
    db.refresh(db_annotated_abc)
    return db_annotated_abc


# prepare function for filters
instruments_funcs = [
    func.json_each(models.AnnotatedAbc.description, "$.instruments").table_valued(
        "value", joins_implicitly=True
    )
    for _i in range(3)
]
mood_funcs = [
    func.json_each(models.AnnotatedAbc.description, "$.mood").table_valued(
        "value", joins_implicitly=True
    )
    for _i in range(3)
]
genre_funcs = [
    func.json_each(models.AnnotatedAbc.description, "$.genre").table_valued(
        "value", joins_implicitly=True
    )
    for _i in range(3)
]

funcs = {"instruments": instruments_funcs, "mood": mood_funcs, "genre": genre_funcs}


def _get_annotated_abcs_from_description(
    description: AudioTextDescription, limit: int
) -> List[schemas.AnnotatedAbcBase]:
    db: Session = get_sqlite_db()

    instrument_filter = [
        funcs["instruments"][i].c.value == description.instruments[i]
        for i in range(len(description.instruments))
        if i < len(funcs["instruments"])
    ]
    mood_filter = [
        funcs["mood"][i].c.value == description.mood[i]
        for i in range(len(description.mood))
        if i < len(funcs["mood"])
    ]
    genre_filter = [
        funcs["genre"][i].c.value == description.genre[i]
        for i in range(len(description.genre))
        if i < len(funcs["genre"])
    ]
    res = (
        db.query(models.AnnotatedAbc)
        .filter(or_(*instrument_filter), or_(*mood_filter), or_(*genre_filter))
        .limit(limit)
        .all()
    )
    return res


def get_annotated_abcs_from_description(
    description: AudioTextDescription,
    limit: int,
    with_constraints_relaxation: bool = True,
) -> List[schemas.AnnotatedAbcBase]:
    res = _get_annotated_abcs_from_description(description, limit)
    if with_constraints_relaxation:
        if len(res) == 0:
            # removing mood constraints
            description_without_mood = AudioTextDescription(
                instruments=description.instruments,
                genre=description.genre,
                mood=[],
                midi_instruments_num=None,
            )
            res = _get_annotated_abcs_from_description(description_without_mood, limit)
            if len(res) == 0:
                # removing genre constraints
                description_without_genre = AudioTextDescription(
                    instruments=description.instruments,
                    mood=description.mood,
                    genre=[],
                    midi_instruments_num=None,
                )
                res = _get_annotated_abcs_from_description(
                    description_without_genre, limit
                )

                if len(res) == 0:
                    # removing both mood and genre constraints
                    description_without_mood_without_genre = AudioTextDescription(
                        instruments=description.instruments,
                        mood=[],
                        genre=[],
                        midi_instruments_num=None,
                    )
                    res = _get_annotated_abcs_from_description(
                        description_without_mood_without_genre, limit
                    )
    return res
