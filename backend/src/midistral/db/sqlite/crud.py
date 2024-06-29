from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from midistral.db import schemas
from midistral.db.sqlite import models
from midistral.db.sqlite.database import get_sqlite_db


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
