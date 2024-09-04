import uuid

from sqlalchemy import JSON, UUID, Boolean, Column, DateTime, String, func

from midistral.db.sqlite.database import Base


class AbcGeneration(Base):
    __tablename__ = "abcgenerations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    text_description = Column(String)
    abc_notation = Column(String)
    file_uuid = Column(String)
    approach = Column(String)
    liked = Column(Boolean, default=False)


class AnnotatedAbc(Base):
    __tablename__ = "annotatedabcs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    abc_notation = Column(String)
    description = Column(JSON)
