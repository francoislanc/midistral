import uuid

from sqlalchemy import UUID, Boolean, Column, DateTime, String, func

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
    liked = Column(Boolean, default=False)
