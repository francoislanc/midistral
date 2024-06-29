from typing import Union
from uuid import UUID

from pydantic import BaseModel


class AbcGenerationBase(BaseModel):
    text_description: str
    abc_notation: str
    file_uuid: str
    liked: Union[bool, None] = None


class AbcGenerationCreate(AbcGenerationBase):
    pass


class AbcGeneration(AbcGenerationBase):
    id: UUID

    class Config:
        from_attributes = True
