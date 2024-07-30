from typing import Union
from uuid import UUID

from pydantic import BaseModel

from midistral.types import AudioTextDescription


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


class AnnotatedAbcBase(BaseModel):
    abc_notation: str
    description: AudioTextDescription


class AnnotatedAbcCreate(AnnotatedAbcBase):
    pass


class AnnotatedAbc(AnnotatedAbcBase):
    id: UUID

    class Config:
        from_attributes = True
