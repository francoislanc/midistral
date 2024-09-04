from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AudioTextDescription(BaseModel):
    genre: List[str]
    mood: List[str]
    instruments: List[str]
    midi_instruments_num: Optional[List[int]] = Field(default=None)

    def filter_values(self):
        self.genre = [g for g in self.genre if g in GENRE_VALUES]
        self.mood = [m for m in self.mood if m in MOOD_VALUES]
        self.instruments = [i for i in self.instruments if i in INSTRUMENTS_VALUES]


class ABCNotation(BaseModel):
    text: str


class InferenceApproach(str, Enum):
    PROMPT_ONLY = "PROMPT_ONLY"
    FINETUNED = "FINETUNED"
    RAG = "RAG"


GENRE_VALUES: List[str] = [
    "classical",
    "electronic",
    "pop",
    "soundtrack",
]

MOOD_VALUES: List[str] = ["positive", "energetic", "calm", "emotional", "film"]

INSTRUMENTS_VALUES: List[str] = [
    "acoustic guitar",
    "piano",
    "clarinet",
    "ocarina",
    "synth lead",
    "trombone",
    "trumpet",
]
