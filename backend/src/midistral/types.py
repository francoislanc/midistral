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
    FINETUNED_1 = "FINETUNED_1"
    FINETUNED_2 = "FINETUNED_2"
    RAG = "RAG"


GENRE_VALUES: List[str] = [
    "electronic",
    "classical",
    "soundtrack",
    "pop",
    "experimental",
    "ambient",
]

MOOD_VALUES: List[str] = [
    "dark",
    "melodic",
    "film",
    "energetic",
    "happy",
    "relaxing",
    "emotional",
    "slow",
    "epic",
]

INSTRUMENTS_VALUES: List[str] = [
    "piano",
    "hammond organ",
    "synth lead",
    "vibraphone",
    "clavinet",
    "acoustic guitar",
    "clarinet",
    "bassoon",
    "trumpet",
    "synth bass",
    "harmonica",
    "ocarina",
    "flute",
    "violin",
]
