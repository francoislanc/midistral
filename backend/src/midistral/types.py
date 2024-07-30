from typing import List

from pydantic import BaseModel


class AudioTextDescription(BaseModel):
    genre: List[str]
    mood: List[str]
    instruments: List[str]

    def filter_values(self):
        self.genre = [g for g in self.genre if g in GENRE_VALUES]
        self.mood = [m for m in self.mood if m in MOOD_VALUES]
        self.instruments = [i for i in self.instruments if i in INSTRUMENTS_VALUES]


class ABCNotation(BaseModel):
    text: str


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
