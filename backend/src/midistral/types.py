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
    "ambient",
    "classical",
    "dance",
    "electronic",
    "experimental",
    "pop",
    "rock",
    "soundtrack",
    "techno",
]

MOOD_VALUES: List[str] = [
    "action",
    "corporate",
    "dark",
    "dream",
    "emotional",
    "energetic",
    "epic",
    "film",
    "happy",
    "inspiring",
    "love",
    "meditative",
    "melodic",
    "relaxing",
    "retro",
    "slow",
    "space",
]

INSTRUMENTS_VALUES: List[str] = [
    "acoustic guitar",
    "hammond organ",
    "piano",
    "synth bass",
    "synth lead",
    "vibraphone",
]
