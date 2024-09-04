import json
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List

import essentia
import numpy as np
from chord_extractor.extractors import Chordino
from essentia.standard import (
    MonoLoader,
    TensorflowPredict2D,
    TensorflowPredictEffnetDiscogs,
)

essentia.log.warningActive = False
essentia.log.infoActive = False

MODEL_FOLDER = Path(__file__).resolve().parent.parent.parent / "models"


genre_classes = [
    "60s",
    "70s",
    "80s",
    "90s",
    "acidjazz",
    "alternative",
    "alternativerock",
    "ambient",
    "atmospheric",
    "blues",
    "bluesrock",
    "bossanova",
    "breakbeat",
    "celtic",
    "chanson",
    "chillout",
    "choir",
    "classical",
    "classicrock",
    "club",
    "contemporary",
    "country",
    "dance",
    "darkambient",
    "darkwave",
    "deephouse",
    "disco",
    "downtempo",
    "drumnbass",
    "dub",
    "dubstep",
    "easylistening",
    "edm",
    "electronic",
    "electronica",
    "electropop",
    "ethno",
    "eurodance",
    "experimental",
    "folk",
    "funk",
    "fusion",
    "groove",
    "grunge",
    "hard",
    "hardrock",
    "hiphop",
    "house",
    "idm",
    "improvisation",
    "indie",
    "industrial",
    "instrumentalpop",
    "instrumentalrock",
    "jazz",
    "jazzfusion",
    "latin",
    "lounge",
    "medieval",
    "metal",
    "minimal",
    "newage",
    "newwave",
    "orchestral",
    "pop",
    "popfolk",
    "poprock",
    "postrock",
    "progressive",
    "psychedelic",
    "punkrock",
    "rap",
    "reggae",
    "rnb",
    "rock",
    "rocknroll",
    "singersongwriter",
    "soul",
    "soundtrack",
    "swing",
    "symphonic",
    "synthpop",
    "techno",
    "trance",
    "triphop",
    "world",
    "worldfusion",
]

decades = ["60s", "70s", "80s", "90s"]
jazz_related = ["acidjazz", "jazz", "jazzfusion", "swing"]
rock_related = [
    "alternative",
    "alternativerock",
    "classicrock",
    "grunge",
    "hardrock",
    "metal",
    "poprock",
    "postrock",
    "progressive",
    "psychedelic",
    "punkrock",
    "rock",
    "rocknroll",
]
electronic_dance = [
    "ambient",
    "atmospheric",
    "breakbeat",
    "chillout",
    "club",
    "dance",
    "darkambient",
    "darkwave",
    "deephouse",
    "disco",
    "downtempo",
    "drumnbass",
    "dub",
    "dubstep",
    "edm",
    "electronic",
    "electronica",
    "electropop",
    "eurodance",
    "experimental",
    "groove",
    "hard",
    "house",
    "idm",
    "minimal",
    "newwave",
    "techno",
    "trance",
    "triphop",
]
folk_world = ["celtic", "ethno", "folk", "latin", "world", "worldfusion"]
classical_instrumental = [
    "choir",
    "classical",
    "instrumentalpop",
    "instrumentalrock",
    "medieval",
    "orchestral",
    "symphonic",
]
pop_related = ["pop", "popfolk", "singersongwriter", "synthpop"]
blues_related = ["blues", "bluesrock"]
hiphop_related = ["hiphop", "rap", "rnb"]
other_genres = [
    "bossanova",
    "chanson",
    "contemporary",
    "country",
    "easylistening",
    "funk",
    "fusion",
    "improvisation",
    "indie",
    "industrial",
    "lounge",
    "newage",
    "reggae",
    "soul",
]

SIMPLIFIED_GENRES = [
    # "rock",
    "electronic",
    "classical",
    # "folk",
    "pop",
    "blues",
    # "hiphop",
    "soundtrack",
]


def get_simplified_genres(genres: List[str]) -> List[str]:
    simplified_genres = set()
    for g in genres:
        if g in electronic_dance:
            simplified_genres.add("electronic")
        elif g in classical_instrumental:
            simplified_genres.add("classical")
        elif g in pop_related:
            simplified_genres.add("pop")
        elif g in blues_related:
            simplified_genres.add("blues")
        elif g == "soundtrack":
            simplified_genres.add("soundtrack")
    return list(simplified_genres)


mood_classes = [
    "action",
    "adventure",
    "advertising",
    "background",
    "ballad",
    "calm",
    "children",
    "christmas",
    "commercial",
    "cool",
    "corporate",
    "dark",
    "deep",
    "documentary",
    "drama",
    "dramatic",
    "dream",
    "emotional",
    "energetic",
    "epic",
    "fast",
    "film",
    "fun",
    "funny",
    "game",
    "groovy",
    "happy",
    "heavy",
    "holiday",
    "hopeful",
    "inspiring",
    "love",
    "meditative",
    "melancholic",
    "melodic",
    "motivational",
    "movie",
    "nature",
    "party",
    "positive",
    "powerful",
    "relaxing",
    "retro",
    "romantic",
    "sad",
    "sexy",
    "slow",
    "soft",
    "soundscape",
    "space",
    "sport",
    "summer",
    "trailer",
    "travel",
    "upbeat",
    "uplifting",
]

# Define some categories
positive_happy_moods = [
    "happy",
    "fun",
    "funny",
    "groovy",
    "hopeful",
    "inspiring",
    "love",
    "positive",
    "powerful",
    "romantic",
    "sexy",
    "upbeat",
    "uplifting",
]
energetic_active_moods = [
    "action",
    "adventure",
    "energetic",
    "epic",
    "fast",
    "game",
    "sport",
    "summer",
]
calm_relaxing_moods = [
    "background",
    "calm",
    "meditative",
    "relaxing",
    "soft",
    "soundscape",
]
emotional_melancholic_moods = [
    "ballad",
    "drama",
    "dramatic",
    "emotional",
    "melancholic",
    "sad",
]
motivational_corporate_moods = ["commercial", "corporate", "motivational"]
nature_environmental_moods = ["nature", "space"]
festive_seasonal_moods = ["christmas", "holiday"]
entertainment_media_moods = [
    "advertising",
    "documentary",
    "film",
    "movie",
    "trailer",
    "travel",
]
other_moods = [
    "children",
    "cool",
    "deep",
    "dream",
    "heavy",
    "melodic",
    "party",
    "retro",
    "slow",
]

SIMPLIFIED_MOODS = ["positive", "energetic", "calm", "emotional", "film"]


def get_simplified_moods(moods: List[str]) -> List[str]:
    simplified_moods = set()
    for m in moods:
        if m in positive_happy_moods:
            simplified_moods.add("positive")
        elif m in energetic_active_moods:
            simplified_moods.add("energetic")
        elif m in calm_relaxing_moods:
            simplified_moods.add("calm")
        elif m in emotional_melancholic_moods:
            simplified_moods.add("emotional")
        elif m in entertainment_media_moods:
            simplified_moods.add("film")

    return list(simplified_moods)


instruments_classes = [
    "accordion",
    "acoustic bass",
    "acoustic guitar",
    "agogo",
    "alto saxophone",
    "applause",
    "bagpipe",
    "banjo",
    "baritone saxophone",
    "bassoon",
    "bird tweet",
    "bottle blow",
    "brass lead",
    "brass section",
    "breath noise",
    "calliope lead",
    "celesta",
    "cello",
    "charang lead",
    "chiffer lead",
    "choir aahs",
    "church organ",
    "clarinet",
    "clavinet",
    "clean electric guitar",
    "contrabass",
    "distortion guitar",
    "drums",
    "dulcimer",
    "electric bass",
    "electric guitar",
    "electric piano",
    "english horn",
    "fiddle",
    "flute",
    "french horn",
    "fretless bass",
    "glockenspiel",
    "guitar fret noise",
    "guitar harmonics",
    "gunshot",
    "hammond organ",
    "harmonica",
    "harpsichord",
    "helicopter",
    "honky-tonk piano",
    "kalimba",
    "koto",
    "marimba",
    "melodic tom",
    "music box",
    "muted trumpet",
    "oboe",
    "ocarina",
    "orchestra hit",
    "orchestral harp",
    "overdriven guitar",
    "pan flute",
    "percussive organ",
    "piano",
    "piccolo",
    "pizzicato strings",
    "recorder",
    "reed organ",
    "reverse cymbal",
    "rock organ",
    "seashore",
    "shakuhachi",
    "shamisen",
    "shana",
    "sitar",
    "slap bass",
    "soprano saxophone",
    "steel drums",
    "string ensemble",
    "synth bass",
    "synth brass",
    "synth drum",
    "synth effects",
    "synth lead",
    "synth pad",
    "synth strings",
    "synth voice",
    "taiko drum",
    "tango accordion",
    "telephone ring",
    "tenor saxophone",
    "timpani",
    "tinkle bell",
    "tremolo strings",
    "trombone",
    "trumpet",
    "tuba",
    "tubular bells",
    "vibraphone",
    "viola",
    "violin",
    "voice oohs",
    "whistle",
    "woodblock",
    "xylophone",
]


def download_models():
    MODEL_FOLDER.mkdir(exist_ok=True, parents=True)

    # embeddings
    model_path = MODEL_FOLDER / "discogs-effnet-bs64-1.pb"

    if not model_path.exists():
        urllib.request.urlretrieve(
            "https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb",
            str(model_path),
        )
    metadata_path = MODEL_FOLDER / "discogs-effnet-bs64-1.json"
    if not metadata_path.exists():
        urllib.request.urlretrieve(
            "https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.json",
            str(metadata_path),
        )

    # genre
    model_path = MODEL_FOLDER / "mtg_jamendo_genre-discogs-effnet-1.pb"
    if not model_path.exists():
        urllib.request.urlretrieve(
            "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb",
            str(model_path),
        )
    metadata_path = MODEL_FOLDER / "mtg_jamendo_genre-discogs-effnet-1.json"
    if not metadata_path.exists():
        urllib.request.urlretrieve(
            "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.json",
            str(metadata_path),
        )

    # mood
    model_path = MODEL_FOLDER / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
    if not model_path.exists():
        urllib.request.urlretrieve(
            "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
            str(model_path),
        )
    metadata_path = MODEL_FOLDER / "mtg_jamendo_moodtheme-discogs-effnet-1.json"
    if not metadata_path.exists():
        urllib.request.urlretrieve(
            "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json",
            str(metadata_path),
        )


@lru_cache
def get_audio_embedding_model():
    model_path = MODEL_FOLDER / "discogs-effnet-bs64-1.pb"

    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_path), output="PartitionedCall:1"
    )

    return embedding_model


@lru_cache
def get_genre_model():
    model_path = MODEL_FOLDER / "mtg_jamendo_genre-discogs-effnet-1.pb"
    metadata_path = MODEL_FOLDER / "mtg_jamendo_genre-discogs-effnet-1.json"
    model = TensorflowPredict2D(graphFilename=str(model_path))
    metadata = str(metadata_path)
    return model, metadata


@lru_cache
def get_mood_model():
    model_path = MODEL_FOLDER / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
    metadata_path = MODEL_FOLDER / "mtg_jamendo_moodtheme-discogs-effnet-1.json"
    model = TensorflowPredict2D(graphFilename=str(model_path))
    metadata = str(metadata_path)
    return model, metadata


@lru_cache
def get_chord_estimator():
    return Chordino()


def get_mtg_tags(embeddings, tag_model, tag_json, max_num_tags=5, tag_threshold=0.01):

    with open(tag_json, "r") as json_file:
        metadata = json.load(json_file)
    predictions = tag_model(embeddings)
    mean_act = np.mean(predictions, 0)
    ind = np.argpartition(mean_act, -max_num_tags)[-max_num_tags:]
    tags = []
    confidence_score = []
    for i in ind:
        # print(metadata['classes'][i] + str(mean_act[i]))
        if mean_act[i] > tag_threshold:
            tags.append(metadata["classes"][i])
            confidence_score.append(mean_act[i])
    ind = np.argsort(-np.array(confidence_score))
    tags = [tags[i] for i in ind]
    confidence_score = np.round((np.array(confidence_score)[ind]).tolist(), 4).tolist()

    return tags, confidence_score


def get_chords(wav_audio_path: Path):
    chord_estimator = get_chord_estimator()
    chords = chord_estimator.extract(str(wav_audio_path))
    chords_out = [(x.chord, round(x.timestamp, 4)) for x in chords[1:-1]]

    # chord summary
    ch_name = []
    ch_time = []
    for ch in chords_out:
        ch_name.append(ch[0])
        ch_time.append(ch[1])
    if len(ch_name) < 3:
        final_seq = ch_name
        final_count = 1
    else:
        final_seq, final_count = _give_me_final_seq(ch_name)
    if final_seq is not None:
        if len(final_seq) == 4:
            if final_seq[0] == final_seq[2] and final_seq[1] == final_seq[3]:
                final_seq = final_seq[0:2]
    chord_summary = [final_seq, final_count]
    return chords_out, chord_summary[0], chord_summary[1]


def get_mood_and_genre(wav_audio_path: Path):
    audio = MonoLoader(
        filename=str(wav_audio_path),
        sampleRate=16000,
        resampleQuality=1,
    )()

    embedding_model = get_audio_embedding_model()
    genmodel, genre_metadata = get_genre_model()
    moodmodel, mood_metadata = get_mood_model()

    embeddings = embedding_model(audio)
    mood_tags, mood_cs = get_mtg_tags(
        embeddings,
        moodmodel,
        mood_metadata,
        max_num_tags=5,
        tag_threshold=0.02,
    )
    genre_tags, genre_cs = get_mtg_tags(
        embeddings,
        genmodel,
        genre_metadata,
        max_num_tags=4,
        tag_threshold=0.05,
    )

    return mood_tags, mood_cs, genre_tags, genre_cs


def _find_most_repeating_sequence(chords_list, sequence_length):
    sequences = [
        tuple(chords_list[i : i + sequence_length])
        for i in range(len(chords_list) - sequence_length + 1)
    ]
    delete_index = []
    for i, seqs in enumerate(sequences):
        if seqs[0] == seqs[-1]:
            delete_index.append(i)
    for i in reversed(delete_index):
        sequences.pop(i)
    sequence_counts = Counter(sequences)
    try:
        most_common_sequence, count = sequence_counts.most_common(1)[0]
        return most_common_sequence, count
    except Exception:
        # print(e)
        return None, 0


def _give_me_final_seq(chords):
    sequence_3, count_3 = _find_most_repeating_sequence(chords, 3)
    sequence_4, count_4 = _find_most_repeating_sequence(chords, 4)
    sequence_5, count_5 = _find_most_repeating_sequence(chords, 5)
    total_count = count_3 + count_4 + count_5
    if count_5 > 0.25 * (total_count):
        if count_5 > 0.79 * count_4:
            return sequence_5, count_5
    if count_4 > 0.3 * (total_count):
        if count_4 > 0.79 * count_3:
            return sequence_4, count_4
    if count_3 == 0:
        if count_4 == 0:
            if count_5 == 0:
                return None, 0  # everything is 0
            else:
                return sequence_5, count_5
        else:
            return sequence_4, count_4
    else:
        return sequence_3, count_3
