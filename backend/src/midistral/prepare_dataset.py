import itertools
import json
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from datasets.utils import disable_progress_bars, enable_progress_bars
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from midistral.abc_utils import count_max_opened_parentheses, has_only_silence
from midistral.audio_analysis import (
    SIMPLIFIED_GENRES,
    SIMPLIFIED_MOODS,
    get_simplified_genres,
    get_simplified_moods,
)
from midistral.dataset_features import midi_features
from midistral.generate import get_instruction_data
from midistral.midi_utils import (
    get_abc_from_midi,
    get_instrument_number,
    get_midi_tracks_nums,
)

pd.options.mode.copy_on_write = True
DATA_FOLDER = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_FOLDER = Path(__file__).resolve().parent.parent.parent / "output"


def extract_midicaps_files() -> None:
    compressed_midi = hf_hub_download(
        repo_id="amaai-lab/MidiCaps", filename="midicaps.tar.gz", repo_type="dataset"
    )

    midi_dir = DATA_FOLDER / "lmd_full"

    if not midi_dir.exists():
        with tarfile.open(compressed_midi) as tar:
            files = tar.getmembers()
            for f in tqdm(iterable=files, total=len(files)):
                tar.extract(path=DATA_FOLDER, member=f)

    output_path_str = hf_hub_download(
        repo_id="amaai-lab/MidiCaps",
        filename="train.json",
        repo_type="dataset",
        revision="84d5b132f2776cc5276455fe90982b4ff768a6f5",
    )

    output_data_folder_path = DATA_FOLDER / "midicaps.jsonl"
    shutil.copy(output_path_str, output_data_folder_path)


def clean_labeled_midi_dataset(
    origin: str,
    labeled_midi_dataset_df: pd.DataFrame,
    keep_only_small_subset: bool,
    num_proc: int,
) -> Dataset:

    labeled_midi_dataset_df = labeled_midi_dataset_df[
        [
            "location",
            "genre",
            "mood",
            "key",
            "time_signature",
            "tempo",
            "duration",
            "instrument_summary",
            "instrument_numbers_sorted",
            "chord_summary",
        ]
    ]

    labeled_midi_dataset_df["origin"] = origin
    if keep_only_small_subset:
        labeled_midi_dataset_df = labeled_midi_dataset_df.head(500)
    # create dataset
    labeled_midi_dataset = Dataset.from_pandas(
        labeled_midi_dataset_df, preserve_index=False
    )
    labeled_midi_dataset = labeled_midi_dataset.filter(
        lambda r: r["tempo"] is not None and r["chord_summary"] is not None
    )
    labeled_midi_dataset = labeled_midi_dataset.cast(midi_features)

    return labeled_midi_dataset


def prepare_midi_dataset(labeled_midi_dataset_path: Path, num_proc: int) -> Dataset:
    labeled_midi_dataset = load_from_disk(labeled_midi_dataset_path)

    labeled_midi_dataset = labeled_midi_dataset.filter(lambda r: r["duration"] < 100)

    midi_abc_dataset = labeled_midi_dataset.map(
        lambda x: {
            "abc_notation": get_abc_from_midi(DATA_FOLDER / x["location"]),
            "midi_tracks_nums": get_midi_tracks_nums(DATA_FOLDER / x["location"]),
        },
        num_proc=num_proc,
        writer_batch_size=int(600 / num_proc),
    )

    midi_abc_dataset = midi_abc_dataset.filter(
        lambda x: x["abc_notation"] is not None,
    )

    return midi_abc_dataset


def generate_train_cases():
    SOME_MAIN_INSTRUMENTS = [
        "piano",
        "trumpet",
        "ocarina",
        "clarinet",
        "electric guitar",
        "string ensemble",
        "trombone",
        "acoustic bass",
        "synth lead",
        "acoustic guitar",
    ]
    all_instruments_combinations = []
    all_moods_combinations = []
    all_genres_combinations = []

    max_instrument_selection = 1
    for i in range(0, max_instrument_selection + 1):
        els = [list(x) for x in itertools.combinations(SOME_MAIN_INSTRUMENTS, i)]
        all_instruments_combinations.extend(els)

    max_mood_selection = 2
    for i in range(0, max_mood_selection + 1):
        els = [
            list(x)
            for x in itertools.combinations(SIMPLIFIED_MOODS, i)
            if list(x) != ["energetic", "calm"]
        ]
        all_moods_combinations.extend(els)

    max_genre_selection = 1
    for i in range(0, max_genre_selection + 1):
        els = [list(x) for x in itertools.combinations(SIMPLIFIED_GENRES, i)]
        all_genres_combinations.extend(els)

    cases = []

    for i in all_instruments_combinations:
        if len(i) > 0:
            cases.append({"instrument_summary": i, "genre": [], "mood": []})

    for i, m, g in itertools.product(
        all_instruments_combinations, all_moods_combinations, all_genres_combinations
    ):
        if len(g) > 0 or len(m) > 0:
            cases.append({"instrument_summary": i, "genre": g, "mood": m})

    return cases


def prepare_dataset(keep_only_small_subset: bool) -> None:
    data_origins = ["midicaps", "vgm", "irishman"]
    labeled_midi_dataset_path = OUTPUT_FOLDER / "datasets" / "labeled_midi_dataset"
    midi_abc_dataset_path = OUTPUT_FOLDER / "datasets" / "midi_abc_dataset"
    num_samples_by_constraint_path = (
        OUTPUT_FOLDER / "datasets" / "num_samples_by_constraint.json"
    )
    train_midi_abc_dataset_path = OUTPUT_FOLDER / "datasets" / "midi_abc_dataset-train"
    test_midi_abc_dataset_path = OUTPUT_FOLDER / "datasets" / "midi_abc_dataset-test"
    llm_finetuning_train_dataset_path = (
        OUTPUT_FOLDER / "llm_finetuning_dataset-train.jsonl"
    )
    llm_finetuning_eval_dataset_path = (
        OUTPUT_FOLDER / "llm_finetuning_dataset-eval.jsonl"
    )

    datasets: List[Dataset] = []
    if not labeled_midi_dataset_path.exists():
        labeled_midi_dataset_path.parent.mkdir(exist_ok=True, parents=True)

        for d in data_origins:
            if d == "midicaps":
                extract_midicaps_files()

            labeled_file = DATA_FOLDER / f"{d}.jsonl"
            if labeled_file.exists():
                labeled_midi_dataset_df = pd.read_json(labeled_file, lines=True)
                labeled_midi_dataset = clean_labeled_midi_dataset(
                    d, labeled_midi_dataset_df, keep_only_small_subset, 5
                )
                print(labeled_midi_dataset)
                datasets.append(labeled_midi_dataset)
        if len(datasets) == 1:
            labeled_midi_dataset = datasets[0]
        else:
            labeled_midi_dataset = concatenate_datasets(datasets)
        labeled_midi_dataset.save_to_disk(labeled_midi_dataset_path)

    if not midi_abc_dataset_path.exists():
        midi_abc_dataset = prepare_midi_dataset(labeled_midi_dataset_path, 5)
        midi_abc_dataset.save_to_disk(midi_abc_dataset_path)

    midi_abc_dataset = load_from_disk(midi_abc_dataset_path)
    print(midi_abc_dataset)

    # subset only of the dataset to simplify the problem
    midi_abc_dataset_subset = midi_abc_dataset.filter(
        lambda r: r["duration"] < 60
        and r["duration"] > 5
        and "[" not in r["abc_notation"]
        and count_max_opened_parentheses(r["abc_notation"]) == 0
        and "%%MIDI program" in r["abc_notation"]
        and len(r["instrument_summary"]) > 0
        and len(r["instrument_summary"]) <= 1
        and not has_only_silence(r["abc_notation"])
    )

    # simplified mood and genre tags
    midi_abc_dataset_subset = midi_abc_dataset_subset.map(
        lambda r: {
            "genre": get_simplified_genres(r["genre"][:2]),
            "mood": get_simplified_moods(r["mood"][:2]),
            "instrument_summary": [i.lower() for i in r["instrument_summary"]],
        }
    )

    # split data with stratify option (to balance genre, mood, and instrument in train and test set)
    if not train_midi_abc_dataset_path.exists():
        train_df_l = []
        test_df_l = []
        split_by_constraints = []

        train_cases = generate_train_cases()
        max_samples = 64
        min_samples = 4
        disable_progress_bars()
        for constraints in tqdm(train_cases):
            numbered_instruments = [
                get_instrument_number(i) for i in constraints["instrument_summary"]
            ]
            subset = midi_abc_dataset_subset.filter(
                lambda r: set(constraints["genre"]).issubset(set(r["genre"]))
                and set(constraints["mood"]).issubset(set(r["mood"]))
                and set(numbered_instruments).issubset(
                    set(r["instrument_numbers_sorted"])
                )
            )
            subset_df = subset.to_pandas()
            sample = subset_df.sample(
                n=min(max_samples, len(subset_df)), random_state=42
            )

            # set their value to the test constraints
            sample["genre"] = sample["genre"].apply(lambda r: constraints["genre"])
            sample["mood"] = sample["mood"].apply(lambda r: constraints["mood"])

            if len(constraints["instrument_summary"]) == 0:
                sample["instrument_summary"] = sample["instrument_summary"].apply(
                    lambda r: []
                )
                sample["instrument_numbers_sorted"] = sample[
                    "instrument_summary"
                ].apply(lambda r: None)

            if len(sample) >= min_samples:
                subset_df_train, subset_df_test = train_test_split(
                    sample, test_size=0.04, random_state=42
                )
                split_by_constraints.append(
                    {
                        "constraints": constraints,
                        "num_samples_train": len(subset_df_train),
                        "num_samples_test": len(subset_df_test),
                    }
                )

                train_df_l.append(subset_df_train)
                test_df_l.append(subset_df_test)
        enable_progress_bars()

        train_df = pd.concat(train_df_l)
        test_df = pd.concat(test_df_l)

        with num_samples_by_constraint_path.open("w") as f:
            json.dump(split_by_constraints, f, indent=2)

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        train_dataset = train_dataset.shuffle(seed=42)
        train_dataset.save_to_disk(train_midi_abc_dataset_path)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
        test_dataset = test_dataset.shuffle(seed=42)
        test_dataset.save_to_disk(test_midi_abc_dataset_path)
    else:
        train_dataset = load_from_disk(train_midi_abc_dataset_path)
        test_dataset = load_from_disk(test_midi_abc_dataset_path)

    with llm_finetuning_train_dataset_path.open("w") as f:
        for r in train_dataset:
            instruct = get_instruction_data(r, with_instrument_num=True)
            f.write(f"{json.dumps(instruct)}\n")
            f.flush()

    with llm_finetuning_eval_dataset_path.open("w") as f:
        for r in test_dataset:
            instruct = get_instruction_data(r, with_instrument_num=True)
            f.write(f"{json.dumps(instruct)}\n")
            f.flush()

    # generate audio file for test dataset
    # for r in splitted_dataset["train"].take(11):
    #     print(r["location"])
    #     midi, ogg = get_midi_and_ogg_from_abc(r["abc_notation"])
    #     p = OUTPUT_FOLDER / "audio" / f"{Path(r['location']).stem}.ogg"
    #     p.parent.mkdir(exist_ok=True)
    #     with p.open("wb") as f:
    #         f.write(ogg)
