import itertools
import json
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List

from midistral.audio_analysis import (
    SIMPLIFIED_GENRES,
    SIMPLIFIED_MOODS,
    get_simplified_genres,
    get_simplified_moods,
)
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from datasets.utils import disable_progress_bars, enable_progress_bars
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from midistral.dataset_features import midi_features
from midistral.midi_utils import (
    get_abc_from_midi,
    get_instrument_number,
    get_midi_tracks_nums,
)
from midistral.types import AudioTextDescription

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


def generate_instruction(desc: AudioTextDescription) -> str:
    instructions = []
    if len(desc.genre) > 0:
        instructions.append("genre : " + ", ".join(desc.genre))
    if len(desc.mood) > 0:
        instructions.append("mood : " + ", ".join(desc.mood))
    if len(desc.instruments) > 0:
        instructions.append("instruments : " + ", ".join(desc.instruments))

    return f"""You are a powerful text to ABC music notation model.
Generate ABC music notation with {'; '.join(instructions)}.
Don't explain anything."""


def generate_instruction_for_finetuned_model(
    desc: AudioTextDescription, with_instrument_num: bool
) -> str:
    # desc.model_dump_json()
    instructions = []
    if len(desc.genre) > 0:
        instructions.append("genre : " + ", ".join(desc.genre))
    if len(desc.mood) > 0:
        instructions.append("mood : " + ", ".join(desc.mood))

    if with_instrument_num:
        if desc.midi_instruments_num and len(desc.midi_instruments_num) > 0:
            instructions.append(
                "instruments : "
                + ", ".join([str(i) for i in desc.midi_instruments_num])
            )
        else:
            if len(desc.instruments) > 0:
                instructions.append(
                    "instruments : "
                    + ", ".join(
                        [str(get_instrument_number(i)) for i in desc.instruments]
                    )
                )
    else:
        if len(desc.instruments) > 0:
            instructions.append("instruments : " + ", ".join(desc.instruments))

    return "\n".join(instructions)


def generate_rag_instruction(abc_notations: List[str]) -> str:
    return (
        "Generate an ABC music notation inspired from these examples. Output only the generated ABC music notation. Keep similar MIDI program.\n"
        + "\n".join(abc_notations)
    )


def get_instruction_data(r, with_instrument_num: bool) -> Dict:

    desc = AudioTextDescription(
        genre=r["genre"],
        mood=r["mood"],
        instruments=r["instrument_summary"],
        midi_instruments_num=list(set(r["instrument_numbers_sorted"])),
    )
    text_instruction = generate_instruction_for_finetuned_model(
        desc, with_instrument_num=with_instrument_num
    )
    generation = r["abc_notation"]

    # remove escaped slash (\/) and continuation character (\\\n)
    formatted_generation = re.sub(r"\\/", "/", generation)
    formatted_generation = re.sub(r"\\\\\\n", "", formatted_generation).strip()

    return {
        "messages": [
            {
                "role": "user",
                "content": text_instruction,
            },
            {
                "role": "assistant",
                "content": formatted_generation,
            },
        ]
    }


def generate_test_cases():
    SOME_MAIN_INSTRUMENTS = [
        "piano",
        "acoustic guitar",
        "trumpet",
        "ocarina",
    ]
    all_instruments_combinations = []
    all_moods_combinations = []
    all_genres_combinations = []

    max_instrument_selection = 2
    for i in range(0, max_instrument_selection + 1):
        els = [list(x) for x in itertools.combinations(SOME_MAIN_INSTRUMENTS, i)]
        all_instruments_combinations.extend(els)

    max_mood_selection = 2
    for i in range(1, max_mood_selection + 1):
        els = [
            list(x)
            for x in itertools.combinations(SIMPLIFIED_MOODS, i)
            if list(x) != ["energetic", "calm"]
        ]
        all_moods_combinations.extend(els)

    max_genre_selection = 1
    for i in range(1, max_genre_selection + 1):
        els = [list(x) for x in itertools.combinations(SIMPLIFIED_GENRES, i)]
        all_genres_combinations.extend(els)

    tests_cases = []
    for i, m, g in itertools.product(
        all_instruments_combinations, all_moods_combinations, all_genres_combinations
    ):
        tests_cases.append({"instrument_summary": i, "genre": g, "mood": m})

    return tests_cases


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
        and "(" not in r["abc_notation"]
        and "<" not in r["abc_notation"]
        and ">" not in r["abc_notation"]
        and "%%MIDI program" in r["abc_notation"]
        and len(r["instrument_summary"]) > 0
        and len(r["instrument_summary"]) <= 2
    )

    # simplified mood and genre tags
    midi_abc_dataset_subset = midi_abc_dataset_subset.map(
        lambda r: {
            "genre": get_simplified_genres(r["genre"][:2]),
            "mood": get_simplified_moods(r["mood"]),
            "instrument_summary": [i.lower() for i in r["instrument_summary"]],
        }
    )

    # split data with stratify option (to balance genre, mood, and instrument in train and test set)
    if not train_midi_abc_dataset_path.exists():
        train_df_l = []
        test_df_l = []
        split_by_constraints = []

        test_cases = generate_test_cases()
        max_count = 40
        disable_progress_bars()
        for constraints in tqdm(test_cases):
            subset_test_case = midi_abc_dataset_subset.filter(
                lambda r: set(constraints["genre"]).issubset(set(r["genre"]))
                and set(constraints["mood"]).issubset(set(r["mood"]))
                and set(constraints["instrument_summary"]).issubset(
                    set(r["instrument_summary"])
                )
            )
            subset_test_case_df = subset_test_case.to_pandas()
            sample = subset_test_case_df.sample(
                n=min(max_count, len(subset_test_case_df)), random_state=42
            )

            # set their value to the test constraints
            sample["genre"] = sample["genre"].apply(lambda r: constraints["genre"])
            sample["mood"] = sample["mood"].apply(lambda r: constraints["mood"])

            if len(constraints["instrument_summary"]) == 0:
                sample["instrument_summary"] = sample["instrument_summary"].apply(
                    lambda r: constraints["instrument_summary"]
                )

            if len(sample) >= 4:
                subset_test_case_df_train, subset_test_case_df_test = train_test_split(
                    sample, test_size=0.04, random_state=42
                )
                split_by_constraints.append(
                    {
                        "constraints": constraints,
                        "num_samples_train": len(subset_test_case_df_train),
                        "num_samples_test": len(subset_test_case_df_test),
                    }
                )

                train_df_l.append(subset_test_case_df_train)
                test_df_l.append(subset_test_case_df_test)
        enable_progress_bars()

        train_df = pd.concat(train_df_l)
        test_df = pd.concat(test_df_l)

        with num_samples_by_constraint_path.open("w") as f:
            json.dump(split_by_constraints, f, indent=2)

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        train_dataset.save_to_disk(train_midi_abc_dataset_path)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
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
