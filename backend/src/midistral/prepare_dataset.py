import re
import shutil
import tarfile
from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from midistral.abc_utils import count_max_opened_parentheses
from midistral.midi_utils import (
    get_abc_from_midi,
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
    origin: str, labeled_midi_dataset_df: pd.DataFrame, keep_only_small_subset: bool
) -> Dataset:

    labeled_midi_dataset_df = labeled_midi_dataset_df[
        [
            "location",
            "genre",
            "mood",
            "key",
            "time_signature",
            "duration",
            "instrument_summary",
        ]
    ]
    labeled_midi_dataset_df["origin"] = origin

    if keep_only_small_subset:
        labeled_midi_dataset_df = labeled_midi_dataset_df.head(500)
    # create dataset
    labeled_midi_dataset = Dataset.from_pandas(
        labeled_midi_dataset_df, preserve_index=False
    )

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
        instructions.append("genre : " + ", ".join([g.lower() for g in desc.genre]))

    if len(desc.mood) > 0:
        instructions.append("mood : " + ", ".join([g.lower() for g in desc.mood]))

    if len(desc.instruments) > 0:
        instructions.append(
            "instruments : " + ", ".join([x.lower() for x in desc.instruments])
        )

    return "\n".join(instructions)


def generate_rag_instruction(abc_notations: List[str]) -> str:
    return (
        "Generate an ABC music notation inspired from these examples. Output only the generated ABC music notation. Keep similar MIDI program.\n"
        + "\n".join(abc_notations)
    )


def add_instruction_data(r):

    desc = AudioTextDescription(
        genre=r["genre"], mood=r["mood"], instruments=r["instrument_summary"]
    )
    text_instruction = generate_instruction(desc)

    return {
        "messages": [
            {
                "role": "user",
                "content": text_instruction,
            },
            {
                "role": "assistant",
                "content": f"{r['abc_notation']}",
            },
        ]
    }


def prepare_dataset(keep_only_small_subset: bool) -> None:
    data_origins = ["midicaps", "vgm", "irishman"]
    labeled_midi_dataset_path = OUTPUT_FOLDER / "datasets" / "labeled_midi_dataset"
    midi_abc_dataset_path = OUTPUT_FOLDER / "datasets" / "midi_abc_dataset"
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
                    d, labeled_midi_dataset_df, keep_only_small_subset
                )
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
        and r["duration"] > 2
        and count_max_opened_parentheses(r["abc_notation"]) <= 1
    )

    llm_finetuning_dataset = midi_abc_dataset_subset.map(
        add_instruction_data,
    )

    splitted_dataset = llm_finetuning_dataset.train_test_split(test_size=0.04, seed=42)
    unused_columns = [
        "location",
        "genre",
        "mood",
        "key",
        "time_signature",
        "duration",
        "instrument_summary",
        "midi_tracks_nums",
        "abc_notation",
    ]

    train_df = splitted_dataset["train"].remove_columns(unused_columns).to_pandas()
    test_df = splitted_dataset["test"].remove_columns(unused_columns).to_pandas()

    # remove escaped slash (\/) and continuation character (\\\n)
    formatted_json_train = train_df.to_json(orient="records", lines=True)
    formatted_json_train = re.sub(r"\\/", "/", formatted_json_train)
    formatted_json_train = re.sub(r"\\\\\\n", "", formatted_json_train).strip()

    formatted_json_eval = test_df.to_json(orient="records", lines=True)
    formatted_json_eval = re.sub(r"\\/", "/", formatted_json_eval)
    formatted_json_eval = re.sub(r"\\\\\\n", "", formatted_json_eval).strip()

    print(formatted_json_train, file=llm_finetuning_train_dataset_path.open("w"))
    print(formatted_json_eval, file=llm_finetuning_eval_dataset_path.open("w"))

    # generate audio file for test dataset
    # for r in splitted_dataset["train"].take(11):
    #     print(r["location"])
    #     midi, ogg = get_midi_and_ogg_from_abc(r["abc_notation"])
    #     p = OUTPUT_FOLDER / "audio" / f"{Path(r['location']).stem}.ogg"
    #     p.parent.mkdir(exist_ok=True)
    #     with p.open("wb") as f:
    #         f.write(ogg)
