import re
import tarfile
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_from_disk
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from midistral.dataset_features import midi_abc_features
from midistral.midi_utils import (
    get_abc_from_midi,
    get_midi_channel_nums2,
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


def clean_midi_caption_dataset(
    caption_dataset_path: Path, keep_only_small_subset: bool
) -> Dataset:
    caption_dataset_path.parent.mkdir(exist_ok=True, parents=True)
    caption_dataset = pd.read_json(
        hf_hub_download(
            repo_id="amaai-lab/MidiCaps",
            filename="captions_with_features.json",
            repo_type="dataset",
        ),
        lines=True,
    )
    # remove confidence score or uneeded value
    for k in ["genre", "mood", "chord_summary", "tempo", "duration"]:
        caption_dataset[k] = caption_dataset[k].map(lambda x: x[0])
    # remove row with invalid value
    caption_dataset = caption_dataset[
        caption_dataset.tempo.apply(lambda x: isinstance(x, int))
    ]

    if keep_only_small_subset:
        caption_dataset = caption_dataset.head(500)
    # create dataset
    caption_dataset = Dataset.from_pandas(caption_dataset, preserve_index=False)

    return caption_dataset


def prepare_midi_dataset(caption_dataset_path: Path, num_proc: int) -> Dataset:
    caption_dataset = load_from_disk(caption_dataset_path)

    caption_dataset = caption_dataset.filter(lambda r: r["duration"] < 100)

    midi_abc_dataset = caption_dataset.map(
        lambda x: {
            "abc_notation": get_abc_from_midi(DATA_FOLDER / x["location"]),
        },
        num_proc=num_proc,
        writer_batch_size=int(600 / num_proc),
    )
    midi_abc_dataset = midi_abc_dataset.map(
        lambda x: {
            "midi_channel_nums": get_midi_channel_nums2(x["abc_notation"]),
        },
        features=midi_abc_features,
        num_proc=num_proc,
        writer_batch_size=int(600 / num_proc),
    )
    midi_abc_dataset = midi_abc_dataset.filter(
        lambda x: x["abc_notation"] is not None,
    )

    return midi_abc_dataset


def generate_instruction(desc: AudioTextDescription):
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
    caption_dataset_path = OUTPUT_FOLDER / "datasets" / "caption_dataset"
    midi_abc_dataset_path = OUTPUT_FOLDER / "datasets" / "midi_abc_dataset"
    llm_finetuning_train_dataset_path = (
        OUTPUT_FOLDER / "llm_finetuning_dataset-train.jsonl"
    )
    llm_finetuning_eval_dataset_path = (
        OUTPUT_FOLDER / "llm_finetuning_dataset-eval.jsonl"
    )

    extract_midicaps_files()

    if not caption_dataset_path.exists():
        caption_dataset = clean_midi_caption_dataset(
            caption_dataset_path, keep_only_small_subset
        )
        caption_dataset.save_to_disk(caption_dataset_path)

    if not midi_abc_dataset_path.exists():
        midi_abc_dataset = prepare_midi_dataset(caption_dataset_path, 5)
        midi_abc_dataset.save_to_disk(midi_abc_dataset_path)

    midi_abc_dataset = load_from_disk(midi_abc_dataset_path)
    print(midi_abc_dataset)

    # subset only of the dataset to simplify the problem
    midi_abc_dataset_subset = midi_abc_dataset.filter(
        lambda r: r["midi_channel_nums"] <= 2
        and r["duration"] < 60
        and r["duration"] > 2
        and "[" not in r["abc_notation"]  # no chords
        and "(" not in r["abc_notation"]  # no slurs and ties
    )

    llm_finetuning_dataset = midi_abc_dataset_subset.map(
        add_instruction_data,
    )

    splitted_dataset = llm_finetuning_dataset.train_test_split(test_size=0.04, seed=42)
    unused_columns = [
        "location",
        "caption",
        "genre",
        "mood",
        "key",
        "time_signature",
        "tempo",
        "duration",
        "chord_summary",
        "instrument_summary",
        "midi_channel_nums",
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