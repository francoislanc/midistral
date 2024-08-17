import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import mido

from midistral.abc_utils import clean_generated_abc


def convert_midi_to_ogg(midi_path: Path, ogg_path: Path):
    cmd = [
        "timidity",
        str(midi_path.resolve()),
        "-Ov",
        "-o",
        str(ogg_path.resolve()),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if len(p.stdout) > 0:
        logging.debug(f"convert_midi_to_ogg {midi_path} - {p.stdout}")
    if len(p.stderr) > 0:
        logging.debug(f"convert_midi_to_ogg {midi_path} - {p.stderr}")


def convert_midi_to_abc(midi_path: Path, abc_path: Path):
    cmd = [
        "midi2abc",
        "-noly",
        # "-xa",
        "-title",
        "",
        "-f",
        str(midi_path.resolve()),
        "-o",
        str(abc_path.resolve()),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if len(p.stdout) > 0:
        logging.debug(f"convert_midi_to_abc {midi_path} - {p.stdout}")
    if len(p.stderr) > 0:
        logging.debug(f"convert_midi_to_abc {midi_path} - {p.stderr}")


def convert_abc_to_midi(
    abc_path: Path,
    midi_path: Path,
):
    cmd = [
        "abc2midi",
        str(abc_path.resolve()),
        "-o",
        str(midi_path.resolve()),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if len(p.stdout) > 0:
        logging.debug(f"convert_abc_to_midi {abc_path} - {p.stdout}")
    if len(p.stderr) > 0:
        logging.debug(f"convert_abc_to_midi {abc_path} - {p.stderr}")


def get_abc_from_midi(midi_path: Path) -> Optional[str]:
    abc = None
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as abc_fp:
        try:
            convert_midi_to_abc(midi_path, Path(abc_fp.name))
            abc_fp.seek(0)
            abc = abc_fp.read()
        except Exception as e:
            logging.exception(e)
            abc = None
    if abc is not None:
        abc = clean_generated_abc(abc)
        if len(abc) == 0:
            abc = None
    return abc


def get_midi_and_ogg_from_abc(abc_notation: str) -> bytes:
    midi = bytes()
    ogg = bytes()
    with tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8"
    ) as abc_fp, tempfile.NamedTemporaryFile() as midi_fp, tempfile.NamedTemporaryFile() as ogg_fp:
        abc_fp.write(abc_notation)
        abc_fp.seek(0)

        convert_abc_to_midi(Path(abc_fp.name), Path(midi_fp.name))
        convert_midi_to_ogg(Path(midi_fp.name), Path(ogg_fp.name))

        midi_fp.seek(0)
        midi = midi_fp.read()
        ogg_fp.seek(0)
        ogg = ogg_fp.read()
    return midi, ogg


def get_midi_tracks_nums(midi_path: Path) -> int:
    mid = mido.MidiFile(midi_path)
    num = 0
    for t in mid.tracks:
        for m in t:
            if (
                isinstance(m, mido.messages.messages.Message)
                and m.type == "program_change"
            ):
                num += 1
    return num
