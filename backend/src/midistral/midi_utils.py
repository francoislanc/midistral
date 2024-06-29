import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import mido


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


def repl_midi_program(match_obj):
    if match_obj.group(1):
        return match_obj.group(1) + match_obj.group(2)
    elif match_obj.group(3):
        return ""


def clean_generated_abc(abc: str):
    abc = abc.replace("%***Missing time signature meta command in MIDI file\n", "")
    abc = abc.replace("\n\n", "\n")
    abc = re.sub(r"% Last note suggests (.*) mode tune\n", "", abc)
    abc = re.sub(r"%%clef (.*)\n", "\n", abc)
    abc = re.sub(r" % \d+ (.)*\n", "\n", abc)
    abc = re.sub(r"(%%MIDI program \d+\n)(?=\1+)", "", abc)
    abc = re.sub(r"T: \n", "", abc)
    abc = re.sub(
        r"(?P<voice>V:\d+\n)(?P<midi>%%MIDI program \d+\n)|(?P<midinovoice>%%MIDI program \d+\n)",
        repl_midi_program,
        abc,
    )
    abc = abc.replace("\n\n", "\n")
    return abc


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


def get_midi_channel_nums(midi_path: Path) -> int:
    mid = mido.MidiFile(midi_path)
    channels = set(
        [
            t.channel
            for t in mid
            if isinstance(t, mido.messages.messages.Message) and hasattr(t, "channel")
        ]
    )
    return len(channels)


def get_midi_channel_nums2(abc_notation: Optional[str]) -> int:
    if abc_notation:
        matches = re.findall(r"V:\d+\n", abc_notation)
        return len(set(matches))
    else:
        return 0
