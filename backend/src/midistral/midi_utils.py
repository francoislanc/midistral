import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mido
import numpy as np
from music21 import midi as music21midi

from midistral.abc_utils import clean_generated_abc


def convert_midi_to_ogg(midi_path: Path, ogg_path: Path):
    cmd = [
        "timidity",
        str(midi_path.resolve()),
        "-Ov",
        "-o",
        str(ogg_path.resolve()),
    ]
    subprocess.run(cmd, capture_output=True)


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
    midi = mido.MidiFile(midi_path)
    num = 0
    for t in midi.tracks:
        for m in t:
            if (
                isinstance(m, mido.messages.messages.Message)
                and m.type == "program_change"
            ):
                num += 1
    return num


def get_duration(midi_path: Path) -> int:
    try:
        midi = mido.MidiFile(midi_path)
        return int(midi.length)
    except Exception:
        return -1


def get_duration_caps(duration: Optional[int]) -> str:
    if duration:
        dur_marks = np.array((30, 120, 300))
        dur_caps = ["Short fragment", "Short song", "Song", "Long piece"]
        index = int(np.sum(duration > dur_marks))
        return dur_caps[index]
    else:
        return ""


def get_tempo(midi_path: Path) -> Optional[int]:
    midi = mido.MidiFile(midi_path)
    try:
        for msg in midi:
            if msg.type == "set_tempo":
                tempo = mido.tempo2bpm(msg.tempo)
                return int(tempo)
    except Exception:
        return None
    return None


def get_tempo_caps(tempo: Optional[int]) -> str:
    if tempo:
        tempo_marks = np.array((80, 120, 160))
        tempo_caps = ["Slow", "Moderate tempo", "Fast", "Very fast"]
        index = int(np.sum(tempo > tempo_marks))
        return tempo_caps[index]
    else:
        return ""


def read_midi(midi_path: Path):
    mf = music21midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    return music21midi.translate.midiFileToStream(mf)


def get_key(midi_path: Path) -> Optional[str]:
    midi = read_midi(midi_path)
    # key
    try:
        res_key = midi.analyze("keys")
        key = res_key.tonic.name + " " + res_key.mode
    except Exception:
        key = None
    # key postprocessing
    if "-" in key:
        key = key.replace("-", "b")

    return key


def get_time_signature(midi_path: Path) -> Optional[str]:
    midi = read_midi(midi_path)

    try:
        time_sig = midi.getTimeSignatures()[0]
        time_signature = str(time_sig.numerator) + "/" + str(time_sig.denominator)
    except Exception:
        time_signature = None

    return time_signature


def get_instruments(midi_file_path: Path):
    # Dictionary to store instrument durations
    instrument_durations = defaultdict(float)
    instrument_names = []
    instrument_channels = []
    instrument_change_times = []
    # Parse MIDI file
    midi = mido.MidiFile(midi_file_path)
    # Iterate through each track
    for track in midi.tracks:
        # Dictionary to store note-on events for each instrument
        active_notes = defaultdict(float)
        last_event_time = 0
        # Iterate through each event in the track
        for msg in track:
            # Update the time since the last event
            delta_time = msg.time
            last_event_time += delta_time
            if msg.type == "program_change":
                prog = msg.program
                chan = msg.channel
                # if chan==9 and prog==0:
                if chan == 9 and not 111 < prog < 120:
                    prog = 128
                if chan in instrument_channels:
                    instrument_names[instrument_channels.index(chan)] = (
                        prog  # replace the existing instrument in this channel, no ambiguity!!!
                    )
                else:
                    instrument_names.append(prog)
                    instrument_channels.append(chan)
                instrument_change_times.append(msg.time)
                # print(msg.time)
            # If it's a note-on or note-off event
            if msg.type == "note_on" or msg.type == "note_off":
                # Extract the instrument (channel) and note number
                channel = msg.channel
                note = msg.note
                # Calculate the duration since the last event
                duration = last_event_time - active_notes[(channel, note)]
                active_notes[(channel, note)] = last_event_time
                # Accumulate the duration for this instrument
                instrument_durations[channel] += duration
    new_dict = sorted(instrument_durations.items(), key=lambda x: x[1], reverse=True)
    if len(instrument_names) > 20:
        print("too many instruments in this one!")
        print(midi_file_path)
        return [], []
    sorted_instrument_list = []
    how_many = min(5, len(set(instrument_names)))
    if how_many == 0:
        return [], []
    add_drums = False
    if 9 not in instrument_channels:
        for rr in new_dict:
            if 9 in rr:
                add_drums = True
                break
            else:
                add_drums = False
    if add_drums:
        instrument_names.append(128)
        instrument_channels.append(9)
    for inst in new_dict:
        try:
            sorted_instrument_list.append(
                instrument_names[instrument_channels.index(inst[0])]
            )
        except Exception as e:
            print(e)
            print(midi_file_path)
            return sorted_instrument_list, []

    out_inst_list = []
    for inst in sorted_instrument_list:
        out_inst_list.append(get_instrument_name(inst))
    # instruments summary - only add one instance of each instrument, then keep top 5
    out_inst_sum_list = []
    for rr in out_inst_list:
        if rr not in out_inst_sum_list:
            out_inst_sum_list.append(rr)
    how_many = np.min((5, len(out_inst_sum_list)))
    instruments_summary = out_inst_sum_list[0:how_many]

    return sorted_instrument_list, instruments_summary


instruments_mapping = {
    0: "Piano",
    1: "Piano",
    2: "Electric Piano",
    3: "Honky-tonk Piano",
    4: "Piano",
    5: "Piano",
    6: "Harpsichord",
    7: "Clavinet",
    8: "Celesta",
    9: "Glockenspiel",
    10: "Music box",
    11: "Vibraphone",
    12: "Marimba",
    13: "Xylophone",
    14: "Tubular Bells",
    15: "Dulcimer",
    16: "Hammond Organ",
    17: "Percussive Organ",
    18: "Rock Organ",
    19: "Church Organ",
    20: "Reed Organ",
    21: "Accordion",
    22: "Harmonica",
    23: "Tango Accordion",
    24: "Acoustic Guitar",
    25: "Acoustic Guitar",
    26: "Electric Guitar",
    27: "Clean Electric Guitar",
    28: "Electric Guitar",
    29: "Overdriven Guitar",
    30: "Distortion Guitar",
    31: "Guitar Harmonics",
    32: "Acoustic Bass",
    33: "Electric Bass",
    34: "Electric Bass",
    35: "Fretless Bass",
    36: "Slap Bass",
    37: "Slap Bass",
    38: "Synth Bass",
    39: "Synth Bass",
    40: "Violin",
    41: "Viola",
    42: "Cello",
    43: "Contrabass",
    44: "Tremolo Strings",
    45: "Pizzicato Strings",
    46: "Orchestral Harp",
    47: "Timpani",
    48: "String Ensemble",
    49: "String Ensemble",
    50: "Synth Strings",
    51: "Synth Strings",
    52: "Choir Aahs",
    53: "Voice Oohs",
    54: "Synth Voice",
    55: "Orchestra Hit",
    56: "Trumpet",
    57: "Trombone",
    58: "Tuba",
    59: "Muted Trumpet",
    60: "French Horn",
    61: "Brass Section",
    62: "Synth Brass",
    63: "Synth Brass",
    64: "Soprano Saxophone",
    65: "Alto Saxophone",
    66: "Tenor Saxophone",
    67: "Baritone Saxophone",
    68: "Oboe",
    69: "English Horn",
    70: "Bassoon",
    71: "Clarinet",
    72: "Piccolo",
    73: "Flute",
    74: "Recorder",
    75: "Pan Flute",
    76: "Bottle Blow",
    77: "Shakuhachi",
    78: "Whistle",
    79: "Ocarina",
    80: "Synth Lead",
    81: "Synth Lead",
    82: "Calliope Lead",
    83: "Chiffer Lead",
    84: "Charang Lead",
    85: "Synth Voice",
    86: "Synth Lead",
    87: "Brass Lead",
    88: "Synth Pad",
    89: "Synth Pad",
    90: "Synth Pad",
    91: "Synth Pad",
    92: "Synth Pad",
    93: "Synth Pad",
    94: "Synth Pad",
    95: "Synth Pad",
    96: "Synth Effects",
    97: "Synth Effects",
    98: "Synth Effects",
    99: "Synth Effects",
    100: "Synth Effects",
    101: "Synth Effects",
    102: "Synth Effects",
    103: "Synth Effects",
    104: "Sitar",
    105: "Banjo",
    106: "Shamisen",
    107: "Koto",
    108: "Kalimba",
    109: "Bagpipe",
    110: "Fiddle",
    111: "Shana",
    112: "Tinkle Bell",
    113: "Agogo",
    114: "Steel Drums",
    115: "Woodblock",
    116: "Taiko Drum",
    117: "Melodic Tom",
    118: "Synth Drum",
    119: "Reverse Cymbal",
    120: "Guitar Fret Noise",
    121: "Breath Noise",
    122: "Seashore",
    123: "Bird Tweet",
    124: "Telephone Ring",
    125: "Helicopter",
    126: "Applause",
    127: "Gunshot",
    128: "Drums",
}

instrument_numbers_mapping = dict(
    [(i.lower(), n) for n, i in instruments_mapping.items()]
)
# default piano with 0
instrument_numbers_mapping["piano"] = 0
instrument_numbers_mapping["synth lead"] = 81


def get_instrument_name(number: int) -> Optional[str]:

    if number in instruments_mapping:
        return instruments_mapping[number].lower()
    else:
        return None


def get_instrument_number(name: str) -> Optional[int]:

    if name.lower() in instrument_numbers_mapping:
        return instrument_numbers_mapping[name.lower()]
    else:
        return None
