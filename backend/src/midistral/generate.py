import re
from typing import Dict, List

from midistral.midi_utils import get_instrument_number
from midistral.types import AudioTextDescription


def generate_instruction(desc: AudioTextDescription) -> str:
    instructions = []
    if len(desc.genre) > 0:
        instructions.append("genre : " + ", ".join(desc.genre))
    if len(desc.mood) > 0:
        instructions.append("mood : " + ", ".join(desc.mood))
    if len(desc.instruments) > 0:
        instructions.append(
            "instruments : "
            + ", ".join([str(get_instrument_number(i)) for i in desc.instruments])
        )

    return f"""You are a powerful text to ABC music notation model.
Use "%%MIDI program" to set an instrument.
Don't explain anything.

For example:
genre : classical; instruments : 71.

X: 1
M: 2/2
L: 1/8
Q:1/4=100
K:C
V:1
%%MIDI program 71
C4 D2 E2| \
F4 E4-| \
E4 E4| \
F2 G2 A2 F2|
G8| \
c4 B2 c2| \
A4 c4-| \
c4 G3A|
F2 E2 D4| \
C8|

{'; '.join(instructions)}.
"""


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

    return "; ".join(instructions)


def generate_rag_instruction(abc_notations: List[str]) -> str:
    examples = "; ".join(abc_notations)
    return f"""Generate an ABC music notation inspired from these examples.
Output only the generated ABC music notation. 
Keep similar MIDI program.
Don't explain anything.

{examples}
"""


def get_instruction_data(r, with_instrument_num: bool) -> Dict:

    # remove drums
    instrument_summary = [i for i in r["instrument_summary"] if i != "drums"]
    if r["instrument_numbers_sorted"]:
        instrument_numbers_sorted = [
            i for i in r["instrument_numbers_sorted"] if i != 128
        ]
        instrument_numbers_sorted = list(set(instrument_numbers_sorted))
    else:
        instrument_numbers_sorted = None

    desc = AudioTextDescription(
        genre=r["genre"],
        mood=r["mood"],
        instruments=instrument_summary,
        midi_instruments_num=instrument_numbers_sorted,
    )
    text_instruction = generate_instruction_for_finetuned_model(
        desc, with_instrument_num=with_instrument_num
    )
    generation = r["abc_notation"]

    # remove escaped slash (\/) and continuation character (\\\n)
    formatted_generation = re.sub(r"\\/", "/", generation)
    formatted_generation = re.sub(r"\\\n", "", formatted_generation).strip()
    formatted_generation = re.sub(r"\|\n", "| ", formatted_generation).strip()

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
