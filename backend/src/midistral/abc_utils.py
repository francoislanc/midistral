import re


def count_max_opened_parentheses(s: str) -> int:
    current_count = 0
    max_count = 0
    for char in s:
        if char == "(":
            current_count += 1
            if current_count > max_count:
                max_count = current_count
        elif char == ")":
            current_count -= 1
    return max_count


def has_only_silence(abc_notation: str) -> bool:
    s = re.sub(r"(.*):(.*)", "", abc_notation)
    s = re.sub(r"%%(.*)", "", s)
    s = s.lower().strip()
    if "a" in s or "b" in s or "c" in s or "d" in s or "e" in s or "f" in s or "g" in s:
        return False
    else:
        return True


def repl_midi_program(match_obj: str) -> str:
    if match_obj.group(1):
        return match_obj.group(1) + match_obj.group(2)
    elif match_obj.group(3):
        return ""


def clean_generated_abc(abc: str) -> str:
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
