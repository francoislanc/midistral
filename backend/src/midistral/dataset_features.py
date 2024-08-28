from datasets import Features, Sequence, Value

midi_abc_features = Features(
    {
        "origin": Value("string"),
        "location": Value("string"),
        "genre": Sequence(Value("string")),
        "mood": Sequence(Value("string")),
        "key": Value("string"),
        "time_signature": Value("string"),
        "tempo": Value("int32"),
        "duration": Value("int32"),
        "instrument_summary": Sequence(Value("string")),
        "instrument_numbers_sorted": Sequence(Value("int32")),
        "chord_summary": Sequence(Value("string")),
        "midi_tracks_nums": Value("int32"),
        "abc_notation": Value("large_string"),
    }
)

midi_features = Features(
    {
        "location": Value("string"),
        "genre": Sequence(Value("string")),
        "mood": Sequence(Value("string")),
        "key": Value("string"),
        "time_signature": Value("string"),
        "tempo": Value("int32"),
        "duration": Value("int32"),
        "instrument_summary": Sequence(Value("string")),
        "instrument_numbers_sorted": Sequence(Value("int32")),
        "chord_summary": Sequence(Value("string")),
        "origin": Value("string"),
    }
)
