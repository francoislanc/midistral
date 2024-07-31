def reformat_key(s: str, to_firestore: bool):
    if to_firestore:
        return s.lower().replace(" ", "_")
    else:
        return s.lower().replace("_", " ")
