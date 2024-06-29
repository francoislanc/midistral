# Midistral finetuning

This repository contains code to finetune a Mistral model to generate [ABC notation](https://abcnotation.com/) which is a widely adopted system for notating music using plain text.

Midistral is finetuned on the [MidiCaps dataset](https://huggingface.co/datasets/amaai-lab/MidiCaps). It is a large-scale dataset of 168,385 midi music files with descriptive text captions, and a set of extracted musical features.
The MIDI files from the dataset are converted to ABC notation files using [midi2abc](https://github.com/sshlien/abcmidi) executables.


## Notebooks

- [Finetune Mistral model to generate ABC notation](./notebooks/midistral_finetuning.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mlDyNw2NUxxU4tY9f2mYwA2n7YrGKa2C?usp=sharing)


## Server for inference and audio conversion

The generated ABC notation are converted to MIDI file using [abc2midi](https://github.com/sshlien/abcmidi) executables and [timidity](https://doc.ubuntu-fr.org/timidity) is used to convert the MIDI file to OGG audio file.

To start the server : 
```bash
fastapi dev src/midistral/serve.py
```