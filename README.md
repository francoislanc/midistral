# Midistral

This repository contains code :
- [to finetune and serve a finetuned Mistral model](./backend/README.md) to generate [ABC notation](https://abcnotation.com/) (and [MIDI file](https://en.wikipedia.org/wiki/MIDI)). The finetuned model is named Midistral.
- [to interact through a web UI](./frontend/README.md) with the Midistral model 

![midistral-frontend.png](./frontend/docs/midistral-frontend.png)

## Future developments

- Improve model
  - Test other models (ChatMusician)
  - Investigate mood and genre constraints
  - Support more voices/tracks
  - Support more Natural Language text input
  - Explore fill-in-the-middle completion
  
- Improve front-end
  - Allow user generations sharing
