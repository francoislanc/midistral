import json
import random
import uuid
from pathlib import Path

from mistralai import Mistral

from midistral.abc_utils import clean_generated_abc
from midistral.config import get_settings
from midistral.db.crud import get_crud
from midistral.generate import (
    generate_instruction,
    generate_instruction_for_finetuned_model,
    generate_rag_instruction,
)
from midistral.midi_utils import get_midi_and_ogg_from_abc
from midistral.storage.gcs import upload_file
from midistral.types import AudioTextDescription, InferenceApproach

OUTPUT_FOLDER = Path(__file__).resolve().parent.parent.parent / "output"


def get_path(id: str, extension: str) -> Path:
    p = OUTPUT_FOLDER / "audio" / "llm-generated" / f"{id}.{extension}"
    return p


def run_inference_after_finetune(monitoring_output_file: str, content: str) -> str:
    monitoring_output_p = Path(monitoring_output_file)
    with monitoring_output_p.open("r") as f:
        monitoring_output = json.load(f)
        if (
            monitoring_output["fine_tuned_model"]
            and monitoring_output["status"] == "SUCCESS"
        ):
            return run_inference(
                monitoring_output["fine_tuned_model"], json.dumps(content)
            )
        else:
            raise Exception("Finetuned model does not exist")


def run_inference(model: str, content: str) -> str:
    client = Mistral(api_key=get_settings().MISTRAL_API_KEY)
    # print(content)
    chat_response = client.chat.complete(
        model=model,
        temperature=get_settings().LLM_TEMPERATURE,
        messages=[{"role": "user", "content": content}],
        max_tokens=get_settings().LLM_MAX_TOKEN,
    )
    # print(chat_response)
    generated_abc_notation = clean_generated_abc(
        chat_response.choices[0].message.content
    )
    return generated_abc_notation


def generate_abc_notation(
    desc: AudioTextDescription, approach: InferenceApproach
) -> str:

    if approach == InferenceApproach.PROMPT_ONLY:
        text_description = generate_instruction(desc)
        model = get_settings().PROMPT_MODEL_NAME
        abc_notation_text = run_inference(model, text_description)
    elif approach == InferenceApproach.RAG:
        annotated_abcs = get_crud().get_annotated_abcs_from_description(
            desc, get_settings().DB_LIMIT
        )
        random.shuffle(annotated_abcs)
        abc_notations = [
            e.abc_notation for e in annotated_abcs[: get_settings().RETRIEVED_LIMIT]
        ]
        text_description = generate_rag_instruction(abc_notations)
        model = get_settings().RAG_MODEL_NAME
        abc_notation_text = run_inference(model, text_description)
    elif approach == InferenceApproach.FINETUNED:
        text_description = generate_instruction_for_finetuned_model(
            desc, with_instrument_num=True
        )
        model = get_settings().FINETUNED_MODEL_NAME
        abc_notation_text = run_inference(model, text_description)
    else:
        raise NotImplementedError(f"Approach '{approach}' not yet supported")
    return abc_notation_text, text_description


def generate_midi_and_ogg_audio(abc_notation: str) -> id:
    file_uuid = str(uuid.uuid4())

    midi, ogg = get_midi_and_ogg_from_abc(abc_notation)
    for extension, b in [("midi", midi), ("ogg", ogg)]:
        p = get_path(file_uuid, extension)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write(b)

        if get_settings().GCP_PROJECT:
            upload_file(p)
    return file_uuid
