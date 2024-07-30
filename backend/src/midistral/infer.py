import json
import uuid
from pathlib import Path

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from midistral.config import get_settings
from midistral.midi_utils import clean_generated_abc, get_midi_and_ogg_from_abc
from midistral.storage.gcs import upload_file

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
            return run_inference(monitoring_output["fine_tuned_model"], content)
        else:
            raise Exception("Finetuned model does not exist")


def run_inference(model: str, content: str) -> str:
    client = MistralClient(api_key=get_settings().MISTRAL_API_KEY)
    # print(content)
    chat_response = client.chat(
        model=model,
        temperature=get_settings().LLM_TEMPERATURE,
        messages=[ChatMessage(role="user", content=content)],
        max_tokens=get_settings().LLM_MAX_TOKEN,
    )
    # print(chat_response)
    generated_abc_notation = clean_generated_abc(
        chat_response.choices[0].message.content
    )
    return generated_abc_notation


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


def run_streaming_inference(model: str, content: str) -> None:
    client = MistralClient(api_key=get_settings().MISTRAL_API_KEY)
    for chunk in client.chat_stream(
        model=model,
        temperature=get_settings().LLM_TEMPERATURE,
        messages=[ChatMessage(role="user", content=content)],
        max_tokens=get_settings().LLM_MAX_TOKEN,
    ):
        delta_content = clean_generated_abc(chunk.choices[0].delta.content)
        print(delta_content, end="")
