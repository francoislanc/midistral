import json
import logging
from pathlib import Path
from typing import Dict

from mistralai import Mistral

from midistral.config import get_settings

OUTPUT_FOLDER = Path(__file__).resolve().parent.parent.parent / "output"


def upload_files(
    chat_train_dataset_path: Path, chat_eval_dataset_path: Path
) -> Dict[str, Dict]:
    mistral_api_key = get_settings().MISTRAL_API_KEY
    client = Mistral(api_key=mistral_api_key)

    with chat_train_dataset_path.open("rb") as f_train, chat_eval_dataset_path.open(
        "rb"
    ) as f_eval:
        train_file_uploaded_info = client.files.upload(
            file={
                "file_name": chat_train_dataset_path.name,
                "content": f_train,
            }
        )
        eval_file_uploaded_info = client.files.upload(
            file={
                "file_name": chat_eval_dataset_path.name,
                "content": f_eval,
            }
        )

        output = {
            "train": train_file_uploaded_info.model_dump(),
            "eval": eval_file_uploaded_info.model_dump(),
        }
        return output


def upload_dataset(train_file: str, eval_file: str, upload_output_file: str):
    train_p = Path(train_file)
    eval_p = Path(eval_file)
    upload_output_path = Path(upload_output_file)
    upload_output_path.parent.mkdir(exist_ok=True, parents=True)
    if train_p.exists() and eval_p.exists():
        output = upload_files(train_p, eval_p)
        with upload_output_path.open("w") as f:
            print(output)
            json.dump(output, f, indent=2)
        logging.info(output)
    else:
        logging.error("Dataset files do not exist")
