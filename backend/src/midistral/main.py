import logging

import fire
from dotenv import load_dotenv

from midistral.finetune import monitor_finetuning_job, run_finetuning
from midistral.infer import run_inference_after_finetune, run_streaming_inference
from midistral.prepare_dataset import prepare_dataset
from midistral.upload_dataset import upload_dataset

logging.basicConfig(
    filename="logs.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s -> %(levelname)s: %(message)s",
)


load_dotenv()


def run() -> None:
    fire.Fire(
        {
            "prepare_dataset": prepare_dataset,
            "upload_dataset": upload_dataset,
            "finetune": run_finetuning,
            "monitor_finetuning_job": monitor_finetuning_job,
            "infer": run_inference_after_finetune,
            "streaming_infer": run_streaming_inference,
        }
    )


if __name__ == "__main__":
    run()
