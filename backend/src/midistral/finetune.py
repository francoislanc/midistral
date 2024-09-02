import json
import time
from pathlib import Path
from pprint import pprint
from typing import Optional

from mistralai import DetailedJobOut, Mistral, WandbIntegration

from midistral.config import get_settings


def run_finetuning(epochs: int, upload_output_file: str, jobs_output_file: str):
    mistral_api_key = get_settings().MISTRAL_API_KEY
    wandb_api_key = get_settings().WANDB_API_KEY
    wandb_project = get_settings().WANDB_PROJECT
    client = Mistral(api_key=mistral_api_key)

    jobs_output_p = Path(jobs_output_file)

    upload_output_p = Path(upload_output_file)
    if upload_output_p.exists():
        with upload_output_p.open("r") as f:
            upload_output = json.load(f)
            training_file_id: str = upload_output["train"]["id"]
            validation_file_id: str = upload_output["eval"]["id"]
            answer = input("Continue? (y/N)\n")
            if answer.lower() in ["y", "yes"]:
                created_jobs = client.fine_tuning.jobs.create(
                    model=get_settings().LLM_BASE_MODEL_FOR_FINETUNING,
                    training_files=[{"file_id": training_file_id, "weight": 1}],
                    validation_files=[validation_file_id],
                    hyperparameters={
                        "epochs": float(epochs),
                        "learning_rate": get_settings().LEARNING_RATE,
                    },
                    integrations=(
                        [
                            WandbIntegration(
                                project=wandb_project,
                                api_key=wandb_api_key,
                            ).model_dump()
                        ]
                        if wandb_project
                        else None
                    ),
                    auto_start=True,
                )
                print(created_jobs)
                with jobs_output_p.open("w") as f:
                    json.dump(created_jobs.model_dump(), f, indent=2)

                # answer = input("Start finetuning? (y/N)\n")
                # if answer.lower() in ["y", "yes"]:
                #     client.fine_tuning.jobs.start(job_id=created_jobs.id)
                #     return created_jobs
                # else:
                #     print("Cancelling")
                #     return None
            else:
                print("Cancelling")
                return None


def monitor_finetuning_job(jobs_output_file: str, monitoring_output_file: str):
    mistral_api_key = get_settings().MISTRAL_API_KEY
    client = Mistral(api_key=mistral_api_key)

    jobs_output_p = Path(jobs_output_file)
    monitoring_output_p = Path(monitoring_output_file)
    if jobs_output_p.exists():
        with jobs_output_p.open("r") as f:
            jobs_output = json.load(f)
            job_id: str = jobs_output["id"]

            retrieved_job: Optional[DetailedJobOut] = client.fine_tuning.jobs.get(
                job_id=job_id
            )
            if retrieved_job:
                while retrieved_job.status in ["RUNNING", "QUEUED", "SUCCESS"]:
                    retrieved_job = client.fine_tuning.jobs.get(job_id=job_id)
                    with monitoring_output_p.open("w") as f:
                        json.dump(retrieved_job.model_dump(), f, indent=2)
                    pprint(retrieved_job)

                    if retrieved_job.status == "SUCCESS":
                        break
                    else:
                        print(f"Job is {retrieved_job.status}, waiting 10 seconds")
                        time.sleep(10)
