from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from midistral.config import get_settings
from midistral.db import schemas
from midistral.db.firestore import crud as firestore_crud
from midistral.db.sqlite import crud as sqlite_crud
from midistral.db.sqlite import models
from midistral.db.sqlite.database import engine
from midistral.infer import generate_midi_and_ogg_audio, get_path, run_inference
from midistral.prepare_dataset import generate_instruction
from midistral.storage.gcs import download_file
from midistral.types import AudioTextDescription

if not get_settings().USE_FIRESTORE_DB:
    models.Base.metadata.create_all(bind=engine)


def get_custom_ipaddr(request: Request) -> str:
    if "X_FORWARDED_FOR" in request.headers:
        return request.headers["X_FORWARDED_FOR"]
    elif "X-FORWARDED-FOR" in request.headers:
        return request.headers["X-FORWARDED-FOR"]
    else:
        if not request.client or not request.client.host:
            return "127.0.0.1"

        return request.client.host


limiter = Limiter(key_func=get_custom_ipaddr)
app = FastAPI()
origins = [get_settings().FRONT_END_ORIGIN]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def get_crud():
    if get_settings().USE_FIRESTORE_DB:
        return firestore_crud
    else:
        return sqlite_crud


def generate_abc_notation(desc: AudioTextDescription):
    text_description = generate_instruction(desc)

    if len(text_description) > 0:
        abc_notation_text = run_inference(
            get_settings().FINETUNED_MODEL_NAME, text_description
        )
        # abc_notation_text = "X: 1\nM: 4/4\nL: 1/8\nQ:1/4=120\nK:D\nV:1\n%%MIDI program 0\n G/2G/2c/2A/2| B/2B/2d/2G/2| A/2A/2F/2G/2| B/2B/2d/2G/2|G/2G/2c/2A/2| B/2B/2d/2G/2| A/2A/2F/2G/2| B/2B/2d/2G/2|G/2G/2c/2A/2| B/2B/2d/2G/2| A/2A/2F/2G/2| B/2B/2d/2G/2| B/2B/2d/2G/2| A/2A/2F/2G/2| B/2B/2d/2G/2| B/2B/2d/2G/2| A/2A/2F/2G/2| B/2B/2d/2G/2|\n"
        file_uuid = generate_midi_and_ogg_audio(abc_notation_text)
        abc_generation = schemas.AbcGenerationCreate(
            text_description=text_description,
            abc_notation=abc_notation_text,
            file_uuid=file_uuid,
        )

        db_abc_generation = get_crud().create_abc_generation(
            abc_generation=abc_generation
        )

        return {
            "id": db_abc_generation.id,
            "file_uuid": file_uuid,
            "abc_notation": abc_notation_text,
        }
    else:
        raise HTTPException(
            status_code=400, detail="Generation failed : no selected items"
        )


@app.post("/generate")
@limiter.limit("5/minute")
async def limited_generate_abc_notation(request: Request):
    req_obj = await request.json()
    des = AudioTextDescription.model_validate(req_obj)
    des.filter_values()
    return generate_abc_notation(des)


@app.post("/feedback")
@limiter.limit("5/minute")
async def post_feedback(request: Request):
    id: UUID = UUID(request.query_params.get("id"))
    liked: bool = request.query_params.get("liked") == "true"
    db_abc_generation = get_crud().get_abc_generation(id)
    if db_abc_generation is None:
        raise HTTPException(status_code=404, detail="AbcGeneration not found")
    else:
        get_crud().update_feedback_abc_generation(id, liked)


@app.get("/file/{filename}")
@limiter.limit("5/minute")
async def get_file(request: Request):
    filename: str = request.path_params["filename"]
    id, extension = filename.split(".")
    p = get_path(id, extension)
    if p.exists():
        return FileResponse(p)
    elif get_settings().GCP_PROJECT:
        if get_settings().GCP_PROJECT:
            download_file(p)
            return FileResponse(p)
    else:
        raise HTTPException(status_code=404, detail="File not found")
