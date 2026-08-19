"""FastAPI application for DermaGuard AI skin classification."""

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.ai.gradcam import generate_gradcam
from backend.ai.predict import predict_image


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
REPORTS_DIR = PROJECT_ROOT / "reports" / "gradcam_examples"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


class GradCAMResponse(BaseModel):
    prediction: str
    confidence: float
    heatmap: str
    overlay: str


app = FastAPI(title="DermaGuard AI API", version="1.0.0")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/reports",
    StaticFiles(directory=str(REPORTS_DIR)),
    name="reports",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "DermaGuard AI API Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


def validate_upload(file: UploadFile) -> str:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG files are allowed")
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image MIME type")
    return extension


@app.get("/verify-static")
def verify_static():
    return {
        "reports_dir": str(REPORTS_DIR),
        "exists": REPORTS_DIR.exists(),
        "sample_files": [p.name for p in REPORTS_DIR.glob("*.jpg")][:10],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    extension = validate_upload(file)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the 10 MB limit")
        logger.info("Received prediction upload: filename=%s bytes=%d", file.filename, len(contents))
        file_path.write_bytes(contents)
        result = PredictionResponse(**predict_image(file_path))
        logger.info("Prediction complete: filename=%s prediction=%s confidence=%.2f", file.filename, result.prediction, result.confidence)
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed for uploaded file")
        raise HTTPException(status_code=500, detail="Unable to process image") from exc
    finally:
        file_path.unlink(missing_ok=True)


@app.post("/gradcam", response_model=GradCAMResponse)
async def gradcam(file: UploadFile = File(...)) -> GradCAMResponse:
    extension = validate_upload(file)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the 10 MB limit")
        logger.info("Received Grad-CAM upload: filename=%s bytes=%d", file.filename, len(contents))
        file_path.write_bytes(contents)
        result = generate_gradcam(file_path)
        logger.info("Grad-CAM complete: filename=%s prediction=%s confidence=%.2f", file.filename, result["prediction"], result["confidence"])
        return GradCAMResponse(**result)
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Grad-CAM generation failed for uploaded file")
        raise HTTPException(status_code=500, detail="Unable to generate Grad-CAM visualization") from exc
    finally:
        file_path.unlink(missing_ok=True)

@app.on_event("startup")
async def startup_check():
    print("REPORTS_DIR =", REPORTS_DIR)
    print("EXISTS =", REPORTS_DIR.exists())
    print("FILES =", [p.name for p in REPORTS_DIR.glob("*.jpg")][:5])
