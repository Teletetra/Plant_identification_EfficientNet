from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.schemas import PredictionResponse
from src.inference.predictor import PlantPredictor

router = APIRouter(prefix="/api/v1", tags=["prediction"])

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PlantPredictor()
    return _predictor

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(content))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        return get_predictor().predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
