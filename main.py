from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tensorflow as tf  # Using standard TF Lite
from PIL import Image
import numpy as np
import io
import os
from typing import AsyncIterator, Any
import logging

from mangum import Mangum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize interpreter
interpreter = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan handler for model loading"""
    global interpreter
    MODEL_PATH = Path(__file__).parent / "model" / "tb_model.tflite"

    try:
        # Load TFLite model (using current stable approach)
        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
        logger.info("✅ Model loaded successfully with TF Lite")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")
    finally:
        logger.info("🚀 Application ready")


app = FastAPI(
    title="TB Detection API",
    description="API for detecting Tuberculosis from chest X-rays",
    lifespan=lifespan
)

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))
app.mount("/static", StaticFiles(directory=str(Path(BASE_DIR, "static"))), name="static")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224)).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")


def predict(image_array: np.ndarray) -> dict:
    """Run model prediction"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        confidence = float(prediction)
        return {
            'diagnosis': 'TB Positive' if confidence > 0.5 else 'Normal',
            'confidence': abs(confidence - 0.5) + 0.5,  # Scales to 0.5-1.0 range
            'percentage': f"{max(confidence, 1 - confidence) * 100:.2f}%"
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")


@app.get("/", include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_api(request: Request, file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file exists and has content
    if not file or file.size == 0:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please upload a valid image file"
        })

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Only JPG, JPEG or PNG files are allowed"
        })

    try:
        contents = await file.read()

        # Validate image content
        try:
            img_array = preprocess_image(contents)
        except Exception as e:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": f"Invalid image file: {str(e)}"
            })

        result = predict(img_array)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": result,
            "filename": file.filename
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Processing error: {str(e)}"
        })

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "framework": "tensorflow-lite",
        "version": tf.__version__
    }

def handler(event: Any, context: Any | None = None) -> Any | None:
    if not event.get("requestContext"):
        return None
    mangum = Mangum(app)
    if context:
        return mangum(event, context)
    return None

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)