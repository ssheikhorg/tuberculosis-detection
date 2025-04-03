from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input  # New import
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

# Constants (new)
MODEL_INPUT_SIZE = (299, 299)  # InceptionV3 requires 299x299 input


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan handler for model loading"""
    global interpreter
    MODEL_PATH = Path(__file__).parent / "model" / "tb_model.tflite"

    try:
        # Verify model file exists first
        if not MODEL_PATH.exists():
            logger.error(f"❌ Model file not found at {MODEL_PATH}")
            raise HTTPException(status_code=500, detail="Model file not found")

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()

        # Verify input shape
        input_details = interpreter.get_input_details()
        expected_shape = np.array([1, *MODEL_INPUT_SIZE, 3], dtype=np.int32)

        # Proper array comparison
        if not np.array_equal(input_details[0]['shape'], expected_shape):
            error_msg = (f"Model expects shape {input_details[0]['shape']}, "
                         f"but expected {expected_shape.tolist()}")
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        logger.info("✅ Model loaded successfully with TF Lite")
        logger.info(f"Input details: {input_details}")
        yield

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    finally:
        logger.info("🚀 Application ready")

app = FastAPI(
    title="TB Detection API",
    description="API for detecting Tuberculosis from chest X-rays using InceptionV3",
    lifespan=lifespan
)

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))
app.mount("/static", StaticFiles(directory=str(Path(BASE_DIR, "static"))), name="static")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for InceptionV3"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize(MODEL_INPUT_SIZE).convert('RGB')  # Now 299x299
        img_array = np.array(img, dtype=np.float32)

        # Replace manual scaling with InceptionV3's preprocessing
        img_array = preprocess_input(img_array)  # Normalizes to [-1, 1]

        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")


def predict(image_array: np.ndarray) -> dict:
    """Run model prediction (unchanged but now compatible with InceptionV3)"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Verify input shape (debugging)
        if image_array.shape != tuple(input_details[0]['shape']):
            logger.warning(
                f"⚠️ Input shape {image_array.shape} doesn't match model's expected {input_details[0]['shape']}")

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


# The following routes remain unchanged
@app.get("/", include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_api(request: Request, file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file or file.size == 0:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please upload a valid image file"
        })

    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Only JPG, JPEG or PNG files are allowed"
        })

    try:
        contents = await file.read()
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
        "input_size": MODEL_INPUT_SIZE,  # Added for debugging
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