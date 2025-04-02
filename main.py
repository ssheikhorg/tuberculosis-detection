# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load your trained TensorFlow model
model = keras.models.load_model('models/tb_detection_model.h5')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # Preprocess the image (resize, normalize, etc.)
        img = img.resize((224, 224))
        img = np.array(img) / 255.0  # Normalize pixel values
        img = img.reshape((1, 224, 224, 3))  # Reshape for model input

        # Make a prediction using the loaded model
        prediction = model.predict(img)
        class_label = "TB Positive" if prediction[0][0] > 0.5 else "TB Negative"

        return templates.TemplateResponse("result.html", {
            "request": request,
            "prediction": class_label
        })
    except Exception as e:
        error_message = str(e)
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error_message": error_message
        })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
