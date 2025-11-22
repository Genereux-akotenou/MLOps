# ------------------------------------------------------- 
# Requirements
# ------------------------------------------------------- 
from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image

# ------------------------------------------------------- 
# App
# ------------------------------------------------------- 
app = FastAPI(
    title="DATA_AFRIQUE_HUB - Cat vs Dog Classifier",
    root_path="/app1/api")

# ------------------------------------------------------- 
# Utils
# ------------------------------------------------------- 
def preprocess(img):
    img = img.resize((150, 150))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img

def load():
    model_path = "notebook/best_model.keras"
    model = load_model(model_path)
    return model

# ------------------------------------------------------- 
# Load the model on app setup
# ------------------------------------------------------- 
model = load()

# ------------------------------------------------------- 
# First route
# ------------------------------------------------------- 
@app.get("/")
def api_info():
    return {"info": "Welcome carapuce"}

@app.get("/health")
def api_health():
    return {"status": "healthy"}

# ------------------------------------------------------- 
# Second route
# ------------------------------------------------------- 
@app.post("/predict")
async def predict(file: UploadFile):
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data))
    img_processed = preprocess(img)
    predictions = model.predict(img_processed)
    print(predictions)
    proba = float(predictions[0][0])
    return {
        "cat_proba": 1 - proba,
        "dog_proba": proba,
        "predict_class": "dog" if proba > 0.5 else "cat"
    }