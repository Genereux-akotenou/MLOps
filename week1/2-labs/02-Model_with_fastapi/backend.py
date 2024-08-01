# ------------------------------------------------------- 
# Requirements
# ------------------------------------------------------- 
from fastapi import FastAPI, UploadFile
#from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image
import joblib

# ------------------------------------------------------- 
# App
# ------------------------------------------------------- 
app = FastAPI()

# ------------------------------------------------------- 
# Utils
# ------------------------------------------------------- 
def preprocess(img):
    img = img.resize((150, 150))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img

def load():
    model_path = "notebook/best_model.pkl"
    model = joblib.load(model_path)
    #model = load_model(model_path)
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

# ------------------------------------------------------- 
# Second route
# ------------------------------------------------------- 
class Params(BaseModel):
    age: int
    weight: float
    height: float
    
@app.post("/predict")
async def predict(params: Params):
#async def predict(file: UploadFile):
#    image_data = await file.read()
    #img = Image.open(io.BytesIO(image_data))
    #img_processed = preprocess(img)
    predictions = model.predict(params)
    print(predictions)
    proba = float(predictions[0][0])
    return {"churn": predictions}
    return {
        "cat_proba": 1 - proba,
        "dog_proba": proba,
        "predict_class": "dog" if proba > 0.5 else "cat"
    }