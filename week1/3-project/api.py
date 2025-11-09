from fastapi import FastAPI
from fastapi import UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import io


app = FastAPI()

@app.get("/")
def api_info():
    return {"info": "Welcome carapuce"}


model = joblib.load("churn_model.pkl") 
label_encoders = joblib.load('label_encoders.pkl')


expected_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
@app.post("/predict")
async def predict_churn(data: CustomerData):

    # Convertir en DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Encoder les catégories
    for col, le in label_encoders.items():
        if col != 'Churn':  # On n'encode pas Churl
            input_df[col] = le.transform(input_df[col])

    # Prédiction
    prediction = model.predict(input_df)[0]
    churn_prob = model.predict_proba(input_df)[0][1]  # Probabilité de churn

    return {"prediction": "Yes" if prediction == 1 else "No", "probability": churn_prob}




