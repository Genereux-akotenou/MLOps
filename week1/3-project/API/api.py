from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import io

app = FastAPI()

@app.get("/")
def api_info():
    return {"info": "Welcome carapuce"}

model = joblib.load("../churn_model.pkl") 
label_encoders = joblib.load('../label_encoders.pkl')

expected_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]


categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

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
def predict_churn(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])

    # Nettoyage numériques
    input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'], errors='coerce').fillna(0)
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0)

    # Forcer catégorielles à str et check
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)
        # Vérif si valeur semble numérique (ex: '0' au lieu de 'Female')
        if input_df[col].str.isnumeric().any():
            return {"error": f"Valeur invalide pour {col} : doit être string comme 'Female' ou 'Male', pas un nombre comme '0'. Vérifie ton input."}

    # Encoder
    for col, le in label_encoders.items():
        if col != 'Churn' and col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError as e:
                return {"error": f"Erreur d'encodage pour {col}: {str(e)}. Assure-toi que les valeurs sont des strings valides (ex: 'Female' pour gender)."}

    # Drop inutiles
    input_df = input_df.drop(columns=['customerID', 'Churn'], errors='ignore')

    # Vérif features
    if list(input_df.columns) != expected_features:
        return {"error": f"Colonnes ne matchent pas : attendu {expected_features}"}

    # Prédiction
    prediction = model.predict(input_df)[0]
    churn_prob = model.predict_proba(input_df)[0][1]

    return {"prediction": "Yes" if prediction == 1 else "No", "probability": churn_prob}

@app.post("/predict_batch")
async def predict_churn_batch(file: UploadFile = File(...)):
    contents = await file.read()
    input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # Nettoyage numériques
    input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'], errors='coerce').fillna(0)
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0)

    # Forcer catégorielles à str et check
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            # Vérif si valeurs semblent numériques
            if input_df[col].str.isnumeric().any():
                return {"error": f"Valeur invalide dans CSV pour {col} : doit être string comme 'Female' ou 'Male', pas un nombre comme '0'. Corrige ton CSV."}

    # Encoder
    for col, le in label_encoders.items():
        if col != 'Churn' and col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError as e:
                return {"error": f"Erreur d'encodage pour {col}: {str(e)}. Assure-toi que les valeurs dans le CSV sont des strings valides (ex: 'Female' pour gender)."}

    # Drop inutiles
    input_df = input_df.drop(columns=['customerID', 'Churn'], errors='ignore')

    # Vérif features
    missing_cols = set(expected_features) - set(input_df.columns)
    extra_cols = set(input_df.columns) - set(expected_features)
    if missing_cols or extra_cols:
        return {"error": f"Manquantes: {missing_cols}. En plus: {extra_cols}. Attendu: {expected_features}"}

    # Prédictions
    predictions = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]

    results = [{"prediction": "Yes" if p == 1 else "No", "probability": prob} for p, prob in zip(predictions, probs)]
    return {"results": results}