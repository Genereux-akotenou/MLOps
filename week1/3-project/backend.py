import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os

# Définir le répertoire où se trouvent les fichiers .pkl
# Le chemin doit être RELATIF au fichier backend.py
# Si backend.py est dans '3-project' et les modèles dans '3-project/notebook', le chemin est 'notebook/'
ARTIFACTS_PATH = "notebook/"

# 1. Charger les artefacts (le modèle, le scaler et les colonnes)
try:
    model = joblib.load(os.path.join(ARTIFACTS_PATH, "best_churn_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
    model_cols = joblib.load(os.path.join(ARTIFACTS_PATH, "model_columns.pkl"))
    print("INFO: Modèles chargés depuis le dossier 'notebook'.") # Message de confirmation
except FileNotFoundError:
    raise RuntimeError(f"Les fichiers .pkl n'ont pas été trouvés dans le chemin: {ARTIFACTS_PATH}. Veuillez vérifier le répertoire.")

# 2. Définir l'application FastAPI
app = FastAPI()

# 3. Définir le schéma des données d'entrée (Pydantic)
# Pour une implémentation complète, tous les champs de votre X_train devraient être listés ici.
# Voici un exemple basé sur les colonnes originales pour simplifier.
class ChurnPredictionData(BaseModel):
    # Démographie
    gender: str = 'Female' # 'Female', 'Male'
    SeniorCitizen: int = 0 # 0, 1
    Partner: str = 'Yes'   # 'Yes', 'No'
    Dependents: str = 'No'   # 'Yes', 'No'

    # Compte et Charges
    tenure: int = 1        
    Contract: str = 'Month-to-month' # 'Month-to-month', 'One year', 'Two year'
    PaperlessBilling: str = 'Yes' # 'Yes', 'No'
    PaymentMethod: str = 'Electronic check' 
    MonthlyCharges: float = 70.0  
    TotalCharges: float = 70.0    

    # Services
    PhoneService: str = 'Yes' # 'Yes', 'No'
    MultipleLines: str = 'No' # 'Yes', 'No', 'No phone service'
    InternetService: str = 'Fiber optic' # 'DSL', 'Fiber optic', 'No'
    OnlineSecurity: str = 'No' # 'Yes', 'No', 'No internet service'
    OnlineBackup: str = 'Yes'  # 'Yes', 'No', 'No internet service'
    DeviceProtection: str = 'No' # 'Yes', 'No', 'No internet service'
    TechSupport: str = 'No'  # 'Yes', 'No', 'No internet service'
    StreamingTV: str = 'No'  # 'Yes', 'No', 'No internet service'
    StreamingMovies: str = 'No' # 'Yes', 'No', 'No internet service'

# 4. Point de terminaison (Endpoint) de prédiction
@app.post("/predict_churn")
def predict_churn(data: ChurnPredictionData):
    # 4.1. Convertir les données Pydantic en DataFrame
    df = pd.DataFrame([data.dict()])

    # 4.2. PRÉ-TRAITEMENT EXACTEMENT COMME DANS LE JUPYTER NOTEBOOK

    # a) Encodage binaire simple (gender, Yes/No columns)
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    binary_map = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # b) One-Hot Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # c) Harmonisation des colonnes (crucial pour l'API)
    # Créer un DataFrame vide avec les colonnes du modèle entraîné
    final_features = pd.DataFrame(columns=model_cols)
    # Remplir les colonnes existantes avec les valeurs encodées
    final_features = final_features.merge(df_encoded, how='outer').fillna(0)
    # S'assurer que les colonnes sont dans le bon ordre et la bonne taille
    final_features = final_features[model_cols]

    # d) Mise à l'échelle (StandardScaler)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # S'assurer que seules les colonnes numériques sont passées au scaler, et qu'elles existent
    X_scaled = final_features.copy()
    if all(col in X_scaled.columns for col in numerical_cols):
        X_scaled[numerical_cols] = scaler.transform(X_scaled[numerical_cols])

    # 4.3. Prédiction
    prediction_proba = model.predict_proba(X_scaled)[0][1]

    return {
        "churn_probability": round(prediction_proba, 4),
        "prediction": "Churn" if prediction_proba >= 0.5 else "No Churn"
    }