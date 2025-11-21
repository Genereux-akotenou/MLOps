import streamlit as st
import requests
import json
import pandas as pd

# 1. Configuration de la page Streamlit
st.set_page_config(layout="wide")

# URL de votre API FastAPI (assurez-vous qu'elle tourne sur le port 8000)
FASTAPI_URL = "http://127.0.0.1:8001/predict_churn"

st.title("💡 Allo Telecom S.A : Outil de Prédiction d'Attrition Client (Churn)")
st.markdown("---")

# 2. Définition des options pour les listes déroulantes
# Ces options doivent correspondre EXACTEMENT aux valeurs attendues par votre modèle encodé !
OPTIONS = {
    # Démographie
    "gender": ['Female', 'Male'],
    "SeniorCitizen": [0, 1],
    "Partner": ['Yes', 'No'],
    "Dependents": ['Yes', 'No'],
    
    # Compte et Services
    "PhoneService": ['Yes', 'No'],
    "MultipleLines": ['No phone service', 'No', 'Yes'],
    "InternetService": ['DSL', 'Fiber optic', 'No'],
    "OnlineSecurity": ['No internet service', 'No', 'Yes'],
    "OnlineBackup": ['No internet service', 'No', 'Yes'],
    "DeviceProtection": ['No internet service', 'No', 'Yes'],
    "TechSupport": ['No internet service', 'No', 'Yes'],
    "StreamingTV": ['No internet service', 'No', 'Yes'],
    "StreamingMovies": ['No internet service', 'No', 'Yes'],
    
    # Contrat
    "Contract": ['Month-to-month', 'One year', 'Two year'],
    "PaperlessBilling": ['Yes', 'No'],
    "PaymentMethod": ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

# 3. Création du formulaire Streamlit avec des colonnes
with st.form(key='churn_form'):
    st.header("Informations du Client")
    
    # Utilisation de st.columns pour une mise en page plus claire
    col1, col2, col3 = st.columns(3)
    
    # Colonne 1 : Démographie et Ancienneté
    with col1:
        st.subheader("👤 Démographie & Ancienneté")
        gender = st.selectbox("Sexe", OPTIONS['gender'])
        senior_citizen = st.selectbox("Citoyen Senior (1=Oui)", OPTIONS['SeniorCitizen'])
        partner = st.selectbox("Partenaire", OPTIONS['Partner'])
        dependents = st.selectbox("Personnes à charge", OPTIONS['Dependents'])
        tenure = st.slider("Ancienneté (mois)", min_value=1, max_value=72, value=12)

    # Colonne 2 : Services et Internet
    with col2:
        st.subheader("📞 Services Souscrits")
        phone_service = st.selectbox("Service Téléphonique", OPTIONS['PhoneService'])
        multiple_lines = st.selectbox("Lignes Multiples", OPTIONS['MultipleLines'])
        internet_service = st.selectbox("Service Internet", OPTIONS['InternetService'])
        
        st.markdown("---")
        st.subheader("🛡️ Sécurité Internet")
        online_security = st.selectbox("Sécurité en ligne", OPTIONS['OnlineSecurity'])
        online_backup = st.selectbox("Sauvegarde en ligne", OPTIONS['OnlineBackup'])
        device_protection = st.selectbox("Protection Appareil", OPTIONS['DeviceProtection'])
        tech_support = st.selectbox("Support Technique", OPTIONS['TechSupport'])

    # Colonne 3 : Streaming et Facturation
    with col3:
        st.subheader("📺 Streaming & Facturation")
        streaming_tv = st.selectbox("Streaming TV", OPTIONS['StreamingTV'])
        streaming_movies = st.selectbox("Streaming Films", OPTIONS['StreamingMovies'])
        
        st.markdown("---")
        st.subheader("💳 Compte & Charges")
        contract = st.selectbox("Contrat", OPTIONS['Contract'])
        payment_method = st.selectbox("Méthode de Paiement", OPTIONS['PaymentMethod'])
        paperless_billing = st.selectbox("Facturation Sans Papier", OPTIONS['PaperlessBilling'])
        
        monthly_charges = st.number_input("Charges Mensuelles (€)", min_value=18.0, max_value=150.0, value=70.0, step=0.01)
        total_charges = st.number_input("Charges Totales (€)", min_value=18.0, max_value=10000.0, value=70.0, step=0.01)
        
    st.markdown("---")
    submit_button = st.form_submit_button(label='Prédire l\'Attrition (Churn)')

# 4. Traitement de l'envoi du formulaire
if submit_button:
    # Construire l'objet de données EXACTEMENT comme la classe Pydantic l'attend
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    
    st.write("Requête envoyée :", input_data)
    
    # Envoi de la requête à l'API FastAPI
    try:
        response = requests.post(FASTAPI_URL, json=input_data)
        response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)

        result = response.json()
        prob = result['churn_probability']
        pred = result['prediction']
        
        st.subheader("Résultat de la Prédiction")

        if pred == "Churn":
            st.error(f"🔴 PRÉDICTION : Le client est à HAUT RISQUE d'attrition.")
            st.write(f"Probabilité d'Attrition : **{prob * 100:.2f}%**")
        else:
            st.success(f"🟢 PRÉDICTION : Le client est à faible risque d'attrition.")
            st.write(f"Probabilité d'Attrition : **{prob * 100:.2f}%**")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API FastAPI. Assurez-vous que l'API est lancée via `uvicorn backend:app --reload`. Détail de l'erreur : {e}")
    except json.JSONDecodeError:
        st.error("Erreur lors de la lecture de la réponse de l'API. Vérifiez les logs FastAPI.")