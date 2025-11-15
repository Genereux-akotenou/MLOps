import streamlit as st
import requests
import json


st.title("Churn Predictor App - Allo Telecom S.A")


st.write("Entre les détails du client pour prédire s'il va churn (quitter). On va réussir ta certification ! 💪")


gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, value=1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=29.85)
total_charges = st.number_input("Total Charges", min_value=0.0, value=29.85)


if st.button("Prédire le Churn"):
    
    data = {
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
        "TotalCharges": total_charges
    }

    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        result = response.json()

        if "error" in result:
            st.error(f"Erreur de l'API : {result['error']}")
        else:
            st.success(f"Prédiction : {result['prediction']}")
            st.write(f"Probabilité de churn : {result['probability'] * 100:.2f}%")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {str(e)}. Vérifie que l'API est lancée !")


st.write("Pour lancer : `streamlit run app.py` dans ton terminal. Ouvre http://localhost:8501 dans ton navigateur.")