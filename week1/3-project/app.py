import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur de Churn Client",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .high-risk {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class ChurnPredictorApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Charger le modèle entraîné"""
        try:
            model_data = joblib.load('churn_predictor_model.joblib')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
        except FileNotFoundError:
            st.error("❌ Modèle non trouvé. Veuillez vous assurer que 'churn_predictor_model.joblib' est dans le même répertoire.")
            st.stop()
    
    def preprocess_customer_data(self, customer_data):
        """Prétraiter les données d'un client"""
        customer_df = pd.DataFrame([customer_data])
        
        # Encoder les variables catégorielles
        for col, encoder in self.label_encoders.items():
            if col in customer_df.columns:
                customer_df[col] = encoder.transform(customer_df[col].astype(str))
        
        # Standardiser les features numériques
        numerical_cols = customer_df.select_dtypes(include=[np.number]).columns
        customer_df[numerical_cols] = self.scaler.transform(customer_df[numerical_cols])
        
        # Assurer l'ordre des colonnes
        customer_df = customer_df[self.feature_names]
        
        return customer_df
    
    def predict(self, customer_data):
        """Faire une prédiction"""
        try:
            processed_data = self.preprocess_customer_data(customer_data)
            probability = self.model.predict_proba(processed_data)[0][1]
            prediction = self.model.predict(processed_data)[0]
            
            return {
                'probability': probability,
                'prediction': prediction,
                'risk_level': self.get_risk_level(probability)
            }
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            return None
    
    def get_risk_level(self, probability):
        """Déterminer le niveau de risque"""
        if probability >= 0.7:
            return "Élevé"
        elif probability >= 0.4:
            return "Modéré"
        else:
            return "Faible"
    
    def get_risk_color(self, risk_level):
        """Couleur selon le niveau de risque"""
        colors = {
            "Élevé": "#ff4444",
            "Modéré": "#ffaa00",
            "Faible": "#44ff44"
        }
        return colors.get(risk_level, "#cccccc")

def main():
    # Initialiser l'application
    predictor_app = ChurnPredictorApp()
    
    # En-tête principale
    st.markdown('<h1 class="main-header">🔮 Prédicteur de Churn Client</h1>', unsafe_allow_html=True)
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choisissez le mode",
        ["🏠 Prédiction Unique", "📊 Analyse par Lot", "ℹ️ Aide & Documentation"]
    )
    
    if app_mode == "🏠 Prédiction Unique":
        show_single_prediction(predictor_app)
    elif app_mode == "📊 Analyse par Lot":
        show_batch_analysis(predictor_app)
    else:
        show_documentation()

def show_single_prediction(predictor_app):
    """Interface pour la prédiction unique"""
    
    st.header("📋 Saisie des Informations Client")
    
    # Layout en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations Démographiques")
        gender = st.selectbox("Genre", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partenaire", ["Yes", "No"])
        dependents = st.selectbox("Dépendants", ["Yes", "No"])
        
        st.subheader("Informations de Contrat")
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
        contract = st.selectbox("Type de Contrat", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Facturation Sans Papier", ["Yes", "No"])
        payment_method = st.selectbox("Méthode de Paiement", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Service Téléphonique", ["Yes", "No"])
        multiple_lines = st.selectbox("Lignes Multiples", ["Yes", "No", "No phone service"])
        
        st.subheader("Services Internet")
        internet_service = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Sécurité En Ligne", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Sauvegarde En Ligne", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Protection d'Appareil", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Support Technique", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("TV en Streaming", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Films en Streaming", ["Yes", "No", "No internet service"])
        
        st.subheader("Coûts")
        monthly_charges = st.slider("Charges Mensuelles ($)", 10.0, 120.0, 50.0)
        total_charges = st.slider("Charges Totales ($)", 0.0, 10000.0, 1000.0)
    
    # Bouton de prédiction
    if st.button("🔮 Prédire le Risque de Churn", type="primary", use_container_width=True):
        # Préparer les données
        customer_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Faire la prédiction
        result = predictor_app.predict(customer_data)
        
        if result:
            display_prediction_result(result, customer_data)

def display_prediction_result(result, customer_data):
    """Afficher les résultats de prédiction"""
    
    probability = result['probability']
    risk_level = result['risk_level']
    risk_color = predictor_app.get_risk_color(risk_level)
    
    st.markdown("---")
    st.header("🎯 Résultats de la Prédiction")
    
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Probabilité de Churn",
            value=f"{probability:.1%}",
            delta=f"Niveau {risk_level}" if probability > 0.5 else None,
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Niveau de Risque",
            value=risk_level
        )
    
    with col3:
        prediction_text = "⚠️ Client à Risque" if result['prediction'] == 1 else "✅ Client Fidèle"
        st.metric(
            label="Recommandation",
            value=prediction_text
        )
    
    # Jauge de probabilité
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score de Risque de Churn"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommandations selon le niveau de risque
    st.subheader("💡 Recommandations")
    
    if risk_level == "Élevé":
        st.error("""
        **Actions Immédiates Recommandées:**
        - 📞 Contact proactif dans les 24h
        - 💰 Offre de fidélisation personnalisée
        - 🔍 Analyse des raisons de mécontentement
        - 🎁 Proposition d'avantages immédiats
        """)
    elif risk_level == "Modéré":
        st.warning("""
        **Actions Préventives:**
        - 📧 Email de vérification de satisfaction
        - 🔄 Revue du plan de service
        - 📊 Surveillance accrue
        - 🏆 Programme de fidélité
        """)
    else:
        st.success("""
        **Actions de Fidélisation:**
        - ✅ Maintenance de la satisfaction
        - 🎯 Offres de services additionnels
        - ⭐ Programme de recommandation
        - 📈 Suivi régulier
        """)
    
    # Analyse détaillée
    with st.expander("📊 Analyse Détaillée du Profil"):
        show_detailed_analysis(customer_data, probability)

def show_detailed_analysis(customer_data, probability):
    """Afficher l'analyse détaillée"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Facteurs de Risque")
        
        risk_factors = []
        
        # Analyse des facteurs de risque
        if customer_data['Contract'] == 'Month-to-month':
            risk_factors.append("Contrat mensuel (risque élevé)")
        if customer_data['tenure'] < 12:
            risk_factors.append("Ancienneté < 1 an")
        if customer_data['OnlineSecurity'] == 'No' and customer_data['InternetService'] != 'No':
            risk_factors.append("Pas de sécurité en ligne")
        if customer_data['TechSupport'] == 'No' and customer_data['InternetService'] != 'No':
            risk_factors.append("Pas de support technique")
        if customer_data['PaymentMethod'] == 'Electronic check':
            risk_factors.append("Paiement par chèque électronique")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.write("Aucun facteur de risque majeur identifié")
    
    with col2:
        st.subheader("Indicateurs Clés")
        
        metrics_data = {
            "Ancienneté": f"{customer_data['tenure']} mois",
            "Type de contrat": customer_data['Contract'],
            "Charges mensuelles": f"${customer_data['MonthlyCharges']}",
            "Services internet": customer_data['InternetService'],
            "Support technique": customer_data['TechSupport']
        }
        
        for key, value in metrics_data.items():
            st.write(f"**{key}:** {value}")

def show_batch_analysis(predictor_app):
    """Interface pour l'analyse par lot"""
    
    st.header("📊 Analyse de Churn par Lot")
    
    uploaded_file = st.file_uploader(
        "Téléchargez un fichier CSV avec les données clients",
        type=['csv'],
        help="Le fichier doit contenir les mêmes colonnes que le dataset d'entraînement"
    )
    
    if uploaded_file is not None:
        try:
            # Charger les données
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Fichier chargé avec succès: {len(df)} clients")
            
            # Aperçu des données
            with st.expander("👀 Aperçu des Données"):
                st.dataframe(df.head())
            
            # Prédictions par lot
            if st.button("🎯 Lancer l'Analyse de Churn", type="primary"):
                with st.spinner("Analyse en cours..."):
                    results = batch_predict(predictor_app, df)
                    display_batch_results(results, df)
        
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")

def batch_predict(predictor_app, df):
    """Prédictions par lot"""
    results = []
    
    for _, row in df.iterrows():
        try:
            # Convertir la ligne en dictionnaire
            customer_data = row.to_dict()
            
            # Faire la prédiction
            result = predictor_app.predict(customer_data)
            if result:
                results.append({
                    'customer_id': customer_data.get('customerID', 'N/A'),
                    'churn_probability': result['probability'],
                    'prediction': result['prediction'],
                    'risk_level': result['risk_level']
                })
        except Exception as e:
            st.warning(f"Erreur avec un client: {str(e)}")
    
    return pd.DataFrame(results)

def display_batch_results(results_df, original_df):
    """Afficher les résultats par lot"""
    
    st.header("📈 Résultats de l'Analyse par Lot")
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(results_df)
        st.metric("Clients Analysés", total_customers)
    
    with col2:
        high_risk = len(results_df[results_df['risk_level'] == 'Élevé'])
        st.metric("Risque Élevé", high_risk)
    
    with col3:
        churn_rate = len(results_df[results_df['prediction'] == 1]) / len(results_df)
        st.metric("Taux de Churn Prédit", f"{churn_rate:.1%}")
    
    with col4:
        avg_probability = results_df['churn_probability'].mean()
        st.metric("Probabilité Moyenne", f"{avg_probability:.1%}")
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        # Répartition des risques
        risk_counts = results_df['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Répartition des Niveaux de Risque",
            color=risk_counts.index,
            color_discrete_map={
                'Élevé': '#ff4444',
                'Modéré': '#ffaa00',
                'Faible': '#44ff44'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution des probabilités
        fig = px.histogram(
            results_df,
            x='churn_probability',
            nbins=20,
            title="Distribution des Probabilités de Churn",
            color_discrete_sequence=['#ff4444']
        )
        fig.update_layout(xaxis_title="Probabilité de Churn", yaxis_title="Nombre de Clients")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des résultats
    st.subheader("📋 Détail des Prédictions")
    results_display = results_df.copy()
    results_display['churn_probability'] = results_display['churn_probability'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(results_display, use_container_width=True)
    
    # Téléchargement des résultats
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les Résultats (CSV)",
        data=csv,
        file_name="predictions_churn.csv",
        mime="text/csv"
    )

def show_documentation():
    """Afficher la documentation"""
    
    st.header("📚 Documentation et Aide")
    
    st.markdown("""
    ## 🔍 À Propos de cette Application
    
    Cette application utilise un modèle de Machine Learning (Random Forest) pour prédire 
    la probabilité qu'un client quitte votre entreprise (churn).
    
    ## 🎯 Comment Utiliser
    
    ### Prédiction Unique
    1. Remplissez toutes les informations du client dans le formulaire
    2. Cliquez sur "Prédire le Risque de Churn"
    3. Consultez les résultats et recommandations
    
    ### Analyse par Lot
    1. Préparez un fichier CSV avec les données de vos clients
    2. Téléchargez le fichier dans l'onglet "Analyse par Lot"
    3. Lancez l'analyse et téléchargez les résultats
    
    ## 📊 Interprétation des Résultats
    
    - **Risque Faible** (< 40%) : Client fidèle, actions de fidélisation standard
    - **Risque Modéré** (40-70%) : Surveillance nécessaire, actions préventives
    - **Risque Élevé** (> 70%) : Intervention immédiate requise
    
    ## 🔧 Facteurs Clés Influençant le Churn
    
    Le modèle considère principalement:
    - Ancienneté du client
    - Type de contrat
    - Services souscrits
    - Méthode de paiement
    - Historique des charges
    
    ## 📞 Support
    
    Pour toute question ou problème technique, contactez l'équipe data science.
    """)

if __name__ == "__main__":
    # Initialiser l'application de prédiction
    predictor_app = ChurnPredictorApp()
    main()
