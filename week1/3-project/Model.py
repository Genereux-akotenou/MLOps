import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
<<<<<<< HEAD

    def load_and_preprocess_data(self, file_path):
        """Charger et prétraiter les données"""
        print("Chargement des données...")
        df = pd.read_csv("churn_predictor.csv")

        # Exploration initiale
        print(f"Shape du dataset: {df.shape}")
        print(f"Taux de churn: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")

        # Nettoyage des données
        df_clean = df.copy()

        # Gérer TotalCharges
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'].fillna(0, inplace=True)

        # Target encoding
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

        # Supprimer customerID
        df_clean = df_clean.drop('customerID', axis=1)

        return df_clean

    def feature_engineering(self, df):
        """Ingénierie des caractéristiques"""
        df_fe = df.copy()

        # Créer de nouvelles features
        df_fe['TenureGroup'] = pd.cut(df_fe['tenure'],
                                    bins=[0, 12, 24, 36, 48, 60, 72],
                                    labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5-6'])

        df_fe['ChargeToTenureRatio'] = df_fe['MonthlyCharges'] / (df_fe['tenure'] + 1)
        df_fe['TotalMonthlyRatio'] = df_fe['TotalCharges'] / (df_fe['MonthlyCharges'] + 1)

        # Gérer les valeurs infinies
        df_fe.replace([np.inf, -np.inf], 0, inplace=True)

        return df_fe

=======
        
    def load_and_preprocess_data(self, file_path):
        """Charger et prétraiter les données"""
        print("Chargement des données...")
        df = pd.read_csv("D:\\Mes_Dossiers\\Python\\Projet\\MLOps\\week1\\3-project\\churn_predictor.csv")
        
        # Exploration initiale
        print(f"Shape du dataset: {df.shape}")
        print(f"Taux de churn: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")
        
        # Nettoyage des données
        df_clean = df.copy()
        
        # Gérer TotalCharges
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'].fillna(0, inplace=True)
        
        # Target encoding
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
        # Supprimer customerID
        df_clean = df_clean.drop('customerID', axis=1)
        
        return df_clean
    
    def feature_engineering(self, df):
        """Ingénierie des caractéristiques"""
        df_fe = df.copy()
        
        # Créer de nouvelles features
        df_fe['TenureGroup'] = pd.cut(df_fe['tenure'], 
                                    bins=[0, 12, 24, 36, 48, 60, 72],
                                    labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5-6'])
        
        df_fe['ChargeToTenureRatio'] = df_fe['MonthlyCharges'] / (df_fe['tenure'] + 1)
        df_fe['TotalMonthlyRatio'] = df_fe['TotalCharges'] / (df_fe['MonthlyCharges'] + 1)
        
        # Gérer les valeurs infinies
        df_fe.replace([np.inf, -np.inf], 0, inplace=True)
        
        return df_fe
    
>>>>>>> 1ef05e11 (commit)
    def encode_features(self, df):
        """Encoder les variables catégorielles"""
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
<<<<<<< HEAD

=======
        
>>>>>>> 1ef05e11 (commit)
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
<<<<<<< HEAD

        return df_encoded

=======
        
        return df_encoded
    
>>>>>>> 1ef05e11 (commit)
    def prepare_features(self, df):
        """Préparer les features pour l'entraînement"""
        # Feature engineering
        df_processed = self.feature_engineering(df)
<<<<<<< HEAD

        # Encoding
        df_encoded = self.encode_features(df_processed)

        # Séparer features et target
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']

        self.feature_names = X.columns.tolist()

        return X, y

=======
        
        # Encoding
        df_encoded = self.encode_features(df_processed)
        
        # Séparer features et target
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
>>>>>>> 1ef05e11 (commit)
    def train(self, file_path, test_size=0.2):
        """Entraîner le modèle complet"""
        # Charger et prétraiter les données
        df = self.load_and_preprocess_data(file_path)
<<<<<<< HEAD

        # Préparer les features
        X, y = self.prepare_features(df)

=======
        
        # Préparer les features
        X, y = self.prepare_features(df)
        
>>>>>>> 1ef05e11 (commit)
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
<<<<<<< HEAD

=======
        
>>>>>>> 1ef05e11 (commit)
        # Standardiser les features numériques
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
<<<<<<< HEAD

        print("Entraînement du modèle Random Forest...")
        # Entraîner le modèle
        self.model.fit(X_train, y_train)

        # Évaluation
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"Score entraînement: {train_score:.4f}")
        print(f"Score test: {test_score:.4f}")

        # Prédictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Métriques détaillées
        self.evaluate_model(y_test, y_pred, y_pred_proba)

        # Feature importance
        self.plot_feature_importance()

        return X_test, y_test, y_pred_proba

=======
        
        print("Entraînement du modèle Random Forest...")
        # Entraîner le modèle
        self.model.fit(X_train, y_train)
        
        # Évaluation
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Score entraînement: {train_score:.4f}")
        print(f"Score test: {test_score:.4f}")
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métriques détaillées
        self.evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        self.plot_feature_importance()
        
        return X_test, y_test, y_pred_proba
    
>>>>>>> 1ef05e11 (commit)
    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """Évaluation complète du modèle"""
        print("\n" + "="*50)
        print("ÉVALUATION DU MODÈLE")
        print("="*50)
<<<<<<< HEAD

        # Métriques de base
        auc_score = roc_auc_score(y_true, y_pred_proba)
        print(f"AUC Score: {auc_score:.4f}")

        # Rapport de classification
        print("\nRapport de Classification:")
        print(classification_report(y_true, y_pred, target_names=['Non-Churn', 'Churn']))

        # Matrice de confusion
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
=======
        
        # Métriques de base
        auc_score = roc_auc_score(y_true, y_pred_proba)
        print(f"AUC Score: {auc_score:.4f}")
        
        # Rapport de classification
        print("\nRapport de Classification:")
        print(classification_report(y_true, y_pred, target_names=['Non-Churn', 'Churn']))
        
        # Matrice de confusion
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
>>>>>>> 1ef05e11 (commit)
                   xticklabels=['Non-Churn', 'Churn'],
                   yticklabels=['Non-Churn', 'Churn'])
        plt.title('Matrice de Confusion')
        plt.ylabel('Vérité terrain')
        plt.xlabel('Prédictions')
<<<<<<< HEAD

=======
        
>>>>>>> 1ef05e11 (commit)
        # Courbe ROC
        plt.subplot(1, 2, 2)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aléatoire')
        plt.xlabel('Taux Faux Positifs')
        plt.ylabel('Taux Vrais Positifs')
        plt.title('Courbe ROC')
        plt.legend()
        plt.grid(True, alpha=0.3)
<<<<<<< HEAD

        plt.tight_layout()
        plt.show()

=======
        
        plt.tight_layout()
        plt.show()
        
>>>>>>> 1ef05e11 (commit)
        # Courbe Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Courbe Precision-Recall')
        plt.grid(True, alpha=0.3)
        plt.show()
<<<<<<< HEAD

=======
    
>>>>>>> 1ef05e11 (commit)
    def plot_feature_importance(self, top_n=15):
        """Visualiser l'importance des features"""
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
<<<<<<< HEAD

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(top_n),
=======
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(top_n), 
>>>>>>> 1ef05e11 (commit)
                   x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Features les Plus Importantes\nRandom Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
<<<<<<< HEAD

        print("\nTop 10 Features les Plus Importantes:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

=======
        
        print("\nTop 10 Features les Plus Importantes:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
>>>>>>> 1ef05e11 (commit)
    def business_insights(self, df):
        """Générer des insights business"""
        print("\n" + "="*50)
        print("INSIGHTS BUSINESS")
        print("="*50)
<<<<<<< HEAD

        insights = []

        # Taux de churn global
        churn_rate = df['Churn'].mean()
        insights.append(f"Taux de churn global: {churn_rate:.2%}")

=======
        
        insights = []
        
        # Taux de churn global
        churn_rate = df['Churn'].mean()
        insights.append(f"Taux de churn global: {churn_rate:.2%}")
        
>>>>>>> 1ef05e11 (commit)
        # Analyse par contrat
        if 'Contract' in df.columns:
            contract_churn = df.groupby('Contract')['Churn'].mean()
            insights.append(f"Churn par type de contrat:")
            for contract, rate in contract_churn.items():
                insights.append(f"  - {contract}: {rate:.2%}")
<<<<<<< HEAD

        # Analyse par tenure
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 60, 72],
=======
        
        # Analyse par tenure
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 60, 72], 
>>>>>>> 1ef05e11 (commit)
                                 labels=['0-6m', '6-12m', '1-2a', '2-5a', '5+a'])
        tenure_churn = df.groupby('TenureGroup')['Churn'].mean()
        insights.append(f"Churn par ancienneté:")
        for tenure, rate in tenure_churn.items():
            insights.append(f"  - {tenure}: {rate:.2%}")
<<<<<<< HEAD

=======
        
>>>>>>> 1ef05e11 (commit)
        # Impact des services
        service_columns = ['OnlineSecurity', 'TechSupport', 'OnlineBackup']
        for service in service_columns:
            if service in df.columns:
                service_impact = df.groupby(service)['Churn'].mean()
                insights.append(f"Churn avec {service}: {service_impact.iloc[0]:.2%} vs sans: {service_impact.iloc[1]:.2%}")
<<<<<<< HEAD

        for insight in insights:
            print(insight)

=======
        
        for insight in insights:
            print(insight)
    
>>>>>>> 1ef05e11 (commit)
    def predict_single_customer(self, customer_data):
        """Prédire le churn pour un nouveau client"""
        # Préparer les données
        customer_df = pd.DataFrame([customer_data])
<<<<<<< HEAD

        # Feature engineering
        customer_processed = self.feature_engineering(customer_df)

=======
        
        # Feature engineering
        customer_processed = self.feature_engineering(customer_df)
        
>>>>>>> 1ef05e11 (commit)
        # Encoder
        customer_encoded = customer_processed.copy()
        for col in self.label_encoders:
            if col in customer_encoded.columns:
                customer_encoded[col] = self.label_encoders[col].transform(customer_encoded[col].astype(str))
<<<<<<< HEAD

        # Standardiser
        numerical_cols = customer_encoded.select_dtypes(include=[np.number]).columns
        customer_encoded[numerical_cols] = self.scaler.transform(customer_encoded[numerical_cols])

        # Assurer l'ordre des colonnes
        customer_encoded = customer_encoded[self.feature_names]

        # Prédiction
        churn_probability = self.model.predict_proba(customer_encoded)[0][1]
        churn_prediction = self.model.predict(customer_encoded)[0]

=======
        
        # Standardiser
        numerical_cols = customer_encoded.select_dtypes(include=[np.number]).columns
        customer_encoded[numerical_cols] = self.scaler.transform(customer_encoded[numerical_cols])
        
        # Assurer l'ordre des colonnes
        customer_encoded = customer_encoded[self.feature_names]
        
        # Prédiction
        churn_probability = self.model.predict_proba(customer_encoded)[0][1]
        churn_prediction = self.model.predict(customer_encoded)[0]
        
>>>>>>> 1ef05e11 (commit)
        return {
            'churn_probability': churn_probability,
            'churn_prediction': 'Oui' if churn_prediction == 1 else 'Non',
            'risk_level': 'Élevé' if churn_probability > 0.7 else 'Modéré' if churn_probability > 0.3 else 'Faible'
        }
<<<<<<< HEAD

=======
    
>>>>>>> 1ef05e11 (commit)
    def save_model(self, filepath):
        """Sauvegarder le modèle entraîné"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Modèle sauvegardé: {filepath}")
<<<<<<< HEAD

=======
    
>>>>>>> 1ef05e11 (commit)
    def load_model(self, filepath):
        """Charger un modèle sauvegardé"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print(f"Modèle chargé: {filepath}")

# UTILISATION DU MODÈLE
def main():
    # Initialiser le prédicteur
    predictor = ChurnPredictor()
<<<<<<< HEAD

    # Entraîner le modèle
    X_test, y_test, y_pred_proba = predictor.train('churn_predictor.csv')

    # Insights business
    df_original = predictor.load_and_preprocess_data('churn_predictor.csv')
    predictor.business_insights(df_original)

    # Sauvegarder le modèle
    predictor.save_model('churn_predictor_model.joblib')

    # Exemple de prédiction
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 5,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.0,
        'TotalCharges': 445.0
    }

    prediction = predictor.predict_single_customer(sample_customer)
    print(f"\nPrédiction pour le client exemple:")
    print(f"Probabilité de churn: {prediction['churn_probability']:.2%}")
    print(f"Prédiction: {prediction['churn_prediction']}")
    print(f"Niveau de risque: {prediction['risk_level']}")

if __name__ == "__main__":
    main()
=======
    
    # Entraîner le modèle
    X_test, y_test, y_pred_proba = predictor.train('churn_predictor.csv')
    
    # Insights business
    df_original = predictor.load_and_preprocess_data('churn_predictor.csv')
    predictor.business_insights(df_original)
    
    # Sauvegarder le modèle
 #   predictor.save_model('churn_predictor_model.joblib')
  
if __name__ == "__main__":
    main()

>>>>>>> 1ef05e11 (commit)
