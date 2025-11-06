import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

# Chargement des données
df = pd.read_csv('churn_predictor.csv')

# Analyse initiale
print("Dimensions du dataset:", df.shape)
print("\nVariables et types:")
print(df.dtypes)
print("\nValeurs manquantes:")
print(df.isnull().sum())

# Conversion de TotalCharges en numérique (gestion des valeurs vides)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Gestion des valeurs manquantes
df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

# Encodage des variables catégorielles
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                      'PaperlessBilling', 'PaymentMethod', 'Churn']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Séparation des features et target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Vérification de l'équilibre des classes
print("Distribution du churn:")
print(y.value_counts())
print("Proportion:", y.value_counts(normalize=True))

# Création de nouvelles features
X['TenureGroup'] = pd.cut(X['tenure'], bins=[0, 12, 24, 48, np.inf], labels=[0, 1, 2, 3])
X['ChargeRatio'] = X['TotalCharges'] / (X['tenure'] + 1)  # +1 pour éviter division par 0
X['MonthlyToTotalRatio'] = X['MonthlyCharges'] / (X['TotalCharges'] + 1)

# Gestion des valeurs infinies
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Suppression de la colonne tenure originale pour éviter la redondance
X.drop('tenure', axis=1, inplace=True)

# Division train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle avec régularisation et gestion de classe déséquilibrée
logistic_model = LogisticRegression(
    penalty='l2',           # Régularisation L2 pour éviter l'overfitting
    C=0.1,                 # Force de régularisation
    class_weight='balanced', # Gestion du déséquilibre des classes
    random_state=42,
    max_iter=1000,
    solver='liblinear'
)

# Entraînement du modèle
logistic_model.fit(X_train_scaled, y_train)

# Prédictions
y_pred = logistic_model.predict(X_test_scaled)
y_pred_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]

# Grid Search pour optimiser les hyperparamètres
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
    'penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# Meilleur modèle
best_logistic_model = grid_search.best_estimator_
print("Meilleurs paramètres:", grid_search.best_params_)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("=== MATRICE DE CONFUSION ===")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.show()
    
    print("\n=== RAPPORT DE CLASSIFICATION ===")
    print(classification_report(y_test, y_pred))
    
    print("\n=== SCORES ===")
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_score:.4f}")
    
    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC')
    plt.legend()
    plt.show()
    
    return auc_score

# Évaluation du modèle optimisé
auc_score = evaluate_model(best_logistic_model, X_test_scaled, y_test)

# Validation croisée pour robustesse
cv_scores = cross_val_score(
    best_logistic_model, X_train_scaled, y_train, 
    cv=5, scoring='roc_auc', n_jobs=-1
)

print("=== VALIDATION CROISÉE ===")
print(f"Scores AUC par fold: {cv_scores}")
print(f"AUC moyenne: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Importance des features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(best_logistic_model.coef_[0])
}).sort_values('importance', ascending=False)

print("=== IMPORTANCE DES FEATURES ===")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 des Features les plus Importantes')
plt.show()

def predict_churn(customer_data, model, scaler, label_encoders):
    """
    Prédit le churn pour de nouvelles données client
    """
    # Prétraitement des nouvelles données
    customer_processed = preprocess_new_data(customer_data, label_encoders)
    
    # Scaling
    customer_scaled = scaler.transform(customer_processed)
    
    # Prédiction
    churn_probability = model.predict_proba(customer_scaled)[0, 1]
    churn_prediction = model.predict(customer_scaled)[0]
    
    return {
        'churn_probability': churn_probability,
        'churn_prediction': churn_prediction,
        'risk_level': 'Élevé' if churn_probability > 0.7 else 
                     'Modéré' if churn_probability > 0.3 else 'Faible'
    }

def preprocess_new_data(data, label_encoders):
    """
    Prétraite les nouvelles données de la même manière que les données d'entraînement
    """
    # Implémentez ici le même prétraitement que pour les données d'entraînement
    pass


# Sauvegarde des artefacts du modèle
model_artifacts = {
    'model': best_logistic_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': X.columns.tolist()
}

joblib.dump(model_artifacts, 'churn_prediction_model.pkl')
