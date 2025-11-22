import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Visualisation des données avec Streamlit')
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu des données")
    st.write(df.head())

    x_col = st.selectbox("Choisissez la colonne pour l'axe X", df.columns)
    y_col = st.selectbox("Choisissez la colonne pour l'axe Y", df.columns)


    st.write(f"Graphique de {x_col} vs {y_col}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    st.pyplot(plt)
