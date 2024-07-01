import streamlit as st
from PIL import Image
import requests


st.title("Poubelle Intelligente")

upload = st.file_uploader("Chargez l'image de votre objet",
                           type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
    files = {"file" :  upload.getvalue()}

    req = requests.post("https://fastapi-image-z6t7tex4qa-od.a.run.app/predict", files=files)
    resultat = req.json()
    print(resultat)
    rec = resultat["predictions"]
    prob_recyclable = rec * 100      
    prob_organic = (1-rec)*100

    c1.image(Image.open(upload))
    if prob_recyclable > 50:
        c2.write(f"Je suis certain à {prob_recyclable:.2f} % que l'objet est recyclable")
    else:
        c2.write(f"Je suis certain à {prob_organic:.2f} % que l'objet n'est pas recyclable")