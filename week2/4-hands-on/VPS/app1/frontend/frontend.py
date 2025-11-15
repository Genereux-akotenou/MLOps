import streamlit as st
from PIL import Image
import requests

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon=":cat: :dog:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Cat vs Dog Classifier :cat: :dog:")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #fff;
        border-right: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# st.markdown(
#     """
#     <style>
#     /* Full app background */
#     .stApp {
#         background-color: #fff9c4;  /* Light Yellow */
#     }

#     /* Main content container */
#     .main {
#         background-color: #fff9c4; /* Same Yellow for consistency */
#         padding: 2rem;
#         border-radius: 1rem;
#     }

#     /* Sidebar background */
#     .sidebar .sidebar-content {
#         background-color: #fffbdd;
#         border-right: 2px solid #f0e68c;
#     }

#     /* Customize headers */
#     h1, h2, h3 {
#         color: #665c00;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


upload = st.file_uploader("Upload an image of a cat or dog", type=['png', 'jpg', 'jpeg'])

if upload:
    files = {"file": upload.getvalue()}

    with st.spinner("Analyzing the image..."):
        req = requests.post("https://week3-mlops-deploy.dataafriquehub.org/app1/api/predict", files=files)
        resultat = req.json()
        prob_cat = resultat["cat_proba"] * 100
        prob_dog = resultat["dog_proba"] * 100

    st.image(Image.open(upload), caption="Uploaded Image", use_column_width=True)

    st.subheader("Prediction Results")
    if prob_cat > prob_dog:
        st.markdown(
            f"""
            <div style='padding: 2rem; background-color: #f8d7da; border-left: 5px solid #dc3545;'>
            <h2 style='color: #721c24;'>Cat</h2>
            <p>I am <strong>{prob_cat:.2f}%</strong> certain this is a cat.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='padding: 2rem; background-color: #d4edda; border-left: 5px solid #28a745;'>
            <h2 style='color: #155724;'>Dog</h2>
            <p>I am <strong>{prob_dog:.2f}%</strong> certain this is a dog.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
