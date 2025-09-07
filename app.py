import streamlit as st
import numpy as np
from utils import load_models, preprocess_image

# 1️⃣ Cargar modelos
model_A, model_B = load_models()

# 2️⃣ Clases
class_names_A = ["Blue reciclable", "Brown Compost", "Gray Thrash", "SPECIAL"]
class_names_B = ["Donation", "Drop Off (recogida municipal)", 
                 "HHW punto municipal tóxico", "Medical take off", "TAKE BACK SHOP"]

# 3️⃣ Interfaz
st.title("♻️ Waste Classificator")
st.write("Upload an image of waste and we'll tell you which container it goes in / Sube una imagen de residuo y te diremos en qué contenedor va.")

uploaded_file = st.file_uploader("Upload an image / Sube una imagen", type=["jpg","jpeg","png"])

if uploaded_file:
    img, img_array = preprocess_image(uploaded_file)

    st.image(img, caption="Image Uploaded / Imagen subida", use_container_width=True)

    # Paso 1 → Modelo A
    preds_A = model_A.predict(img_array)
    class_A = class_names_A[np.argmax(preds_A)]

    st.subheader(f"📦 General classification / Clasificación general: **{class_A}**")

    # Paso 2 → Si es SPECIAL → usar Modelo B
    if class_A == "SPECIAL":
        preds_B = model_B.predict(img_array)
        class_B = class_names_B[np.argmax(preds_B)]
        st.subheader(f"⚠️ Subcategoría detectada: **{class_B}**")
