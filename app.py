import streamlit as st
import numpy as np
from utils import load_models, preprocess_image

# 1️⃣ Cargar modelos
model_A, model_B = load_models()

# Detectar tamaño de input esperado por los modelos
input_shape_A = model_A.input_shape[1:3]  # height, width
input_shape_B = model_B.input_shape[1:3]  # height, width

st.write("Model A input shape:", input_shape_A)
st.write("Model B input shape:", input_shape_B)

# 2️⃣ Clases
class_names_A = ["Blue reciclable", "Brown Compost", "Gray Thrash", "SPECIAL"]
class_names_B = ["Donation", "Drop Off (recogida municipal)", 
                 "HHW punto municipal tóxico", "Medical take off", "TAKE BACK SHOP"]

# 3️⃣ Interfaz
st.title("♻️ Waste Classificator")
st.write("Upload an image of waste and we'll tell you which container it goes in / Sube una imagen de residuo y te diremos en qué contenedor va.")

uploaded_file = st.file_uploader("Upload an image / Sube una imagen", type=["jpg","jpeg","png"])

if uploaded_file:
    # Preprocesar imagen según input_shape del modelo A
    img, img_array = preprocess_image(uploaded_file, target_size=input_shape_A)

    st.image(img, caption="Image Uploaded / Imagen subida", use_container_width=True)

    # Paso 1 → Modelo A
    preds_A = model_A.predict(img_array)
    class_A = class_names_A[np.argmax(preds_A)]

    st.subheader(f"📦 General classification / Clasificación general: **{class_A}**")

    # Paso 2 → Si es SPECIAL → usar Modelo B
    if class_A == "SPECIAL":
        # Redimensionar imagen según input_shape del modelo B
        _, img_array_B = preprocess_image(uploaded_file, target_size=input_shape_B)
        preds_B = model_B.predict(img_array_B)
        class_B = class_names_B[np.argmax(preds_B)]
        st.subheader(f"⚠️ Subcategoría detectada: **{class_B}**")
