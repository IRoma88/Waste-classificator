import streamlit as st
import numpy as np
from PIL import Image
from utils import load_models, preprocess_image

# -----------------------------
# Cargar modelos y clases
# -----------------------------
@st.cache_resource
def _load():
    (model_A, classes_A), (model_B, classes_B) = load_models()
    return model_A, classes_A, model_B, classes_B

model_A, class_names_A, model_B, class_names_B = _load()

# -----------------------------
# Interfaz
# -----------------------------
st.title("♻️ Waste Classificator")
st.write("Upload an image of waste and we'll tell you which container it goes in / "
         "Sube una imagen de residuo y te diremos en qué contenedor va.")

uploaded_file = st.file_uploader(
    "Upload an image / Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Abrir imagen
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    # -----------------------------
    # Paso 1: Modelo A
    # -----------------------------
    arr_A = preprocess_image(img, model_A)
    preds_A = model_A.predict(arr_A)
    idx_A = np.argmax(preds_A)
    class_A = class_names_A[idx_A]
    prob_A = preds_A[0][idx_A]

    st.subheader(f"📦 General classification: **{class_A}** ({prob_A:.2f})")

    # -----------------------------
    # Paso 2: Modelo B si es SPECIAL
    # -----------------------------
    if class_A.upper() == "SPECIAL":
        arr_B = preprocess_image(img, model_B)
        preds_B = model_B.predict(arr_B)
        idx_B = np.argmax(preds_B)
        class_B = class_names_B[idx_B]
        prob_B = preds_B[0][idx_B]

        st.subheader(f"⚠️ Subcategoría detectada: **{class_B}** ({prob_B:.2f})")
