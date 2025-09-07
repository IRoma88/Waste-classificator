import streamlit as st
import numpy as np
from utils import load_models, preprocess_image

st.set_page_config(page_title="Waste Classificator", page_icon="♻️", layout="centered")

st.title("♻️ Waste Classificator")
st.write("Upload an image of waste and we'll tell you which container it goes in / "
         "Sube una imagen de residuo y te diremos en qué contenedor va.")

# 1) Cargar modelos + clases (cacheado para no recargar en cada rerun)
@st.cache_resource
def _load():
    (model_A, classes_A), (model_B, classes_B) = load_models()
    return model_A, classes_A, model_B, classes_B

model_A, class_names_A, model_B, class_names_B = _load()

# Detectar tamaño de input esperado por los modelos
input_shape_A = model_A.input_shape[1:3]
input_shape_B = model_B.input_shape[1:3]

with st.expander("ℹ️ Debug info (ocultar/mostrar)"):
    st.write("Model A input shape:", input_shape_A)
    st.write("Model B input shape:", input_shape_B)
    st.write("Classes A:", class_names_A)
    st.write("Classes B:", class_names_B)

# 2) UI de upload
uploaded_file = st.file_uploader("Upload an image / Sube una imagen", type=["jpg","jpeg","png"])

if uploaded_file:
    # Preprocesar para modelo A
    img, img_array = preprocess_image(uploaded_file, target_size=input_shape_A)
    st.image(img, caption="Image Uploaded / Imagen subida", use_container_width=True)

    # Paso 1 → Modelo A
    preds_A = model_A.predict(img_array)
    idx_A = int(np.argmax(preds_A, axis=1)[0])
    class_A = class_names_A[idx_A]
    conf_A = float(np.max(preds_A))

    st.subheader(f"📦 Clasificación general: **{class_A}** (conf: {conf_A:.2f})")

    # Paso 2 → Si es SPECIAL → usar Modelo B
    if class_A == "SPECIAL":
        _, img_array_B = preprocess_image(uploaded_file, target_size=input_shape_B)
        preds_B = model_B.predict(img_array_B)
        idx_B = int(np.argmax(preds_B, axis=1)[0])
        class_B = class_names_B[idx_B]
        conf_B = float(np.max(preds_B))

        st.subheader(f"⚠️ Subcategoría detectada: **{class_B}** (conf: {conf_B:.2f})")
