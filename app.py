import streamlit as st
import numpy as np
from utils import load_models, preprocess_image

@st.cache_resource
def _load():
    (mA, classes_A), (mB, classes_B) = load_models()
    return mA, classes_A, mB, classes_B

model_A, classes_A, model_B, classes_B = _load()

# detectar input sizes:
input_shape_A = model_A.input_shape[1:3]
input_shape_B = model_B.input_shape[1:3]

uploaded_file = st.file_uploader("Sube imagen", type=["jpg","jpeg","png"])
if uploaded_file:
    img, arrA = preprocess_image(uploaded_file, target_size=input_shape_A)
    st.image(img, use_container_width=True)
    preds = model_A.predict(arrA)
    idx = int(np.argmax(preds, axis=1)[0])
    st.write("Contenedor:", classes_A[idx])
    if classes_A[idx] == "SPECIAL":
        _, arrB = preprocess_image(uploaded_file, target_size=input_shape_B)
        predsB = model_B.predict(arrB)
        idxb = int(np.argmax(predsB, axis=1)[0])
        st.write("SPECIAL →", classes_B[idxb])
