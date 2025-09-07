import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Cargar modelos y metadatos
# -----------------------------
def load_models():
    # Cargar modelo A
    try:
        model_A = tf.keras.models.load_model("modelo_A_fallback.keras", compile=False)
        print("✅ Modelo A cargado desde .keras")
    except:
        model_A = tf.keras.models.load_model("modelo_A.h5", compile=False)
        print("⚠️ Modelo A cargado desde .h5")

    # Cargar modelo B
    try:
        model_B = tf.keras.models.load_model("modelo_B_fallback.keras", compile=False)
        print("✅ Modelo B cargado desde .keras")
    except:
        model_B = tf.keras.models.load_model("modelo_B_finetuned.h5", compile=False)
        print("⚠️ Modelo B cargado desde .h5")

    # Leer metadatos de clases
    with open("model_A_meta.json", "r") as f:
        meta_A = json.load(f)
    with open("model_B_meta.json", "r") as f:
        meta_B = json.load(f)

    classes_A = meta_A["classes"]
    classes_B = meta_B["classes"]

    return (model_A, classes_A), (model_B, classes_B)

# -----------------------------
# Preprocesar imágenes
# -----------------------------
def preprocess_image(img, model):
    """Prepara una imagen para el modelo dado"""
    target_h, target_w = model.input_shape[1], model.input_shape[2]

    # Convertir a array y redimensionar
    img = img.resize((target_w, target_h))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    # Normalización de MobileNetV2
    arr = preprocess_input(arr)

    return arr
