import tensorflow as tf
from PIL import Image
import numpy as np

def load_models():
    # carga directa de archivos .keras en la raíz del repo
    model_A = tf.keras.models.load_model("modelo_A_fallback.keras", compile=False)
    model_B = tf.keras.models.load_model("modelo_B_fallback.keras", compile=False)
    # Devuelve modelos y (opcional) arrays de clases si quieres
    classes_A = ["Blue reciclable", "Brown Compost", "Gray Thrash", "SPECIAL"]
    classes_B = ["Donation", "Drop Off (recogida municipal)",
                 "HHW punto municipal tóxico", "Medical take off", "TAKE BACK SHOP"]
    return (model_A, classes_A), (model_B, classes_B)

def preprocess_image(uploaded_file, target_size):
    from PIL import Image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(target_size)
    arr = tf.keras.utils.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)
    return img, arr
