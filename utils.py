import tensorflow as tf
import numpy as np
from PIL import Image

def load_models():
    """
    Carga los modelos completos desde los .h5
    modelo_A.h5 -> Modelo principal
    modelo_B_finetuned.h5 -> Modelo para subcategorías (SPECIAL)
    """
    model_A = tf.keras.models.load_model("modelo_A.h5", compile=False)
    model_B = tf.keras.models.load_model("modelo_B_finetuned.h5", compile=False)
    return model_A, model_B

def preprocess_image(uploaded_file, target_size):
    """
    Preprocesa la imagen para que tenga el tamaño que espera el modelo.
    """
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch
    return img, img_array
