import tensorflow as tf
import numpy as np
from PIL import Image

def load_models():
    model_A = tf.keras.models.load_model("modelo_A.keras", compile=False)
    model_B = tf.keras.models.load_model("modelo_B_finetuned.keras", compile=False)
    return model_A, model_B

def preprocess_image(uploaded_file, target_size):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array
