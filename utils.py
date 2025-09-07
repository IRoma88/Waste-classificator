import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
import os

def load_models():
    try:
        model_A = tf.keras.models.load_model("modelo_A.h5")
        model_B = tf.keras.models.load_model("modelo_B_finetuned.h5")
        return model_A, model_B
    except Exception as e:
        print("Error loading models:", e)
        raise


def preprocess_image(uploaded_file, target_size=(224, 224)):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # batch
    return img, img_array
