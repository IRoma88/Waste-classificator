import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

def build_model_A():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),  # Cambia según tu modelo
        Flatten(),
        Dense(4, activation='softmax')
    ])
    model.load_weights("modelo_A.h5")
    return model

def build_model_B():
    model = Sequential([
        # Reconstruye la arquitectura exacta de tu modelo B
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        Flatten(),
        Dense(5, activation='softmax')
    ])
    model.load_weights("modelo_B_finetuned.h5")
    return model

def load_models():
    return build_model_A(), build_model_B()

def preprocess_image(uploaded_file, target_size):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array
