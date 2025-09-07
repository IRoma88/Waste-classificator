import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from PIL import Image

# --- Arquitectura usada en el entrenamiento ---
def build_model(num_classes: int, input_size=(224, 224, 3)) -> tf.keras.Model:
    # OJO: weights=None porque vamos a cargar NUESTROS pesos entrenados.
    base_model = MobileNetV2(
        input_shape=input_size,
        include_top=False,
        weights=None
    )
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_models():
    # Lee metadatos
    meta_A = load_json("model_A_meta.json")
    meta_B = load_json("model_B_meta.json")

    input_h_A, input_w_A = meta_A["input_size"]
    input_h_B, input_w_B = meta_B["input_size"]

    # Construye arquitecturas
    model_A = build_model(num_classes=len(meta_A["classes"]),
                          input_size=(input_h_A, input_w_A, 3))
    model_B = build_model(num_classes=len(meta_B["classes"]),
                          input_size=(input_h_B, input_w_B, 3))

    # Carga pesos
    model_A.load_weights("modelo_A.weights.h5")
    model_B.load_weights("modelo_B.weights.h5")

    # Devuelve modelos y metadatos (clases)
    return (model_A, meta_A["classes"]), (model_B, meta_B["classes"])

def preprocess_image(uploaded_file, target_size):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    # NOTA: durante el entrenamiento no usaste preprocess_input de MobileNetV2,
    # así que aquí mantenemos la misma escala [0..255]
    return img, img_array
