import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(
    "model/fungi_mobilenet_pertype.keras",
    compile=False
)

model.export("model/fungi_savedmodel")