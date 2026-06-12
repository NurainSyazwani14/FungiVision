from tensorflow.keras.models import load_model

model = load_model("model/fungi_mobilenet_pertype.h5")

model.save("model/fungi_mobilenet_pertype.keras")