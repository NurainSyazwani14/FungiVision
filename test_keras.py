from tensorflow.keras.models import load_model

model = load_model(
    "model/fungi_mobilenet_pertype.keras",
    compile=False
)

print("SUCCESS")