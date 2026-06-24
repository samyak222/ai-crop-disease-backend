import tensorflow as tf

model = tf.keras.models.load_model(
    "model/crop_disease_model.h5",
    compile=False
)

model.save("model/crop_disease_model.keras")
print("Model converted successfully")