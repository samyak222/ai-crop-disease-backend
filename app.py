from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import traceback

app = Flask(__name__)
CORS(app)

import os
print("MODEL FILES:", os.listdir("model"))

# Load trained model
model = tf.keras.models.load_model(
    "model/crop_disease_model.h5",
    compile=False
)

# Actual classes from dataset
class_names = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "PlantVillage",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

# Treatments
treatments = {
    "Pepper Bell Bacterial Spot": "Use copper-based sprays and remove infected leaves.",
    "Pepper Bell Healthy": "No treatment required.",
    "PlantVillage": "Dataset folder detected. Dataset structure should be checked.",
    "Potato Early Blight": "Apply fungicide and remove infected foliage.",
    "Potato Late Blight": "Use disease-resistant varieties and fungicide treatment.",
    "Potato Healthy": "No treatment required.",
    "Tomato Bacterial Spot": "Use copper sprays and avoid overhead watering.",
    "Tomato Early Blight": "Use fungicide and remove infected leaves.",
    "Tomato Late Blight": "Apply copper-based fungicide and improve air circulation.",
    "Tomato Leaf Mold": "Improve ventilation and apply fungicide.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves and apply fungicide.",
    "Tomato Spider Mites": "Use miticides and maintain proper humidity.",
    "Tomato Target Spot": "Apply fungicide and prune infected leaves.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Tomato Mosaic Virus": "Remove infected plants and disinfect tools.",
    "Tomato Healthy": "No treatment required."
}


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/")
def home():
    return jsonify({
        "message": "AI Crop Disease Detection API Running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["file"]

        image = Image.open(file).convert("RGB")
        processed = preprocess_image(image)

        prediction = model.predict(processed, verbose=0)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        print("\n====================")
        print("Prediction:", prediction[0])
        print("Class Index:", np.argmax(prediction))
        print("Max Confidence:", np.max(prediction))
        print("====================\n")

        if class_index >= len(class_names):
            return jsonify({
                "disease": f"Unknown Class {class_index}",
                "confidence": round(confidence, 2),
                "treatment": "Class mapping missing."
            })

        disease = class_names[class_index]

        treatment = treatments.get(
            disease,
            "Consult an agricultural expert for treatment recommendations."
        )

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "treatment": treatment
        })

    except Exception as e:
        print("\n===== ERROR =====")
        traceback.print_exc()
        print("=================\n")

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)