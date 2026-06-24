from flask import Flask, request, jsonify
from flask_cors import CORS
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import traceback
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://crop-frontend-navy.vercel.app"]}})

print("Loading model...")
interpreter = tflite.Interpreter(model_path="model/crop_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("MODEL LOADED SUCCESSFULLY")

class_names = [
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy", "PlantVillage",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus", "Tomato Healthy"
]

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

@app.route("/")
def home():
    return jsonify({"message": "AI Crop Disease Detection API Running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        disease = class_names[class_index]
        treatment = treatments.get(disease, "Consult an agricultural expert.")

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "treatment": treatment
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)