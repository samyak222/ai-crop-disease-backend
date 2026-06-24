# 🌿 AI Crop Disease Detector

An AI-powered web application that detects crop diseases from leaf images and provides treatment recommendations — supporting sustainable farming practices.

**Live Demo:** [crop-frontend-navy.vercel.app](https://crop-frontend-navy.vercel.app)

---

## Features

- Upload a crop leaf image and get instant disease detection
- Supports 16 disease classes across Tomato, Potato, and Pepper plants
- Confidence score with visual progress bar
- Treatment recommendations for each detected disease
- SDG contribution — promoting sustainable agriculture

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + Vite, deployed on Vercel |
| Backend | Flask + Gunicorn, deployed on Render |
| AI Model | TensorFlow Lite (converted from Keras) |
| Image Processing | Pillow, NumPy |

---

## Supported Classes

| Crop | Disease |
|------|---------|
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Pepper Bell | Bacterial Spot, Healthy |

---

## How It Works

1. User uploads a crop leaf image on the frontend
2. Image is sent to the Flask backend via REST API
3. TFLite model processes the image (224×224 RGB)
4. Predicted disease, confidence score, and treatment are returned
5. Results displayed instantly on the frontend

---

## Local Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## API

**GET** `/` — Health check

**POST** `/predict` — Predict disease from image

Request: `multipart/form-data` with `file` field

Response:
```json
{
  "disease": "Tomato Early Blight",
  "confidence": 94.32,
  "treatment": "Use fungicide and remove infected leaves."
}
```

---

## Dataset

Trained on [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — 16 classes, 224×224 RGB images.

---

## Deployment

- Frontend: Vercel (auto-deploy from Git)
- Backend: Render (free tier, TFLite for low memory usage)

---

*Built to support SDG Goal 2 — Zero Hunger, by helping farmers identify and treat crop diseases early.*