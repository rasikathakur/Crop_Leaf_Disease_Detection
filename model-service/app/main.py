from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow import keras
import numpy as np
import os
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    image_array: list
    
# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/crop_disease_model.keras")
model = keras.models.load_model(MODEL_PATH)

app = FastAPI(title="Model Prediction Service")

CLASS_NAMES = [
    'Pepper__bell___healthy', 'Potato___healthy', 'Tomato_healthy', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___healthy', 'Corn___healthy', 'Grape___healthy',
    'Peach___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___healthy', 'Pepper__bell___Bacterial_spot', 'Potato___Early_blight',
    'Tomato_Early_blight', 'Tomato_Septoria_leaf_spot', 'Apple___Apple_scab',
    'Apple___Black_rot', 'Cherry___Powdery_mildew', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Peach___Bacterial_spot', 'Strawberry___Leaf_scorch', 'Potato___Late_blight',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot', 'Tomato_Tomato_mosaic_virus', 'Tomato_Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot', 'Apple___Cedar_apple_rust', 'Corn___Northern_Leaf_Blight',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'
]

CLASS_TO_SEVERITY = {
    'Pepper__bell___healthy': 'Healthy', 'Potato___healthy': 'Healthy', 'Tomato_healthy': 'Healthy',
    'Apple___healthy': 'Healthy', 'Blueberry___healthy': 'Healthy', 'Cherry___healthy': 'Healthy',
    'Corn___healthy': 'Healthy', 'Grape___healthy': 'Healthy', 'Peach___healthy': 'Healthy',
    'Raspberry___healthy': 'Healthy', 'Soybean___healthy': 'Healthy', 'Squash___Powdery_mildew': 'Healthy',
    'Strawberry___healthy': 'Healthy', 'Pepper__bell___Bacterial_spot': 'Early_Disease',
    'Potato___Early_blight': 'Early_Disease', 'Tomato_Early_blight': 'Early_Disease',
    'Tomato_Septoria_leaf_spot': 'Early_Disease', 'Apple___Apple_scab': 'Early_Disease',
    'Apple___Black_rot': 'Early_Disease', 'Cherry___Powdery_mildew': 'Early_Disease',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Early_Disease', 'Corn___Common_rust': 'Early_Disease',
    'Grape___Black_rot': 'Early_Disease', 'Grape___Esca_(Black_Measles)': 'Early_Disease',
    'Peach___Bacterial_spot': 'Early_Disease', 'Strawberry___Leaf_scorch': 'Early_Disease',
    'Potato___Late_blight': 'Severe_Disease', 'Tomato_Late_blight': 'Severe_Disease',
    'Tomato_Leaf_Mold': 'Severe_Disease', 'Tomato_Spider_mites_Two_spotted_spider_mite': 'Severe_Disease',
    'Tomato_Target_Spot': 'Severe_Disease', 'Tomato_Tomato_mosaic_virus': 'Severe_Disease',
    'Tomato_Tomato_YellowLeaf__Curl_Virus': 'Severe_Disease', 'Tomato_Bacterial_spot': 'Severe_Disease',
    'Apple___Cedar_apple_rust': 'Severe_Disease', 'Corn___Northern_Leaf_Blight': 'Severe_Disease',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Severe_Disease'
}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        img = np.array(request.image_array, dtype=np.float32)
        preds = model.predict(img)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx])
        
        if class_idx < len(CLASS_NAMES):
            disease_name = CLASS_NAMES[class_idx]
            disease_status = CLASS_TO_SEVERITY.get(disease_name, "Unknown")
        else:
            disease_name = "Unknown"
            disease_status = "Uncertain"
        return {
            "disease_name": disease_name,
            "disease_status": disease_status,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "service": "model-service"}
