from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List
from datetime import datetime, timezone
import os
import time
import re
from db.database import get_db, LeafPrediction
from preprocessing.image_processor import ImagePreprocessor
import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:3000")
preprocessor = ImagePreprocessor(target_size=(224, 224))
# Rate Limiting
RATE_LIMIT = 10  # each IP can have 10 requests per minute max
RATE_LIMIT_WINDOW = 60  
rate_limit_cache = {}

def check_rate_limit(request: Request):
    ip = request.client.host
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    if ip not in rate_limit_cache:
        rate_limit_cache[ip] = []
    # Remove old requests
    rate_limit_cache[ip] = [t for t in rate_limit_cache[ip] if t > window_start]
    if len(rate_limit_cache[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    rate_limit_cache[ip].append(now)

# Model and Preprocessor
# MODEL_PATH = "F:/Crop_Leaf_Disease_Detection/models/crop_disease_model.keras"
# model = keras.models.load_model(MODEL_PATH)
# preprocessor = ImagePreprocessor(target_size=(224, 224))
CLASS_NAMES = ['Healthy', 'Early_Disease', 'Severe_Disease']

router = APIRouter()

def infer_crop_type(filename: str) -> str:
    """
    Infer crop type from filename using known crop names.
    """
    crops = [
        "Tomato", "Potato", "Pepper", "Apple", "Blueberry", "Cherry", "Corn",
        "Grape", "Peach", "Raspberry", "Soybean", "Squash", "Strawberry"
    ]
    for crop in crops:
        if crop.lower() in filename.lower():
            return crop
        
    match = re.match(r"([A-Za-z]+)[_\-]", filename)
    if match:
        return match.group(1)
    return "Unknown"

CLASS_NAMES = [
    'Pepper__bell___healthy',
    'Potato___healthy',
    'Tomato_healthy',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___healthy',
    'Corn___healthy',
    'Grape___healthy',
    'Peach___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',  
    'Strawberry___healthy',
    
    # Early Disease classes
    'Pepper__bell___Bacterial_spot',
    'Potato___Early_blight',
    'Tomato_Early_blight',
    'Tomato_Septoria_leaf_spot',
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Peach___Bacterial_spot',
    'Strawberry___Leaf_scorch',
    
    # Severe Disease classes
    'Potato___Late_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Apple___Cedar_apple_rust',
    'Corn___Northern_Leaf_Blight',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'
]

# Mapping to severity categories
CLASS_TO_SEVERITY = {
    # Healthy classes
    'Pepper__bell___healthy': 'Healthy',
    'Potato___healthy': 'Healthy',
    'Tomato_healthy': 'Healthy',
    'Apple___healthy': 'Healthy',
    'Blueberry___healthy': 'Healthy',
    'Cherry___healthy': 'Healthy',
    'Corn___healthy': 'Healthy',
    'Grape___healthy': 'Healthy',
    'Peach___healthy': 'Healthy',
    'Raspberry___healthy': 'Healthy',
    'Soybean___healthy': 'Healthy',
    'Squash___Powdery_mildew': 'Healthy',  
    'Strawberry___healthy': 'Healthy',
    
    # Early Disease classes
    'Pepper__bell___Bacterial_spot': 'Early_Disease',
    'Potato___Early_blight': 'Early_Disease',
    'Tomato_Early_blight': 'Early_Disease',
    'Tomato_Septoria_leaf_spot': 'Early_Disease',
    'Apple___Apple_scab': 'Early_Disease',
    'Apple___Black_rot': 'Early_Disease',
    'Cherry___Powdery_mildew': 'Early_Disease',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Early_Disease',
    'Corn___Common_rust': 'Early_Disease',
    'Grape___Black_rot': 'Early_Disease',
    'Grape___Esca_(Black_Measles)': 'Early_Disease',
    'Peach___Bacterial_spot': 'Early_Disease',
    'Strawberry___Leaf_scorch': 'Early_Disease',
    
    # Severe Disease classes
    'Potato___Late_blight': 'Severe_Disease',
    'Tomato_Late_blight': 'Severe_Disease',
    'Tomato_Leaf_Mold': 'Severe_Disease',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Severe_Disease',
    'Tomato_Target_Spot': 'Severe_Disease',
    'Tomato_Tomato_mosaic_virus': 'Severe_Disease',
    'Tomato_Tomato_YellowLeaf__Curl_Virus': 'Severe_Disease',
    'Tomato_Bacterial_spot': 'Severe_Disease',
    'Apple___Cedar_apple_rust': 'Severe_Disease',
    'Corn___Northern_Leaf_Blight': 'Severe_Disease',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Severe_Disease',
}

# Endpoint 1: Upload and Classify
@router.post("/upload_leaf_image", tags=["Predictions"])
async def upload_leaf_image(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    check_rate_limit(request)

    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG files are allowed.")
    # File size validation (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB allowed.")

    # Preprocess image
    try:
        processed_img, meta = preprocessor.preprocess(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    
    try:
        response = requests.post(
            f"{MODEL_SERVICE_URL}/predict",
            json={"image_array": processed_img.tolist()},
            timeout=30
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Model service unreachable: {e}")

    if response.status_code != 200:
        try:
            err_detail = response.json().get('detail', response.text)
        except:
            err_detail = response.text
        raise HTTPException(
            status_code=502,
            detail=f"Model service error (HTTP {response.status_code}): {err_detail}"
        )

    try:
        result = response.json()
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from model service: {e}")

    print(f"[Backend] Model service response: {result}")  # Debug logging

    # Extract disease status and confidence, with fallbacks for different response formats
    try:
        disease_status = result.get("disease_status")
        if disease_status is None and "disease_name" in result:
            # Use CLASS_TO_SEVERITY mapping if available
            disease_name = result["disease_name"]
            disease_status = CLASS_TO_SEVERITY.get(disease_name, "Unknown")
        
        confidence = result.get("confidence")
        if confidence is None:
            confidence = result.get("confidence_score", 0.0)

        if disease_status is None or confidence is None:
            raise KeyError("Missing required fields")

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid response format from model service. Fields missing: {str(e)}. Got: {result}"
        )

    crop_type = infer_crop_type(file.filename)
    # Store in DB
    prediction = LeafPrediction(
        crop_type=crop_type,
        disease_status=disease_status,
        confidence_score=float(confidence),  # Ensure float type
        timestamp=datetime.now(timezone.utc),
        image_filename=file.filename
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return {
        "leaf_id": prediction.leaf_id,
        "crop_type": crop_type,
        "disease_status": prediction.disease_status,
        "confidence_score": prediction.confidence_score,
        "timestamp": prediction.timestamp.isoformat() if prediction.timestamp else None,
        "image_filename": prediction.image_filename
    }

# Endpoint 2: Get Results
@router.get("/get_results", tags=["Predictions"])
def get_results(
    db: Session = Depends(get_db),
    limit: int = Query(10, ge=1, le=100),
    crop_type: Optional[str] = None,
    disease_status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    query = db.query(LeafPrediction)
    if crop_type:
        query = query.filter(LeafPrediction.crop_type == crop_type)
    if disease_status:
        query = query.filter(LeafPrediction.disease_status == disease_status)
    if start_date:
        query = query.filter(LeafPrediction.timestamp >= start_date)
    if end_date:
        query = query.filter(LeafPrediction.timestamp <= end_date)
    total = query.count()
    results = query.order_by(LeafPrediction.timestamp.desc()).limit(limit).all()

    # Summary statistics
    disease_counts = (
        db.query(LeafPrediction.disease_status, func.count(LeafPrediction.disease_status))
        .group_by(LeafPrediction.disease_status)
        .all()
    )
    disease_distribution = {status: count for status, count in disease_counts}

    return {
        "total_predictions": total,
        "disease_distribution": disease_distribution,
        "results": [r.to_dict() for r in results]
    }
