# Crop Leaf Disease Detection System

## 1. Overview

This project provides an end-to-end system for classifying crop leaf health using computer vision and deep learning, exposed via REST APIs. It covers:

- Image preprocessing (segmentation, background removal, color correction)
- AI inference (MobileNet model)
- Results storage (PostgreSQL)
- REST endpoints for prediction and history

---

## 2. Project Structure

```
/
├─ model-service/
│  ├─ app/main.py              # FastAPI app: model inference endpoint
│  ├─ models/                  # .keras trained model file
│  └─ requirements.txt
│
├─ backend/
│  ├─ app/
│  │  ├─ main.py              # FastAPI app: all business logic endpoints
│  │  ├─ api/endpoints.py     # /upload_leaf_image and /get_results
│  │  ├─ preprocessing/       # image processing module
│  │  └─ db/database.py       # SQLAlchemy models, DB setup
│  └─ requirements.txt
│
├─ frontend/
│  ├─ streamlit_app.py        # Streamlit interface
│  └─ requirements.txt
│
├─ ml_training/               # (not deployed) model training scripts
│
├─ infra/
└─ README.md
```

---

## 3. Database Schema

### Predictions Table

The core table for storing prediction results:

| Column Name | Data Type | Primary Key | Not NULL | Description |
|-------------|-----------|-------------|----------|-------------|
| `leaf_id` | integer | Yes | Yes | Unique prediction ID |
| `crop_type` | varchar | No | Yes | Crop type (Tomato, Potato, etc.) |
| `disease_status` | varchar | No | Yes | Health/Stage: Healthy, Early_Disease, Severe_Disease |
| `confidence_score` | double | No | Yes | Model confidence (0.0 – 1.0) |
| `timestamp` | timestamp | No | Yes | Time when prediction stored |
| `image_filename` | varchar | No | No | Filename of original uploaded image |

---

## 4. Core Architecture

### 4.1. Image Preprocessing

Implemented in `backend/app/preprocessing/image_processor.py` using OpenCV and Pillow:

- **Color correction**: CLAHE histogram equalization (illumination normalization)
- **Background removal**: HSV thresholding, Gaussian blur, morphological ops, contour extraction
- **Segmentation**: Largest contour selected, image cropped and resized (224x224)
- **Exposed via**: Called as a utility in `/upload_leaf_image` endpoint

### 4.2. Model Service (DL Inference)

- FastAPI service (`model-service/app/main.py`)
- Loads pre-trained Keras `.keras` model (MobileNet, outputs 3-class)
- Accepts preprocessed arrays, returns predicted class and confidence
- Maps fine-grained class (from model) into: `Healthy`, `Early_Disease`, `Severe_Disease`

### 4.3. Backend API

Implemented in `backend/app/main.py` and `backend/app/api/endpoints.py`:

#### `/upload_leaf_image` [POST]

**Accepts**: `multipart/form-data` (image file)

**Process**:
1. Applies preprocessing steps
2. Sends processed image to Model Service; receives prediction and confidence
3. Infers crop type (from filename)
4. Stores result with all metadata in Postgres DB (see schema above)
5. Returns JSON response

**Example Response**:
```json
{
  "leaf_id": 1,
  "crop_type": "Tomato",
  "disease_status": "Early_Disease",
  "confidence_score": 0.91,
  "timestamp": "2025-10-20T12:05:01",
  "image_filename": "tomato_leaf_1.jpg"
}
```

#### `/get_results` [GET]

**Query params**: `limit`, `crop_type`, `disease_status`, `start_date`, `end_date`

**Features**:
- Lists recent/past predictions (paginated)
- Summary stats: total predictions, class distribution

### 4.4. Frontend

- Implemented in Streamlit (`frontend/streamlit_app.py`)
- Calls backend `/upload_leaf_image` for new predictions
- Calls `/get_results` for fetching and displaying DB history/stats

---

## 5. Setup Instructions

### Clone Repository

```bash
git clone https://github.com/rasikathakur/Crop_Leaf_Disease_Detection
```

### Installation

(Optional) Create separate environments for each service (model-service, backend, frontend).

Install dependencies per `requirements.txt` in each service's folder.

### Run Locally

#### Model Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

#### Backend

Set environment variables:

```bash
export MODEL_SERVICE_URL=http://localhost:8001
export DATABASE_URL=sqlite:///./test.db
```

Run server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Frontend

Set environment variable:

```bash
export BACKEND_URL=http://localhost:8000
```

Run Streamlit:

```bash
streamlit run streamlit_app.py
```

---

## 6. Key Points

- **Image preprocessing is critical for accuracy** - proper color correction and background removal significantly improve model performance
- **Separated model service** allows for easy scaling, updating, or moving to GPU infrastructure if needed
- **Clear schema** ensures robust data storage and easy querying
- **Modular project structure** enables easy updates and future extensions

---

## 7. Deployed Links

- **Frontend**: [https://crop-leaf-disease-detection-2.onrender.com](https://crop-leaf-disease-detection-2.onrender.com)
- **Backend API**: [https://crop-leaf-disease-detection-1.onrender.com/docs](https://crop-leaf-disease-detection-1.onrender.com/docs#/)
- **Model Service**: [https://crop-leaf-disease-detection.onrender.com/docs](https://crop-leaf-disease-detection.onrender.com/docs#/)
- **Source Code**: [https://github.com/rasikathakur/Crop_Leaf_Disease_Detection](https://github.com/rasikathakur/Crop_Leaf_Disease_Detection)

---

*Documentation generated for Crop Leaf Disease Detection System*
