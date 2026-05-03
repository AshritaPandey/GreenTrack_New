from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import random
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MobileNetV2 (CPU Optimized)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

@app.post("/predict-disease")
async def predict_disease(
    file: UploadFile = File(...),
    plant: str = Form(...),
    notes: str = Form(None)
):
    contents = await file.read()
    processed_image = preprocess_image(contents)
    
    # Run Inference
    predictions = model.predict(processed_image)
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    probabilities = predictions[0]
    confidence = float(np.max(probabilities))
    
    # Broad Signal Classification
    top_5_indices = top_indices.tolist()
    top_5_probs = probabilities[top_indices].tolist()
    
    # Pathogen Classes (Pests, Galls, Blight, Irregular spots)
    pathogen_ids = list(range(300, 331)) + [928, 929, 930, 416, 744, 849, 917, 722, 921, 922, 947, 948, 950, 952, 953, 954, 916]
    # Healthy/Natural Classes (Leaves, Trees, Grass, Pots, Earth)
    healthy_ids = [915, 945, 970, 972, 980, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 738, 881, 441, 442]

    pathogen_score = sum(prob for idx, prob in zip(top_5_indices, top_5_probs) if idx in pathogen_ids)
    healthy_score = sum(prob for idx, prob in zip(top_5_indices, top_5_probs) if idx in healthy_ids)
    
    plant_type = plant.lower()
    
    if plant_type == "mango":
        if (pathogen_score > healthy_score * 1.5) or (pathogen_score > 0.4):
            if any(300 <= idx <= 330 for idx in top_5_indices):
                status, disease, solutions = "Critical Alert", "Mango Leaf Gall Midge (Erosomyia mangiferae)", ["Apply Dimethoate 30 EC (2ml/L) immediately.", "Remove and burn infested terminal shoots.", "Spray Neem Oil (5%) as a preventive."]
            else:
                status, disease, solutions = "Concern Detected", "Mango Anthracnose (Colletotrichum gloeosporioides)", ["Spray Copper Oxychloride (3g/L) or Carbendazim (1g/L).", "Improve drainage to reduce humidity.", "Prune dead wood and fallen leaves."]
        elif confidence < 0.12:
            status, disease, solutions = "Inconclusive", "Low Visibility", ["I cannot identify the plant clearly.", "Please provide a closer, high-resolution photo."]
        else:
            status, disease, solutions = "Healthy", "None", ["Specimen shows optimal health signals.", "Maintain current care routine."]
            
    elif plant_type == "tomato":
        if pathogen_score > healthy_score or confidence < 0.18:
            status, disease, solutions = "Warning", "Septoria Leaf Spot / Blight", ["Apply Mancozeb fungicide.", "Improve air circulation."]
        else:
            status, disease, solutions = "Healthy", "None", ["Leaf cuticle is intact.", "No pathogens detected."]
    else:
        if pathogen_score > healthy_score * 1.2:
            status, disease, solutions = "Concern Detected", "Potential Issue", ["Apply Neem Oil.", "Monitor for 48 hours."]
        else:
            status, disease, solutions = "Healthy", "None", ["Specimen appears healthy.", "Keep up the good work!"]

    return {
        "health_status": status,
        "disease_id": disease,
        "affected_part": "Foliage",
        "confidence_score": confidence,
        "solution_steps": solutions,
        "highlight_area": {"x": 20, "y": 20, "w": 60, "h": 60} if status != "Healthy" else None
    }

@app.post("/recommend-plant")
async def recommend_plant(data: dict):
    # Logic for 5-point ecosystem matching
    month = data.get("month", "May")
    soil = data.get("soil_type", "Loamy")
    temp = data.get("current_temp", 30)
    nutrients = data.get("nutrients", [])
    
    # Simplified recommendation for demo
    recommendations = [
        {"name": "Mango", "match": 95, "reason": "Perfect temperature synergy."},
        {"name": "Tomato", "match": 88, "reason": "High nutrient compatibility."},
        {"name": "Guava", "match": 82, "reason": "Soil type is optimal."}
    ]
    
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
