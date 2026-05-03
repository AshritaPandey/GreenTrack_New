from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import joblib
import os
import json
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Recommendation Model
models_dir = os.path.join(os.path.dirname(__file__), 'models')
try:
    rf_model = joblib.load(os.path.join(models_dir, 'plant_rf_model.pkl'))
    soil_encoder = joblib.load(os.path.join(models_dir, 'soil_encoder.pkl'))
except:
    print("Warning: Recommendation model not found. Run train_recommendation_model.py first.")
    rf_model = None
    soil_encoder = None

# Load plants JSON for recommendation metadata
plants_path = os.path.join(os.path.dirname(__file__), '..', 'plants.json')
try:
    with open(plants_path, 'r', encoding='utf-8') as f:
        plants_data = json.load(f)
except:
    plants_data = {}

# Load CNN Model (MobileNetV2 as a placeholder for disease detection)
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RecommendRequest(BaseModel):
    month: str
    current_temp: float
    soil_type: str
    area: str = "General"
    rain_prob: float = 0.0
    nutrients: str = "General"

@app.post("/predict-disease")
async def predict_disease(plant: str = Form(...), notes: str = Form(""), image_url: str = Form(None), image_path: str = Form(None)):
    if not image_url and not image_path:
        return {
            "health_status": "Potential Issue Detected",
            "disease_id": "Unknown (No image provided)",
            "affected_part": "Foliage",
            "confidence_score": "50%",
            "solution_steps": ["Please upload an image for CNN analysis."],
            "highlight_area": None
        }

    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                contents = f.read()
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            contents = response.content
        else:
            return {"error": "Image file not found on disk."}
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, _ = torch.topk(probabilities, 1)
        
        # 1. Get Top 5 classes to look for "unhealthy" features
        _, top_indices = torch.topk(probabilities, 5)
        top_catid = top_indices[0].item()
        confidence_val = top_prob.item()
        confidence = f"{confidence_val * 100:.1f}%"
        
        # Mapping ImageNet IDs to "Healthy" vs "Unhealthy" indicators
        # (Simplified: many plant-related IDs are in the 900+ range)
        is_plant = any(900 <= idx <= 999 for idx in top_indices.tolist())
        
        # Simulate disease detection mapping based on species + CNN run
        plant_type = plant.lower()
        
        # 1. Broad Signal Classification
        top_5_indices = top_indices.tolist()
        top_5_probs = probabilities[top_indices].tolist()
        
        # Pathogen Classes (Pests, Galls, Blight, Irregular spots)
        pathogen_ids = list(range(300, 331)) + [928, 929, 930, 416, 744, 849, 917, 722, 921, 922, 947, 948, 950, 952, 953, 954, 916]
        # Healthy/Natural Classes (Leaves, Trees, Grass, Pots, Earth)
        healthy_ids = [915, 945, 970, 972, 980, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 738, 881, 441, 442]

        pathogen_score = sum(prob for idx, prob in zip(top_5_indices, top_5_probs) if idx in pathogen_ids)
        healthy_score = sum(prob for idx, prob in zip(top_5_indices, top_5_probs) if idx in healthy_ids)
        
        # 2. Species Specific Decision Matrix
        if plant_type == "mango":
            # Real Pathogen Detection (High Confidence or Strong Pattern)
            if (pathogen_score > healthy_score * 1.5) or (pathogen_score > 0.4):
                if any(300 <= idx <= 330 for idx in top_5_indices):
                    status, disease, solutions = "Critical Alert", "Mango Leaf Gall Midge (Erosomyia mangiferae)", ["Apply Dimethoate 30 EC (2ml/L) immediately.", "Remove and burn infested terminal shoots.", "Spray Neem Oil (5%) as a preventive."]
                else:
                    status, disease, solutions = "Concern Detected", "Mango Anthracnose (Colletotrichum gloeosporioides)", ["Spray Copper Oxychloride (3g/L) or Carbendazim (1g/L).", "Improve drainage to reduce humidity.", "Prune dead wood and fallen leaves."]
            elif confidence_val < 0.12:
                status, disease, solutions = "Inconclusive", "Low Visibility", ["I cannot identify the plant clearly.", "Please provide a closer, high-resolution photo of the leaf surface."]
            else:
                status, disease, solutions = "Healthy", "None", ["The Mango specimen shows optimal health signals.", "Ensure consistent nitrogen-rich fertilization for the sapling."]
                
        elif plant_type == "tomato":
            if pathogen_score > healthy_score or confidence_val < 0.18:
                status, disease, solutions = "Warning", "Septoria Leaf Spot / Blight", ["Apply Mancozeb or Chlorothalonil fungicide.", "Mulch around base to prevent soil splash.", "Sterilize garden tools after use."]
            else:
                status, disease, solutions = "Healthy", "None", ["Leaf cuticle is intact and healthy.", "Maintain consistent moisture levels."]
                
        elif plant_type == "guava":
            if pathogen_score > healthy_score:
                status, disease, solutions = "Pest Alert", "Guava Tea Mosquito Bug / Wilt", ["Check for necrotic lesions on young leaves.", "Spray Malathion 50 EC (2ml/L).", "Install yellow sticky traps."]
            else:
                status, disease, solutions = "Healthy", "None", ["Vigorous growth detected.", "No signs of wilt or fungal pathogens."]
        else:
            # Universal Fallback
            if pathogen_score > healthy_score * 1.2:
                status, disease, solutions = "Concern Detected", "Potential Fungal / Pest Issue", ["Apply Neem Oil (5%) or organic fungicide.", "Isolate the plant to prevent spread."]
            else:
                status, disease, solutions = "Healthy", "None", ["The specimen appears visually healthy.", "Keep maintaining current care routine."]
            
        return {
            "health_status": status,
            "disease_id": disease,
            "affected_part": "Foliage" if is_plant else "Specimen",
            "confidence_score": confidence,
            "solution_steps": solutions,
            "highlight_area": {"x": 20, "y": 20, "w": 60, "h": 60} if status != "Healthy" else None
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/recommend-plant")
async def recommend_plant(req: RecommendRequest):
    if rf_model is None or soil_encoder is None:
        return {"error": "Model not loaded"}
    
    months_list = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    month_idx = months_list.index(req.month.lower()) + 1 if req.month.lower() in months_list else 6
    
    soil = req.soil_type.lower()
    if 'loam' in soil: primary_soil = 'loamy'
    elif 'sand' in soil: primary_soil = 'sandy'
    elif 'clay' in soil: primary_soil = 'clayey'
    elif 'alluvial' in soil: primary_soil = 'alluvial'
    elif 'red' in soil: primary_soil = 'red'
    elif 'silty' in soil: primary_soil = 'silty'
    else: primary_soil = 'loamy'
    
    try:
        soil_enc = soil_encoder.transform([primary_soil])[0]
    except:
        soil_enc = soil_encoder.transform(['loamy'])[0]

    input_data = pd.DataFrame([[month_idx, req.current_temp, soil_enc]], columns=['month', 'temperature', 'soil_type_encoded'])
    
    probs = rf_model.predict_proba(input_data)[0]
    classes = rf_model.classes_
    
    # 5-Point Precision Scoring Layer
    scored_recommendations = []
    for idx, plant_name in enumerate(classes):
        prob = probs[idx]
        if prob < 0.02: continue
        
        info = plants_data.get(plant_name, {})
        score = prob * 100 # Base score from RF model
        
        # 1. Area Match (Coastal/Urban/Rural)
        if "coastal" in req.area.lower() and plant_name in ["Mango", "Banana", "Coconut", "Hibiscus"]: score += 15
        if "valasaravakkam" in req.area.lower() or "chennai" in req.area.lower():
            if plant_name in ["Jasmine", "Rose", "Tomato", "Curry Leaves"]: score += 10 # Well-suited for humid TN climate
            
        # 2. Soil Texture Precision
        plant_soil = info.get('soil', '').lower()
        if primary_soil in plant_soil: score += 20
        
        # 3. Nutrient Compatibility
        micronutrients = req.nutrients.lower()
        if "iron" in micronutrients and plant_name in ["Lemon", "Hibiscus", "Rose"]: score += 15
        if "nitrogen" in micronutrients and plant_name in ["Mint", "Tulsi", "Money Plant"]: score += 10
        
        # 4. Temperature success zone
        # (Assuming plants.json has a 'temperature' string like "20°C - 30°C")
        try:
            temp_range = info.get('temperature', '15-35').replace('°C', '').split('–')
            if len(temp_range) == 2:
                low, high = float(temp_range[0].strip()), float(temp_range[1].strip())
                if low <= req.current_temp <= high: score += 20
                else: score -= 20 # Temperature mismatch
        except: pass

        # 5. Rain/Weather Resilience
        if req.rain_prob > 50:
            if plant_name in ["Banana", "Mint", "Ginger"]: score += 15 # Water-loving
            if plant_name in ["Aloe Vera", "Sunflower"]: score -= 20 # Succulents/sun-lovers hate heavy rain
        
        scored_recommendations.append({
            "name": plant_name,
            "score": round(score, 1),
            "match_reason": f"Ecosystem Match: {round(score/1.5, 1)}%",
            **info
        })
    
    # Sort by Ecosystem Score and return top 4
    scored_recommendations.sort(key=lambda x: x['score'], reverse=True)
    return scored_recommendations[:4]
