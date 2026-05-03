import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re

# Load plants
plants_path = os.path.join(os.path.dirname(__file__), '..', 'plants.json')
with open(plants_path, 'r', encoding='utf-8') as f:
    plants = json.load(f)

months_list = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

data = []
for name, info in plants.items():
    # Extract temp range
    temp_matches = re.findall(r'\d+', info['temperature'])
    min_temp = int(temp_matches[0]) if temp_matches else 20
    max_temp = int(temp_matches[1]) if len(temp_matches) > 1 else min_temp + 5
    
    # Extract soil type (simplified)
    soil = info['soil'].lower()
    if 'loam' in soil: primary_soil = 'loamy'
    elif 'sand' in soil: primary_soil = 'sandy'
    elif 'clay' in soil: primary_soil = 'clayey'
    elif 'alluvial' in soil: primary_soil = 'alluvial'
    elif 'red' in soil: primary_soil = 'red'
    elif 'silt' in soil: primary_soil = 'silty'
    else: primary_soil = 'loamy'
    
    # Extract months
    months_str = info['months'].lower()
    valid_months = []
    if 'year-round' in months_str:
        valid_months = list(range(1, 13))
    elif '–' in months_str or '-' in months_str:
        parts = [p.strip() for p in re.split(r'[–-]', months_str)]
        try:
            start_idx = months_list.index(parts[0]) + 1
            end_idx = months_list.index(parts[1]) + 1
            if start_idx <= end_idx:
                valid_months = list(range(start_idx, end_idx + 1))
            else:
                valid_months = list(range(start_idx, 13)) + list(range(1, end_idx + 1))
        except ValueError:
            valid_months = [6,7,8] # fallback summer
    else:
        valid_months = list(range(1, 13)) # fallback
        
    # Generate synthetic samples
    for _ in range(100): # 100 samples per plant
        t = np.random.uniform(min_temp - 2, max_temp + 2)
        m = np.random.choice(valid_months)
        data.append({
            'month': m,
            'temperature': t,
            'soil_type': primary_soil,
            'plant': name
        })

df = pd.DataFrame(data)

# Encode categorical variables
soil_encoder = LabelEncoder()
df['soil_type_encoded'] = soil_encoder.fit_transform(df['soil_type'])

X = df[['month', 'temperature', 'soil_type_encoded']]
y = df['plant']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Save model and encoder
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(rf, os.path.join(models_dir, 'plant_rf_model.pkl'))
joblib.dump(soil_encoder, os.path.join(models_dir, 'soil_encoder.pkl'))

print("Model trained and saved successfully in ml_service/models/")
