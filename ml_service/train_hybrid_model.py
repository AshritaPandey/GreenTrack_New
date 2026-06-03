import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

def train_hybrid_model(dataset_dir="dataset"):
    print("🚀 Initializing Hybrid Model Training (CNN Feature Extractor + Random Forest)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Pre-trained CNN as Feature Extractor
    cnn_model = models.mobilenet_v2(pretrained=True)
    # Remove the classification head, keep only the feature extractor and pooling
    feature_extractor = nn.Sequential(
        cnn_model.features,
        nn.AdaptiveAvgPool2d(1)
    ).to(device)
    feature_extractor.eval()
    
    # 2. Data Loading Pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    dataloader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    # Save Class Indices
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'hybrid_class_indices.json'), 'w') as f:
        json.dump(full_dataset.class_to_idx, f, indent=4)
        
    print("Extracting deep features from dataset...")
    X_features = []
    y_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # Extract features: output shape (batch_size, 1280, 1, 1)
            features = feature_extractor(inputs)
            # Flatten to (batch_size, 1280)
            features = features.view(features.size(0), -1).cpu().numpy()
            
            X_features.extend(features)
            y_labels.extend(labels.numpy())
            
    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    
    print(f"Feature extraction complete. Shape: {X_features.shape}")
    print("🌲 Training Random Forest Classifier...")
    
    # 3. Train Random Forest (The Hybrid Classification Layer)
    rf_clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf_clf.fit(X_features, y_labels)
    
    accuracy = rf_clf.score(X_features, y_labels)
    print(f"✅ Random Forest Training Complete! Accuracy: {accuracy*100:.2f}%")
    
    # 4. Save the Hybrid Model
    joblib.dump(rf_clf, os.path.join(models_dir, 'hybrid_disease_rf.pkl'))
    print("💾 Saved Hybrid Model to models/hybrid_disease_rf.pkl")

if __name__ == '__main__':
    train_hybrid_model(r"c:\Projects\Community Connect\dataset")
