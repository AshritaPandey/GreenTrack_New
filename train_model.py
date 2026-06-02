import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image

class CropToLeaf(object):
    def __call__(self, img):
        img_hsv = img.convert('HSV')
        np_img = np.array(img_hsv)
        # H is 0-255 in PIL HSV, green is roughly 60-140 (which is 40-100 in typical 0-180 scale, so ~50-120 in 0-255)
        H = np_img[:, :, 0]
        S = np_img[:, :, 1]
        V = np_img[:, :, 2]
        green_mask = (H > 50) & (H < 130) & (S > 30) & (V > 30)
        coords = np.argwhere(green_mask)
        if len(coords) < 100:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        h, w = img.size[1], img.size[0]
        pad_y = int((y1 - y0) * 0.2)
        pad_x = int((x1 - x0) * 0.2)
        return img.crop((max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y)))

def train_model():
    dataset_dir = 'dataset'
    models_dir = os.path.join('ml_service', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Data augmentation and normalization
    data_transforms = transforms.Compose([
        CropToLeaf(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms)
    
    # Save class indices
    class_indices = full_dataset.class_to_idx
    with open(os.path.join(models_dir, 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f)
        
    dataloader = DataLoader(full_dataset, batch_size=16, shuffle=True)
    
    # 2. Setup Model (MobileNetV2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_classes = len(full_dataset.classes)
    print(f"Training on {num_classes} classes: {full_dataset.classes}")
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze early layers
    for param in model.features[:10].parameters():
        param.requires_grad = False
        
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)
    
    # 3. Setup Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # 4. Training Loop (Fast fine-tuning: 5 epochs)
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(full_dataset)
        epoch_acc = running_corrects.double() / len(full_dataset)
        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
    # 5. Save model
    model_path = os.path.join(models_dir, 'plant_disease_mobilenet.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
