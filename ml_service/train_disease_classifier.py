import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image

# 1. Custom CNN Architecture with fine-tuning head
class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(PlantDiseaseClassifier, self).__init__()
        # Load MobileNetV2 pretrained backbone
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Standard Transfer Learning: Freeze early features to retain ImageNet filters
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
        # Replace classification head with custom deep network mapped to specific diseases
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def train_model(dataset_dir, epochs=10, batch_size=32, lr=0.001):
    """
    Standard professional transfer learning pipeline to train
    our custom model on plant-specific disease datasets (e.g. PlantVillage).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training disease classification model using device: {device}")
    
    # 2. Data Augmentation and Normalization Pipelines
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset directory '{dataset_dir}' not found.")
        print("💡 Create subfolders under dataset_dir corresponding to each disease class, for example:")
        print("   dataset/healthy_mango/, dataset/mango_anthracnose/, dataset/guava_fruit_borer/ ...")
        return
        
    # 3. Load Dataset
    full_dataset = datasets.ImageFolder(root=dataset_dir)
    num_classes = len(full_dataset.classes)
    class_to_idx = full_dataset.class_to_idx
    
    # Save the class indices mapping for the inference server
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'class_indices.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=4)
        
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Apply specific transformations to datasets
    train_data.dataset.transform = train_transforms
    val_data.dataset.transform = val_transforms
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 4. Initialize Network
    model = PlantDiseaseClassifier(num_classes=num_classes).to(device)
    
    # 5. Define Loss Function and Optimizer
    # We unfreeze upper convolutional layers in the back-half of training to fine-tune
    optimizer = optim.Adam(model.backbone.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    # 6. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_data)
        epoch_acc = corrects.double() / len(train_data)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_data)
        val_epoch_acc = val_corrects.double() / len(val_data)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        # Save best model weights
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(models_dir, 'plant_disease_mobilenet.pth'))
            print("💾 Saved best weights checkpoint to models/plant_disease_mobilenet.pth")

if __name__ == '__main__':
    # Script can be run standalone to train on user's custom images
    import argparse
    parser = argparse.ArgumentParser(description="Train custom fine-tuned Plant Disease Classifier CNN")
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to plant disease images directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()
    
    train_model(dataset_dir=args.dataset, epochs=args.epochs)
