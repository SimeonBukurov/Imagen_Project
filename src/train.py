import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import MultitaskFaceModel, get_device
from dataset import get_data_loader

def train():
    device = get_device()
    print(f"Training on device: {device}")
    
    model = MultitaskFaceModel().to(device)
    
    # --- UPDATE THESE PATHS IF NECESSARY ---
    data_path = r"D:\Imagen Classes UPV\Project\data\raw"
    attr_path = r"D:\Imagen Classes UPV\Project\data\raw\list_attr_celeba.txt" 
    
    loader = get_data_loader(data_path, attr_path)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy grades answers from 0.0 to 1.0

    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # The AI makes its guesses
        pred_gender, pred_hair, pred_age = model(data)
        
        # We grade the AI against the real answers
        # Target Array layout: [Gender, Black, Blond, Brown, Gray, Age]
        loss_gender = criterion(pred_gender, target[:, 0:1]) 
        loss_hair = criterion(pred_hair, target[:, 1:5]) 
        loss_age = criterion(pred_age, target[:, 5:6]) 
        
        # Combine the grades into one master score
        total_loss = loss_gender + loss_hair + loss_age
        
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Total Loss: {total_loss.item():.4f} (Gender: {loss_gender.item():.4f}, Hair: {loss_hair.item():.4f}, Age: {loss_age.item():.4f})")

    save_path = r"D:\Imagen Classes UPV\Project\models\custom_face_model_v1.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Epoch Complete! Model saved successfully to {save_path}")

if __name__ == "__main__":
    train()