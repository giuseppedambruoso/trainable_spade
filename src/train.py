import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import logging
from .model import QuantumOpticsClassifier

logger = logging.getLogger(__name__)

def train_and_evaluate(model: QuantumOpticsClassifier, train_loader, test_loader, epochs: int, lr: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    metrics = []
    
    logger.info(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        metrics.append({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_acc})
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
        
    # Testing
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    final_test_acc = test_correct / test_total
    logger.info(f"Final Test Accuracy: {final_test_acc:.4f}")
    
    return pd.DataFrame(metrics), final_test_acc
