import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


import sys
import os # adding filepaths list from data_processing directory

# Import the ANNModel from model_definitions.py
from model_definitions import ANNModel


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data_processing import load_data
y_test = load_data.y_test
y_train = load_data.y_train




X_train = pd.read_csv('/Users/adamclarke/Desktop/Data/sentence-classifications/data/X_train.csv')
X_test = pd.read_csv('/Users/adamclarke/Desktop/Data/sentence-classifications/data/X_test.csv')
# y_test = pd.read_csv('/Users/adamclarke/Desktop/Data/sentence-classifications/data/y_test.csv')
# y_train = pd.read_csv('/Users/adamclarke/Desktop/Data/sentence-classifications/data/y_train.csv')

# Convert arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

print(y_train_tensor.shape)
print(X_train_tensor.shape)

# Create TensorDataset and DataLoader for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)


# Initialize the model, loss function, and optimizer
model = ANNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Training the model
num_epochs = 700
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Save the model state
torch.save(model.state_dict(), 'model_state_dict.pth')


# Testing the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
