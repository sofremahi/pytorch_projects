import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#get the data ready (data , labels) for our computing
file_path = 'bc.data'  
data = pd.read_csv(file_path, header=None)
data = data.dropna()
num_columns = data.shape[1]
columns = [
    "ID", "Diagnosis", "Radius", "Texture", "Perimeter", "Area", "Smoothness",
    "Compactness", "Concavity", "ConcavePoints", "Symmetry", "FractalDimension",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE",
    "Compactness_SE", "Concavity_SE", "ConcavePoints_SE", "Symmetry_SE",
    "FractalDimension_SE", "Radius_Worst", "Texture_Worst", "Perimeter_Worst",
    "Area_Worst", "Smoothness_Worst", "Compactness_Worst", "Concavity_Worst",
    "ConcavePoints_Worst", "Symmetry_Worst", "FractalDimension_Worst"
][:num_columns]
data.columns = columns
data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": 0})

if "ID" in data.columns:
    data = data.drop("ID", axis=1)

X = data.iloc[:, 1:].values  
y = data.iloc[:, 0].values   
#create test and train data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#turning data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
#initialize dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

from model import NeuralNetwork , evaluate_model , train_model

input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=50)
evaluate_model(model, test_loader)
