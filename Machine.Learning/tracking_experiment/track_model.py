#creating datalaoders with transforms 1. manual approach 2. automatic approach
#1.
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torchvision
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
manual_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize
])
from pathlib import Path
images_path = Path("Machine.Learning/data/pizza_steak_sushi")
from going_modular.data_setup import create_dataloaders
train_dataloader,test_dataloader,class_names= create_dataloaders(
    images_path/"train",images_path/"test",manual_transform,num_workers=0 ,batch_size=32
)
#2.
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT #best available
#get transform from weights
auto_transform = weights.transforms()
#data loaders
train_dataloader,test_dataloader,class_names= create_dataloaders(
    images_path/"train",images_path/"test",auto_transform,num_workers=0 ,batch_size=32
)
#instantaite a pre trained model and freeze the base layers for computation
