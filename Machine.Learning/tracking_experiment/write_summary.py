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
images_path = Path("data/pizza_steak_sushi")
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
#old way ::
# model = torchvision.models.efficientnet_b0(pretrained=True)
#new way ::
model = torchvision.models.efficientnet_b0(weights=weights)
#freeze the base layers
for param in model.features.parameters():
    param.requires_grad=False
#create our custom classifier
from torch import nn
print(model.classifier)
classifier = nn.Sequential(
    nn.Dropout(p=0.2 , inplace = True),
    nn.Linear(in_features=1280,out_features=len(class_names))
)
model.classifier=classifier
print(model.classifier)

#use tensor board and track any model performance :::
#choose loss function and optimizer for training our model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),lr=0.01)
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="runs/experiment1")
from going_modular.engine import create_writer
writer = create_writer("pizza_steak_sushi","effnet_b0")
from going_modular.engine import train_write
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = train_write(model,train_dataloader,test_dataloader,optimizer,loss_fn,3,device=device,writer=writer)