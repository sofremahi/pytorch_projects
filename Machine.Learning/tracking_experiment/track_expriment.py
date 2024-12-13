#download two diffrent data set amounts for our expriment tracking
import sys
import os
import torch
import torchvision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from going_modular.resource import download_data
data_10_percent_path = download_data(url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                     folder_name="pizza_steak_sushi" , is_zip=True)

data_20_percent_path = download_data(url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     folder_name="pizza_steak_sushi_20_percent" , is_zip=True)
#train test paths for 10 percent data
train_dir_10_percent = data_10_percent_path/"train"
test_dir_10_percent = data_10_percent_path/"test"
#train test paths for 20 percent data
train_dir_20_percent = data_20_percent_path/"train"
test_dir_20_percent = data_20_percent_path/"test"
#create simple transform 
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
simple_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize
])
batch_size=32
from going_modular.data_setup import create_dataloaders
train_dataloader_10_percent , test_dataloader_10_percent , class_names = create_dataloaders(
    train_dir_10_percent,test_dir_10_percent,simple_transform,batch_size,num_workers=0
)
train_dataloader_20_percent , test_dataloader_20_percent , class_names = create_dataloaders(
    train_dir_20_percent,test_dir_20_percent,simple_transform,batch_size,num_workers=0
)
out_features = len(class_names)
from torch import nn
def create_effnet_b0():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model=torchvision.models.efficientnet_b0(weights=weights)
    #freeze base model layers
    for param in model.features.parameters():
        param.requires_grad=False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280 , out_features = out_features)
    )    
    return model
def create_effnet_b2():
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model=torchvision.models.efficientnet_b2(weights=weights)
    #freeze base model layers
    for param in model.features.parameters():
        param.requires_grad=False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3,inplace=True),
        nn.Linear(in_features=1408 , out_features = out_features)
    )    
    return model
model_b0 = create_effnet_b0()
model_b2 = create_effnet_b2()
#created nested loops for our expriment tracking 
experiment_number = 0
epochs_list = [5 , 10]
models=["model_b0","model_b2"]
train_data_loaders={"10_percent":train_dataloader_10_percent,
                    "20_percent":train_dataloader_20_percent}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from going_modular.engine import train , create_writer
from going_modular.utils import save_model
for dataloader_name , train_dataloader in train_data_loaders.items():
    for model_name in models:
        for epochs in epochs_list:
            #print out the details 
            experiment_number += 1
            print(f"[INFO] Experiment number: {experiment_number}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] DataLoader: {dataloader_name}")
            print(f"[INFO] Number of epochs: {epochs}")  
            
            if model_name == "model_b0":
                model = create_effnet_b0()
            else:
                model = create_effnet_b2()
                
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)    
        train(model=model,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader_10_percent, 
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  epochs=epochs,
                  device=device,
                  writer=create_writer(experiment_name=dataloader_name,
                                       model_name=model_name,
                                       extra=f"{epochs}_epochs"))
        save_filepath = f"{model_name}_{dataloader_name}_{epochs}_epochs.pth"
        save_model(model=model,
                       target_dir="models",
                       model_name=save_filepath)
