import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import going_modular
from going_modular.data_setup import create_dataloaders
from get_data import get_train_test_path
#device agnostic code 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#get our data paths
train_dir , test_dir = get_train_test_path()
#turning our data to datasets and data loaders
#defining pre trained models inside torchvision.models
# import torchvision.models

import torchvision
from torchvision import transforms
#transforming manually 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
manual_transform = transforms.Compose([
                                       transforms.Resize(size=(224,224)),
                                       transforms.ToTensor(),
                                       normalize #expected for our model which has its own data on imagenet
                                       ])
train_dataloader , test_dataloader , class_names = create_dataloaders(train_dir ,
                                                                                              test_dir ,
                                                                                              manual_transform,32, num_workers=0)
#automatic transform torch vision +0.13
#getting a set of pre trained models weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # best weights 
#creating automatic transfroms
auto_transform = weights.transforms()
#using our automatic created transform
train_dataloader , test_dataloader , class_names  = create_dataloaders(train_dir , test_dir ,auto_transform,32 , num_workers=0)
#getting a pre trained model
#old way of instantiating a pre trained model
# model = torchvision.models.efficientnet_b0(pretrained=True)
#new way of instantiating a model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)
#our pre trained model is efficient net _ b0
print(f"input and output of our pre trained instantiated model is {model.classifier}")
#before feature extraction lets get our model summary
import torchinfo
from torchinfo import summary
#print a summary
# print(summary(model=model,
#         input_size=(1, 3, 224, 224), # example of [batch_size, color_channels, height, width]
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]))


#freezing our base model layers and update the classifier head
for param in model.parameters():
    #freezing the base layers not updateble anymore
    param.requires_grad =False
#update the classifier layer 
from torch import nn
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2 , inplace=True),
    nn.Linear(in_features = 1280 , out_features=len(class_names))
).to(device)

#lets train our pre trained instantiated model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(),lr=0.01)
from going_modular.engine import train
# torch.cuda.manual_seed(20)
# torch.manual_seed(20)
#timing 
from timeit import default_timer as timer
start_time = timer()
results = train(model , train_dataloader,test_dataloader,optimizer,loss_fn,5,device)
end_time = timer()
print(f"total training time : {end_time-start_time}")

def get_trained_model():
    return model
def get_class_names():
    return class_names
def get_train_test_dataloader():
    return train_dataloader , test_dataloader
