
#prepare data
from get_data import get_image_and_label , get_data_loaders , get_train_test_dir
train_dataLoader , test_dataLoader , class_names = get_data_loaders()
single_image , single_image_label = get_image_and_label()
train_dir , test_dir = get_train_test_dir()
#instantiate out ViT implemented model
from Vit_model import ViT
vit = ViT()
#get a summary of our implemented ViT model
import torch
from torch import nn
# dummy_input = torch.randn(32, 3, 224, 224)  # Adjust batch size and dimensions as needed
# output = vit(dummy_input)
# print(output.shape)
# from torchinfo import summary
# print(summary(model=vit,
#         input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# ))

#initialize an optimizer and loss funstion to train our model

optimizer = torch.optim.Adam( vit.parameters(),lr=1e-3,betas=(0.9,0.999) ,weight_decay=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
#train our model

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from going_modular import engine
#poor results probably beacuse of small scale of data
# results = engine.train(vit,train_dataLoader,test_dataLoader,optimizer,loss_fn,10,device)

#get pre trained models weights and parameters to suit(customize) our needs
import torchvision
pre_trained_vit_weighs = torchvision.models.ViT_B_16_Weights.DEFAULT # best available
#set up vit model with pre trained weights
pre_trained_vit = torchvision.models.vit_b_16(weights=pre_trained_vit_weighs).to(device)
#freeze the base parameters
for parameter in pre_trained_vit.parameters():
    parameter.requires_grad=False
pre_trained_vit.heads = nn.Linear(in_features = 768 , out_features=len(class_names))
#get the pre trained model transforms
vit_transforms = pre_trained_vit_weighs.transforms()
from going_modular import data_setup
train_dataLoader_pretrained , test_dataLoader_pretrained , class_names = data_setup.create_dataloaders(
    train_dir,test_dir,vit_transforms,32,num_workers=0)
#choose a loss function and an optimzier
optimizer = torch.optim.Adam(params=pre_trained_vit.parameters(),
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
pre_trained_vit_results = engine.train(pre_trained_vit,train_dataLoader,test_dataLoader,optimizer,loss_fn,20,device)

#save our best model
from going_modular import utils
pre_trained_vit_save_path = utils.save_model(pre_trained_vit,"Machine.Learning/models" , "pre_trained_vit.pth")
def get_pre_trained_vit_path():
    return pre_trained_vit_save_path