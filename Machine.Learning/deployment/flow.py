from pathlib import Path
import sys
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from going_modular import data_setup , engine
pizza_steak_sushi_path = Path("Machine.Learning/data/")/"pizza_steak_sushi_20_percent"
print(f"path of our already downloaded data : {pizza_steak_sushi_path}")
train_dir = pizza_steak_sushi_path/"train"
test_dir = pizza_steak_sushi_path/"test"
from pre_trained import create_effnetb2_model
#create efficient net b2 
effnet_b2 , effnet_b2_transforms = create_effnetb2_model(num_classes=3)
#create data loaders
effnet_train_dataloader , effnet_test_dataloader , class_names = data_setup.create_dataloaders(train_dir,test_dir,effnet_b2_transforms,batch_size=32 , num_workers=0) 
# optimizer
optimizer = torch.optim.Adam(params=effnet_b2.parameters(),
                             lr=1e-3)
# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Set seeds for reproducibility and train the model
effnetb2_results = engine.train(model=effnet_b2,
                                train_dataloader=effnet_train_dataloader,
                                test_dataloader=effnet_test_dataloader,
                                epochs=5,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                device=device)
from helper_functions import plot_loss_curves
from matplotlib import pyplot as plt
plot_loss_curves(effnetb2_results)
# plt.show()

#create vit 16
from pre_trained import create_vit_model
vit, vit_transforms = create_vit_model(num_classes=len(class_names))
# Setup ViT DataLoaders
train_dataloader_vit, test_dataloader_vit, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                       test_dir=test_dir,
                                                                                       transform=vit_transforms,
                                                                                       batch_size=32)
#train our vit pre trained model
#  optimizer
optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=1e-3)
# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Train ViT
vit_results = engine.train(model=vit,
                           train_dataloader=train_dataloader_vit,
                           test_dataloader=test_dataloader_vit,
                           epochs=5,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           device=device)
plot_loss_curves(vit_results)
plt.show()