import torch
from torch import nn
from get_data import get_image_and_label , get_data_loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ ,_,class_names = get_data_loaders()
single_image , single_iamge_label = get_image_and_label()
import torchvision

#instantiating our pre trained model for loading the state dict with suitable classifier head
pre_trained_vit_weighs = torchvision.models.ViT_B_16_Weights.DEFAULT # best available
loading_vit_model = torchvision.models.vit_b_16(weights=pre_trained_vit_weighs).to(device)
for parameter in loading_vit_model.parameters():
    parameter.requires_grad=False
loading_vit_model.heads = nn.Linear(in_features = 768 , out_features=len(class_names))


loading_vit_model.load_state_dict(torch.load("Machine.Learning/models/pre_trained_vit.pth" , weights_only=True))
from going_modular import predictions
from pathlib import Path
custom_image_path = Path("Machine.Learning/data/random_sushi.jpeg")
predictions.pred_and_plot_image(
    loading_vit_model,class_names,custom_image_path)
from matplotlib import pyplot as plt
plt.show()
