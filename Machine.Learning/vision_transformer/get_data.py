#using our pather model to create embeddings with an image
import sys
import os
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
pizza_steak_sushi_path = Path("Machine.Learning/data/")/"pizza_steak_sushi"
print(f"path of our already downloaded data : {pizza_steak_sushi_path}")
train_dir = pizza_steak_sushi_path/"train"
test_dir = pizza_steak_sushi_path/"test"
from going_modular.data_setup import create_dataloaders
import torchvision
from torchvision import transforms
transformer = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])
train_dataloader , test_dataloader , class_names = create_dataloaders(train_dir ,test_dir,transformer,batch_size=32,num_workers=0)
image_batch , image_label = next(iter(train_dataloader))
#get a single image and label through out the batch remove batch extra dimention
image , label =  image_batch[0] , image_label[0]
def get_image_and_label():
    return image , label
def get_data_loaders():
    return train_dataloader , test_dataloader , class_names
def get_train_test_dir():
    return train_dir , test_dir
    


