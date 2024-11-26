import sys
import os
import torch
#downloading our data if no exist already
from pathlib import Path
import zipfile
import requests
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
#if the folder not existing downlaoding the data
if image_path.is_dir():
    print(f"path {image_path} already exists")
else:
    image_path.mkdir(parents=True , exist_ok=True)
    #downloading data
    with open(data_path/"pizza_steak_sushi.zip" , "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip" , verify=False)
        f.write(request.content)
    #extracting the file
    with open(data_path / "pizza_steak_sushi.zip" , "r") as zip_ref:
            zip_ref.extraclall(image_path)
    #removing zip file
    os.remove(data_path/"pizza_steak_suzhi.zip") 
    
train_path = image_path/"train"     
test_path = image_path/"test"  
 
def get_train_test_path():
    return train_path , test_path   