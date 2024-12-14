#downloading our custom datasets 
from pathlib import Path
import requests
import zipfile

#creating directory
data_path = Path("Machine.Learning/data/")
image_path = data_path / "pizza_steak_sushi"
if image_path.is_dir():
    print(f"directory {image_path} already exists")
else:
    image_path.mkdir(parents=True,exist_ok =True)
    
#downloading resources     
with open( data_path / "pizza_steak_sushi.zip" ,"wb") as f :
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip" , verify=False)
    f.write(request.content)
#extracting the zip file 
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
  print("Unzipping pizza, steak and sushi data...")
  zip_ref.extractall(image_path)