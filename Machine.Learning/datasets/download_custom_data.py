import requests
from pathlib import Path

data_path = Path("data/")
custom_image_path = data_path / "04-pizza-dad.jpeg"
if not custom_image_path.is_file():
    with open(custom_image_path , "wb")as f :
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg" , verify=False)
        f.write(request.content)
else:
    print(f"the path of {custom_image_path} already exists")        
        
        