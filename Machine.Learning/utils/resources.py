import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("it already exists")
else:
    print("downloading helper_functions.py")   
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
                           , verify=False)  
    with open("helper_functions.py" , "wb") as f:
        f.write(request.content)   
        