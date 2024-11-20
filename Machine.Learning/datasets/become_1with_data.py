import os
from pathlib import Path

image_path = Path("data/") / "pizza_steak_sushi"

def explor_directory(dir_path):
    """" returning a path contents """
    for dirpath , dirnames , filenames in os.walk(dir_path):
     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")    
explor_directory(image_path)    

train_dir = image_path / "train"
test_dir = image_path / "test"
# lets see our images 
import random
from PIL import Image
#create list of image paths in a patterned order
image_path_list = list(image_path.glob("*/*/*.jpg"))
#choose a random image path 
random_image_path = random.choice(image_path_list)
print(random_image_path)
#get the image class 
image_class = random_image_path.parent.stem
print(image_class)
#open image
image = Image.open(random_image_path)
# image.show()
print(f"this is a picture of a {image_class} with a height of : {image.height} and width of : {image.width} ")   

#visualising with matplotlib
from matplotlib import pyplot as plt
import numpy as np
#turn to byte array
image_as_array = np.asarray(image)
# plotting the image
# plt.figure(figsize=(10,9))
# plt.imshow(image_as_array)
# plt.axis(False)
# plt.title(f"image class is : {image_class} image shape height , width : {image.height} , {image.width} image shape : {image_as_array.shape} HWC")
#color channels are last likely to note 
# plt.show()
def get_image_as_array():
    return image_as_array
def get_image():
    return image
def get_image_path_list()->list:
    return image_path_list
def get_train_dir():
    return train_dir
def get_test_dir():
    return test_dir