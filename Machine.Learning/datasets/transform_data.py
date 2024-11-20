from become_1with_data import get_image , get_image_path_list ,get_train_dir ,get_test_dir
image = get_image()
image_paths = get_image_path_list()
test_dir = get_test_dir()
train_dir = get_train_dir()
# print(image_as_array)
import torch
from torch.utils.data import DataLoader
from torchvision import datasets , transforms
#transforming data with torch vision transform
data_transform = transforms.Compose([
    #resizing our image
    transforms.Resize(size=(64,64)),
    #fliping our images randomly up tp 50%
    transforms.RandomHorizontalFlip(p=0.5),
    #turn images to tensors
    transforms.ToTensor()
])
print(data_transform(image).shape)
import random
from PIL import Image
from matplotlib import pyplot as plt
def plot_transformed_images(image_paths , transform , n=3 , seed =None):
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)   
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
         fig , ax = plt.subplots(nrows=1 , ncols=2)
         ax[0].imshow(f)
         ax[0].set_title(f"Original\nSize: {f.size}")
         ax[0].axis(False)
         transformed_image = transform(f).permute(1, 2, 0) #  (C, H, W) -> (H, W, C)
         ax[1].imshow(transformed_image)
         ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
         ax[1].axis("off")

        fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
# plot_transformed_images(image_paths=image_paths,
#                         transform=data_transform,
#                         n=3,
#                         seed=None)
# plt.show()    

#instantialing our train and test data 
train_data = datasets.ImageFolder(root=train_dir,
                                  transform = data_transform , 
                                  target_transform = None)
test_data = datasets.ImageFolder(root=test_dir,
                                  transform = data_transform , 
                                  target_transform = None)
class_names = train_data.classes
class_dict = train_data.class_to_idx
print(class_dict)