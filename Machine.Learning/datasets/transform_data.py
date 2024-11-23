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
from helper import plot_transformed_images
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
#visualize our train data 
img , label = train_data[0][0] , train_data[0][1]
print(f" the shape of our image tensor is {img.shape} and the lable is {label}")
img_permute = img.permute(1, 2, 0)

# diffrent shapes
from matplotlib import pyplot as plt
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute: {img_permute.shape} -> [height, width, color_channels]")
# Plot the image
# plt.figure(figsize=(10, 7))
# plt.imshow(img_permute)
# plt.axis("off")
# plt.title(class_names[label], fontsize=14)
# plt.show()

#creating data loaders
import os
cores = os.cpu_count()
BATCH_SIZE = 32
train_data_loader = DataLoader(dataset=train_data, 
                                   batch_size=BATCH_SIZE, 
                                   num_workers=0, 
                                   shuffle=True)

test_data_loader = DataLoader(dataset=test_data, 
                                  batch_size=BATCH_SIZE, 
                                  num_workers=0, 
                                  shuffle=False)
img , label = next(iter(train_data_loader))
print(f"image shape is : {img.shape} [batch , color channels , height , width]")
print(f"label shape is : {label.shape} [batch size * label(1)]")

