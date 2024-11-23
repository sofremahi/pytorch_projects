import torch
from torch.utils.data import DataLoader
from torchvision import datasets , transforms
from become_1with_data import get_train_dir , get_test_dir , get_image_path_list
from helper import find_classes_in_directory , ImageFolderCustom 
import pathlib 
import os
#replicating generating tensor data from directory path in a formatted approach

#replicating class and idx_classes for a directory of train or test directory
train_dir = get_train_dir()
test_dir = get_test_dir()
print(find_classes_in_directory(train_dir))

#replciating datasets.ImageFolder for transforming images in a directory path to tensors
#creating transform function 
train_transforms = transforms.Compose([
                                      transforms.Resize(size=(64, 64)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor() 
])

test_transforms = transforms.Compose([
                                      transforms.Resize(size=(64, 64)),
                                      transforms.ToTensor()
])
#using our replicated ImageFolderCustom 
train_data_custom= ImageFolderCustom(directory=train_dir , transform=train_transforms )
test_data_custom = ImageFolderCustom(test_dir , test_transforms)
print(f"shape of the first data is : {train_data_custom[0][0].shape} and the label is : {test_data_custom[0][1]}")
#visualizimg our images from out custom ImageFolder 
from helper import display_random_images
# display_random_images(train_data_custom , train_data_custom.classes , n =4)

#generating data loaders with our custom ImageFolder data retrieve
train_data_loader_custom = DataLoader(dataset = train_data_custom ,
                                      batch_size = 32 ,
                                      num_workers = 0,
                                      shuffle = True)
test_data_loader_custom = DataLoader(dataset = test_data_custom ,
                                      batch_size = 32 ,
                                      num_workers=0,
                                      shuffle = False)
imge , label = next(iter(train_data_loader_custom))
print(f"our x on train data loader shape is : {imge.shape} and the y shape is {label.shape}")

#testing transfom of trivialaugmentwide for manipulating our data to the highest scale
#so our model will learn from every pattern
train_transform_TAW = transforms.Compose([
                                      transforms.Resize(size=(64, 64)),
                                      transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                      transforms.ToTensor()
])
#lets diplay some pictures that ernt through our TAW transform manipulation (State of the Art)SOTA
from helper import plot_transformed_images
# plot_transformed_images(
#     image_paths=get_image_path_list(),
#     transform=train_transform_TAW,
#     n=3,
#     seed=None
# )
train_data_custom_TAW = ImageFolderCustom(directory=train_dir , transform=train_transform_TAW )
train_data_loader_custom_TAW = DataLoader(dataset = train_data_custom_TAW ,
                                      batch_size = 32 ,
                                      num_workers = 0,
                                      shuffle = True)

def get_train_data_loader_custom():
    return train_data_loader_custom
def get_test_data_loader_custom():
    return test_data_loader_custom
def get_train_data_loader_custom_TAW():
    return train_data_loader_custom_TAW

