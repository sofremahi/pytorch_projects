#replicating vision transformer 
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#replicating ViT for our food mini problem
from pathlib import Path
pizza_steak_sushi_path = Path("data/")/"pizza_steak_sushi"
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
#visualize a single image 
from matplotlib import pyplot as plt
#get a batch of images batches are 32 but this tensor has a extra deminstion of 1 
image_batch , image_label = next(iter(train_dataloader))
#get a single image and label through out the batch
image , label =  image_batch[0] , image_label[0]
plt.imshow(image.permute(1,2,0))
plt.title(class_names[label])
plt.axis(False)
# plt.show()

#recreating vision transformer architecture
#Exploring Figure 
# Machine.Learning\vision_transformer\vit-paper-figure-inputs-and-outputs.png
#equations
#1 x_input = [class_token, image_patch_1, image_patch_2, image_patch_3...] + [class_token_position, image_patch_1_position, image_patch_2_position, image_patch_3_position...]
#2 x_output_MSA_block = MSA_layer(LN_layer(x_input)) + x_input
#3 x_output_MLP_block = MLP_layer(LN_layer(x_output_MSA_block)) + x_output_MSA_block
#4 y = Linear_layer(LN_layer(x_output_MLP_block[0]))
#table of content
# Model	        Layers	Hidden size $D$	  MLP size	 Heads	 Params
# ViT-Base	      12	      768	        3072	  12      $86M$
# ViT-Large	      24	     1024	        4096	  16	  $307M$
# ViT-Huge	      32	     1280	        5120	  16	  $632M$
height = 224
width = 224 
color_channels = 3
patch_size = 16
num_of_patches = int((height*width)/patch_size**2)
embedding_input_shape = (height , width , color_channels)
embedding_output_shape = (num_of_patches , patch_size**2*color_channels)
print(f"input shape is {embedding_input_shape} -- output shape is {embedding_output_shape}")
print(image.shape)
permuted_image = image.permute(1,2,0) #--> (H,W,C)
print(permuted_image.shape)
plt.figure(figsize=(4 ,4 ))
plt.imshow(permuted_image[:patch_size,:,:]) #-> (16,224,3)
# plt.show()

#set up image as patches on to the first row of 16 pixels
img_size = 224
num_patches = img_size/patch_size
#assert that the num_patches will be a constant
# fig, axs = plt.subplots(nrows=img_size//patch_size,
#                         ncols=img_size // patch_size, 
#                         figsize=(8, 8),
#                         sharex=True,
#                         sharey=True)
# for i, patch in enumerate(range(0, img_size, patch_size)):
#     for j , inner_patch in enumerate(range(0,img_size,patch_size)):
#         axs[i, j].imshow(permuted_image[inner_patch:inner_patch + patch_size, patch:patch + patch_size, :])
#         axs[i, j].set_xlabel(f'{i * (img_size // patch_size) + j + 1}')
#         axs[i, j].set_xticks([])
#         axs[i, j].set_yticks([])
# plt.tight_layout()      
# Create a series of subplots
fig, axs = plt.subplots(nrows=1,
                        ncols=img_size // patch_size, # one column for each patch
                        figsize=(num_patches, num_patches),
                        sharex=True,
                        sharey=True)

# Iterate through number of patches in the top row
for i, patch in enumerate(range(0, img_size, patch_size)):
    axs[i].imshow(permuted_image[:patch_size, patch:patch+patch_size, :]); # keep height index constant, alter the width index
    axs[i].set_xlabel(i+1) # set the label
    axs[i].set_xticks([])
    axs[i].set_yticks([])  
plt.show()