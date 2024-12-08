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
# plt.imshow(torch.randn(224,224,3))
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
# Create a series of subplots
# fig, axs = plt.subplots(nrows=1,
#                         ncols=img_size // patch_size, # one column for each patch
#                         figsize=(6, 6),
#                         sharex=True,
#                         sharey=True)

# Iterate through number of patches in the top row
# for i, patch in enumerate(range(0, img_size, patch_size)):
#     axs[i].imshow(permuted_image[:patch_size, patch:patch+patch_size, :]);
#     axs[i].set_xlabel(i+1) 
#     axs[i].set_xticks([])
#     axs[i].set_yticks([])  
# plt.show()



#iterate through the whole image




# fig, axs = plt.subplots(nrows=img_size // patch_size,
#                         ncols=img_size // patch_size,
#                         figsize=(num_patches, num_patches),
#                         sharex=True,
#                         sharey=True)

# Loop through height and width of image
# for i, patch_height in enumerate(range(0, img_size, patch_size)):
#     for j, patch_width in enumerate(range(0, img_size, patch_size)): 

    
#         axs[i, j].imshow(permuted_image[patch_height:patch_height+patch_size,
#                                         patch_width:patch_width+patch_size, 
#                                         :]) 
#         axs[i, j].set_ylabel(i+1,
#                              rotation="horizontal",
#                              horizontalalignment="right",
#                              verticalalignment="center")
#         axs[i, j].set_xlabel(j+1)
#         axs[i, j].set_xticks([])
#         axs[i, j].set_yticks([])
#         axs[i, j].label_outer()

# # Set a super title
# fig.suptitle(f"{class_names[label]} -> patched", fontsize=16)
# plt.show()     

#create a conv2d layer 
from torch import nn
conv2d = nn.Conv2d(in_channels = 3, #color channels
                    out_channels = 768 ,#hidden size from table 1
                    kernel_size = patch_size,
                    stride=patch_size,
                    padding=0
                   )
image_out_of_conv = conv2d(image.unsqueeze(dim=0))
print(image_out_of_conv.shape)
import random
random_indexes = random.sample(range(0,768) , k=5)
fig ,axs = plt.subplots(nrows=1,ncols=5,figsize=(6,6))
for i,idx in enumerate(random_indexes):
    image_conv_feature_map = image_out_of_conv[:,idx,:,:] #torch.Size([1, 768, 14, 14])
    axs[i].imshow(image_conv_feature_map.squeeze(dim=0).detach().numpy())
    axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
# plt.show()
print(f"faltting the shape of the embeddings : {image_out_of_conv[:,0,:,:].shape} to suit our needs of 14*14 flatten info ")
#we want to create a tensor of (196,768) or (1,196,768) a batch size of 1 additional demention
#creating the flatten layer for reshaping torch.Size([1, 768, 14, 14])
flatten_layer = nn.Flatten(start_dim=2,end_dim=3)
print(f"our final tensor shape with use of flatten layer is {flatten_layer(image_out_of_conv).shape}") # torch.Size([1, 768, 196])
image_out_of_conv_flatten_arranged = flatten_layer(image_out_of_conv).permute(0,2,1) # [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
print(f"we need the re arranged format of flatten layer to shape : {image_out_of_conv_flatten_arranged.shape}") # torch.Size([1, 196, 768])
print(f"Patch embedding sequence shape: {image_out_of_conv_flatten_arranged.shape} -> [batch_size, num_patches, embedding_size]")
plt.figure(figsize=(6,6))
plt.imshow(image_out_of_conv_flatten_arranged[:,:,0].detach().numpy())
plt.axis(False)
# plt.show()


#configuration of class token for equation 1 
#(1,196,768) - > append a learnable sequence -> (1,197,768) adding a extra nn.parameter

# Get the batch size and embedding dimension
batch_size = image_out_of_conv_flatten_arranged.shape[0] #=1
embedding_dimension = image_out_of_conv_flatten_arranged.shape[-1]
print(f"batch size is {batch_size} and the embedding dimention (output) is {embedding_dimension}")
class_token = nn.Parameter(torch.ones(batch_size , 1 ,embedding_dimension),
                           requires_grad=True)
print(f"class token shape {class_token.shape}")
print(f"patch embedded image shape {image_out_of_conv_flatten_arranged.shape}")

patch_embedded_image_with_class_embedding = torch.cat((class_token, image_out_of_conv_flatten_arranged),
                                                      dim=1) 
print(f"patch embedded iamge with class token shape {patch_embedded_image_with_class_embedding.shape}") # [batch_size, class_token+number_of_patches, embedding_dimension]

#creating position embeddings 
num_of_patches = (height*width)/(patch_size**2)
embedding_dimention = image_out_of_conv_flatten_arranged.shape[-1]
position_embeddings = nn.Parameter(torch.ones(1 , num_of_patches+1,embedding_dimension),
                           requires_grad=True)