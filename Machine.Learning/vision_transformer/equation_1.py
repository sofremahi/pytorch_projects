#from single image to embeddings
from Vit_model_use import get_image_and_label
image , label = get_image_and_label()

patch_size = 16
print(f"our image shape is {image.shape}")#torch.Size([3, 224, 224])
height , width = image.shape[1] , image.shape[2]
#add a batch dimention to image
image_with_batch = image.unsqueeze(dim=0) #([1, 3, 224, 224])
#creating patch embedding layer
from Vit_model import PatchEmbeddings
patch_embedding_layer = PatchEmbeddings(in_channels=3,patch_size=patch_size,embedding_dim=768) #according to table 1
#pass image
patch_embedding = patch_embedding_layer(image_with_batch)#torch.Size([1, 196, 768])
#adding class token and position embeddings
#create class token
batch_size = patch_embedding.shape[0] #  = 1
embedding_dimention = patch_embedding.shape[-1] #  = 768
import torch
from torch import nn
class_token =nn.Parameter(torch.ones(batch_size , 1 , embedding_dimention), 
                         requires_grad=True) # making it learnable  #torch.Size([1, 1, 768])
#prepend the class token to patch embedding
patch_embedding_with_class_token = torch.cat((class_token,patch_embedding) , dim=1) #torch.Size([1, 197, 768])
#create position embeddings 
num_patches = int((height*width)/patch_size**2)
position_embedding = nn.Parameter(torch.ones(1 , num_patches+1 , embedding_dimention),requires_grad = True)#torch.Size([1, 197, 768])
#add position embedding to patch embeddings with class token
patch_token_position_embedding = patch_embedding_with_class_token +  position_embedding # adding values to our tensor  torch.Size([1, 197, 768])

def image_final_patch_embedding():
    return patch_token_position_embedding