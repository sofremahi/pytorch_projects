import torch
from torch import nn
class PatchEmbeddings(nn.Module): #(3,224,224)->(1,196,768)
    def __init__(self , in_channels:int=3 , patch_size:int=16 ,embedding_dim:int=768 #from table 1 ViT_base
                 ):
        super().__init__()
        #create needed layers
        self.patch_size = patch_size
        self.patch = nn.Conv2d(in_channels=in_channels,out_channels=embedding_dim,
                               kernel_size=patch_size,
                               stride=patch_size , padding=0) #-> (1,768,14,14)
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)
    def forward(self , x):
         # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        return self.flatten(self.patch(x)).permute(0,2,1)  