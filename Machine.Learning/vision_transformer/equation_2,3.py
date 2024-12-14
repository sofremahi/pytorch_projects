
from equation_1 import image_final_patch_embedding
#equation 2

patch_embedding_from_equation_1 = image_final_patch_embedding()
#create the multi head self attetion layer
from Vit_model import MultiHeadSelfAttentionBlock
multihead_self_attention_block = MultiHeadSelfAttentionBlock(embedding_dim=768,num_heads=12,attn_dropout=0)
#pass patch embedding to multi head self attention block
patched_image_to_msa_block = multihead_self_attention_block(patch_embedding_from_equation_1) #torch.Size([1, 197, 768])\
    
    
#equation 3    
# linear layer -> non-linear layer -> linear layer -> non-linear layer
#create instance of MLP block
from Vit_model import MLPBlock
mlp_block = MLPBlock(embedding_dim=768 , mlp_size=3072,dropout=0.1)
patched_image_through_mlp = mlp_block(patched_image_to_msa_block + patch_embedding_from_equation_1 ) #torch.Size([1, 197, 768])


#to the next layer we pass ==> patched_image_through_mlp + patched_image_to_msa_block + patch_embedding_from_equation_1

#EQUATIONS 2 AND 3 IN A SINGLE LINE
from Vit_model import TransformerEncoder
transformer_encoder_block = TransformerEncoder()


#equations 2 and 3 with in build pytorch methods
#table of content
# Model	        Layers	Hidden size $D$	  MLP size	 Heads	 Params
# ViT-Base	      12	      768	        3072	  12      $86M$
# ViT-Large	      24	     1024	        4096	  16	  $307M$
# ViT-Huge	      32	     1280	        5120	  16	  $632M$
import torch
from torch import nn
transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = 768 , #embedding size from table 
                                                       nhead = 12 ,#head from table 1 for ViT-base
                                                       dim_feedforward = 3072 , #MLP size from table 
                                                       dropout=0.1,# Amount of dropout for dense layers from Table for ViT-base
                                                       activation="gelu",# GELU non-linear activation
                                                       batch_first=True, # our batches come first?
                                                       norm_first=True, # Normalize first or after MSA/MLP layers
                                                       )