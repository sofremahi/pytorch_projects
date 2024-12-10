
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