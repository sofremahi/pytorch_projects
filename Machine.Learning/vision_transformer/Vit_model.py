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
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self , embedding_dim :int=768, #hidden size in table
                num_heads:int=12 , #heads from table
                attn_dropout :int=0) :
        super().__init__()    
        #create the layer norm (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)
        #create multi head attention layer (MSA)
        self.multihead_attn = nn.MultiheadAttention(embed_dim = embedding_dim , 
                                                    num_heads= num_heads ,
                                                    dropout = attn_dropout ,
                                                    batch_first= True) # (batch,seq,feature)
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, # query embeddings
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output
class MLPBlock(nn.Module):
    def __init__(self , embedding_dim :int=768,mlp_size:int=3072 , dropout:int=0.1):
        super().__init__()
        #create norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        #create MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), 
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, 
                      out_features=embedding_dim), 
            nn.Dropout(p=dropout) 
        )
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x   
class TransformerEncoder(nn.Module):
    """Creates a Transformer Encoder block."""
# Model	        Layers	Hidden size $D$	  MLP size	 Heads	 Params
# ViT-Base	      12	      768	        3072	  12      $86M$
# ViT-Large	      24	     1024	        4096	  16	  $307M$
# ViT-Huge	      32	     1280	        5120	  16	  $632M$
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 1 for ViT-Base
                 attn_dropout:float=0): # Amount of dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        # 4. Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)

    def forward(self, x):
        
        x =  self.msa_block(x) + x
        
        x = self.mlp_block(x) + x
        
        return x