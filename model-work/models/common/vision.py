import torch
import torch.nn as nn
from .layers import LayerNorm, PatchEmbed, clone, Block, Causal_Mask,MultiHeadAttention,MLP,SublayerConnection


class VisionTransformer(nn.Module):
    def __init__(self,embed_dim,patch_embed,block,dropout,proj_dim=512):
        super().__init__()
        self.patch_embed=patch_embed
        self.pos_drop=nn.Dropout(dropout)
        self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim)*0.02)
        self.pos_embed=nn.Parameter(torch.randn(1,patch_embed.num_patchs+1,embed_dim)*0.02)
        self.blocks=clone(block,12)
        self.image_proj=nn.Linear(embed_dim,proj_dim)
        
    def forward_features(self,x):
        x=self.patch_embed(x)
        B=x.shape[0]
        cls_tokens=self.cls_token.expand(B,-1,-1)
        x=torch.cat([cls_tokens,x],dim=1)
        x=x+self.pos_embed
        for block in self.blocks:
            x=block(x)
        return x
    def forward_cls(self,x):
        x=self.forward_features(x)
        cls_features=x[:,0,:]
        return cls_features
    def forward(self,x):
        x=self.forward_cls(x)
        proj_features=self.image_proj(x)
        proj_features=proj_features/(torch.norm(proj_features,dim=-1,keepdim=True)+1e-10)
        return proj_features