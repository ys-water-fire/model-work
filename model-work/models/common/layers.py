import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.Module):
    def __init__(self,embed_dim,eps=1e-6):
        super().__init__()
        self.embed_dim=embed_dim
        self.eps=eps
        self.a=nn.Parameter(torch.ones(embed_dim))
        self.b=nn.Parameter(torch.zeros(embed_dim))
    def forward(self,x):
        x_mean=torch.mean(x,dim=-1,keepdim=True)
        x_std=torch.std(x,dim=-1,keepdim=True)
        return self.a*(x-x_mean)/(x_std+self.eps)+self.b


class PatchEmbed(nn.Module):
    def __init__(self,img_size,patch_size,in_c,embed_dim):
        super().__init__()
        self.img_size=(img_size,img_size)
        self.patch_size=(patch_size,patch_size)
        self.in_c=in_c
        self.embed_dim=embed_dim
        self.grid_size=(img_size//patch_size,img_size//patch_size)
        self.num_patchs=self.grid_size[0]*self.grid_size[1]
        self.proj=nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm_layer=LayerNorm(embed_dim)
    def forward(self,x):
        B,C,H,W=x.shape
        x=self.proj(x).flatten(2).transpose(1,2)
        x=self.norm_layer(x)
        return x

def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def Causal_Mask(size):
    b=torch.ones((1,size,size),dtype=torch.long)
    mask=1-torch.triu(b,diagonal=1)
    return mask.to(device)

class MultiHeadAttention(nn.Module):
    def __init__(self,head,embed_dim,dropout):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.head=head
        self.embed_dim=embed_dim
        assert embed_dim%head==0
        self.d_k=embed_dim//head
        self.linears=clone(nn.Linear(embed_dim,embed_dim),4)
    def forward(self,query,key,value,mask=None):
        self.batch_size=query.size(0)
        query,key,value=[Model(x).view(self.batch_size,-1,self.head,self.d_k).transpose(1,2)
                          for Model,x in zip(self.linears,(query,key,value))]
        atten=torch.matmul(query,key.transpose(-1,-2)/math.sqrt(self.d_k))
        if mask is not None:
            if mask.dim()==3:
                 mask=mask.unsqueeze(1)
            atten=atten.masked_fill(mask==0,-1e9)

        atten_score=F.softmax(atten,dim=-1)
        atten_result=torch.matmul(atten_score,value)
        result=atten_result.transpose(1,2).contiguous().view(self.batch_size,-1,self.head*self.d_k)
        result=self.dropout(result)
        return self.linears[-1](result)
       
class MLP(nn.Module):
    def __init__(self,embed_dim,hidden_dim,dropout):
        super().__init__()
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.dropout=nn.Dropout(dropout)
        self.linear1=nn.Linear(embed_dim,hidden_dim)
        self.linear2=nn.Linear(hidden_dim,embed_dim)
        self.act=nn.GELU()
    def forward(self,x):
        x=self.linear1(x)
        x=self.act(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return self.dropout(x)

class SublayerConnection(nn.Module):
    def __init__(self,embed_dim,dropout_p):
        super().__init__()
        self.norm=LayerNorm(embed_dim)
        self.dropout=nn.Dropout(p=dropout_p)
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))
    
class Block(nn.Module):
    def __init__(self,mlp,attn,dropout,embed_dim):
        super().__init__()
        self.mlp=mlp
        self.attn=attn
        self.embed_dim=embed_dim
        self.norm_layer=LayerNorm(embed_dim)
        self.dropout=nn.Dropout(dropout)
        self.sublayers=clone(SublayerConnection(embed_dim,dropout),2)

    def forward(self,x):
        x=self.sublayers[0](x,lambda x:self.attn(x,x,x))
        x=self.sublayers[1](x,lambda x:self.mlp(x))
        return x
