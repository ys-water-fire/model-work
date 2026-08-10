import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import re
from torchvision import transforms
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import jieba
import random
import math
import time
import copy
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score,accuracy_score




SOS_token=1
EOS_token=2
Encode=3
Decode=4
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
data_path='./Flickr8kCN/flickr8kcn/data/flickr8kzhc.caption.txt'
img_path='./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/'
img_caption_dict={}
caption_dict={"<pad>":0,"SOS_token":1,"EOS_token":2,"Encode":3,"Decode":4}
accum_steps=2
micro_batch=8


def handle_img(img_name):
    full_path=os.path.join(img_path,img_name)
    img=Image.open(full_path).convert("RGB")
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    img=transform(img)
    return img

def handle_caption(caption):
    s1=caption.strip()
    s2=re.sub(r'([，。？！])',r' \1',s1)
    s3=re.sub(r'[^，。！？\u4e00-\u9fa5]+',r' ',s2)
    s3=jieba.cut(s3)
    return list(s3)

def get_data():
    with open(data_path,'r',encoding='utf-8') as f:
        data=f.read().strip().split('\n')
    image_list=[]
    for pair in [text.split(maxsplit=1) for text in data]:
        img_name=pair[0].split('#')[0]
        image_list.append([img_name,handle_img(img_name),handle_caption(pair[1])])
    for bair in image_list:
        if bair[0] not in img_caption_dict:
            img_caption_dict[bair[0]]=[]
        img_caption_dict[bair[0]].append(bair[2])

        for caption in bair[2]:
            if caption not in caption_dict:
                caption_dict[caption]=len(caption_dict)
    
    cap_index2word={k:v for v,k in caption_dict.items()}
    name_to_tensor={item[0]:item[1] for item in image_list}
    return image_list,img_caption_dict,caption_dict,cap_index2word,name_to_tensor
image_list,img_caption_dict,caption_dict,cap_index2word,name_to_tensor=get_data()

random.seed(42)
all_img_names=list(img_caption_dict.keys())
random.shuffle(all_img_names)

n_train=int(0.7*len(all_img_names))
n_val=int(0.15*len(all_img_names))

train_img_names=all_img_names[:n_train]
val_img_names=all_img_names[n_train:n_train+n_val]
test_img_names=all_img_names[n_train+n_val:]

class DataSet(Dataset):
    def __init__(self,img_caption_dict,name_to_tensor,type_set=None):
        super().__init__()
        self.img_caption_dict=img_caption_dict
        self.name_to_tensor=name_to_tensor
        if type_set is not None:
            self.img_names=[x for x in img_caption_dict.keys() if x in type_set]
        else:
            self.img_names=list(img_caption_dict.keys())
        self.sample_len=len(self.img_names)

    def __len__(self):
        return self.sample_len
    
    def __getitem__(self,item):
        r_index=min(max(item,0),self.sample_len-1)
        img_name=self.img_names[r_index]
        img_tensor=self.name_to_tensor[img_name]
        cap_data=self.img_caption_dict[img_name]
        select_cap=random.choice(cap_data)

        caption2index=[caption_dict[item] for item in select_cap]
        caption2index.append(caption_dict["EOS_token"])
        caption_tensor=torch.tensor(caption2index,dtype=torch.long,device=device)

        return img_tensor,caption_tensor
    
def get_dataloader():
    data_set=DataSet(img_caption_dict,name_to_tensor)
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=True,collate_fn=collate_fn)
    return dataloader

def get_train_dataloader():
    data_set=DataSet(img_caption_dict,name_to_tensor,train_img_names)
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=True,collate_fn=collate_fn)
    return dataloader

def get_val_dataloader():
    data_set=DataSet(img_caption_dict,name_to_tensor,val_img_names)
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=False,collate_fn=collate_fn)
    return dataloader

def get_test_dataloader():
    data_set=DataSet(img_caption_dict,name_to_tensor,test_img_names)
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=False,collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    img_batch=[item[0] for item in batch]
    cap_batch=[item[1] for item in batch]
    img_tensor=torch.stack(img_batch,dim=0)
    img_tensor=img_tensor.to(device)

    max_len=max(len(word) for word in cap_batch)+1
    cap_pad_list=[]
    for word in cap_batch:
        pad_math=max_len-len(word)
        padded_tensor=torch.tensor([0]*pad_math,dtype=torch.long,device=device)
        padded_word=torch.cat((word,padded_tensor),dim=0)
        cap_pad_list.append(padded_word)
    cap_pad_tensor=torch.stack(cap_pad_list,dim=0)

    return img_tensor,cap_pad_tensor

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
    b=torch.ones((1,1,size,size),dtype=torch.long)
    mask=1-torch.triu(b,diagonal=1)
    return mask.to(device)

def Padding_Mask(cap_pad_tensor):
    cap_pad_mask=(cap_pad_tensor!=0).unsqueeze(1).unsqueeze(2).to(device,dtype=torch.long)
    return cap_pad_mask

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

class Embedding(nn.Module):
    def __init__(self,vocab_size,embed_dim):
        super().__init__()
        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.embedding=nn.Embedding(vocab_size,embed_dim)
    def forward(self,x_tensor):
        x1=self.embedding(x_tensor)
        return x1*math.sqrt(self.embed_dim)

class PositionDecoder(nn.Module):
   def __init__(self,embed_dim,max_len):
       super().__init__()
       self.embed_dim=embed_dim
       pe=torch.zeros(max_len,embed_dim)
       pos=torch.arange(0,max_len).unsqueeze(1)
       password=torch.exp(torch.arange(0,embed_dim//2)*(-math.log(10000.0)/embed_dim))
       position=pos*password
       pe[:,0::2]=torch.sin(position)
       pe[:,1::2]=torch.cos(position)
       pe=pe.unsqueeze(0)
       self.register_buffer('pe1',pe)
   def forward(self,x):
       position_x=x+self.pe1[:,:x.shape[1],:]
       return position_x

class DecoderLayer(nn.Module):
    def __init__(self,self_atten,ff,embed_dim,dropout):
        super().__init__()
        self.self_atten=self_atten
        self.ff=ff
        self.embed_dim=embed_dim
        self.dropout=nn.Dropout(dropout)
        self.sublayers=clone(SublayerConnection(embed_dim,dropout),2)
    def forward(self,x,padding_mask):
        x1=self.sublayers[0](x,lambda x:self.self_atten(x,x,x,padding_mask))
        x=self.sublayers[1](x1,lambda x:self.ff(x))
        return x

class Decoder(nn.Module):
    def __init__(self,decoder_layer,N,proj_dim=512):
        super().__init__()
        self.layers=clone(decoder_layer,N)
        self.cls_token=nn.Parameter(torch.randn(1,1,decoder_layer.embed_dim)*0.02)
        self.text_proj=nn.Linear(decoder_layer.embed_dim,proj_dim)
        self.norm_layer=LayerNorm(decoder_layer.embed_dim)
        self.embed=Embedding(len(caption_dict),decoder_layer.embed_dim)
        self.position=PositionDecoder(decoder_layer.embed_dim,max_len=60)
    def forward(self,padding_mask,text,eos_token_id=2):
        x=self.embed(text)
        B=x.shape[0]
        cls_tokens=self.cls_token.expand(B,-1,-1)
        x=torch.cat([cls_tokens,x],dim=1)
        x=self.position(x)
        for layer in self.layers:
            x=layer(x,padding_mask)
        x=self.norm_layer(x)
        eos_pos=(text==eos_token_id).long().argmax(dim=1)
        eos_feature=x[torch.arange(x.shape[0],device=text.device),eos_pos]
        proj_feature=self.text_proj(eos_feature)
        proj_feature=proj_feature/(torch.norm(proj_feature,dim=-1,keepdim=True)+1e-10)
        return proj_feature
        
class clip(nn.Module):
    def __init__(self,vit,decoder,init_temperature=0.17):
        super().__init__()
        self.vit=vit
        self.decoder=decoder
        self.logit_scale=nn.Parameter(torch.log(torch.tensor(1.0/init_temperature)))
    def forward(self,image,captions):
        image_features=self.vit(image)
        padding_mask=Padding_Mask(captions)
        cls_mask=torch.ones(captions.shape[0],1,1,1,device=device,dtype=padding_mask.dtype)
        padding_mask=torch.cat([cls_mask,padding_mask],dim=3)
        captions_features=self.decoder(padding_mask,captions)
        logit_scale=self.logit_scale.exp()
        logits_per_image=logit_scale*torch.matmul(image_features,captions_features.transpose(-1,-2))
        logits_per_caption=logit_scale*torch.matmul(captions_features,image_features.transpose(-1,-2))
        return logits_per_image,logits_per_caption
    
class ITMorLMEncoderLayer(nn.Module):
    def __init__(self,self_atten,ff,embed_dim,dropout):
        super().__init__()
        self.self_atten=self_atten
        self.ff=ff
        self.embed_dim=embed_dim
        self.dropout=nn.Dropout(dropout)
        self.sublayers=clone(SublayerConnection(embed_dim,dropout),3)
    def forward(self,x,mask_kind,vit_output):
        x1=self.sublayers[0](x,lambda x:self.self_atten(x,x,x,mask_kind))
        x2=self.sublayers[1](x1,lambda x:self.self_atten(x,vit_output,vit_output))
        x=self.sublayers[2](x2,lambda x:self.ff(x))
        return x
    
class ITMorLMEncoder(nn.Module):
    def __init__(self,encoder_layer,N):
        super().__init__()
        self.layers=clone(encoder_layer,N)
        self.norm_layer=LayerNorm(encoder_layer.embed_dim)
        self.embed_dim=Embedding(len(caption_dict),encoder_layer.embed_dim)
        self.position=PositionDecoder(encoder_layer.embed_dim,max_len=60)
    def forward(self,x,mask_kind,vit_output):
        x=self.embed_dim(x)
        x=self.position(x)
        for layer in self.layers:
            x=layer(x,mask_kind,vit_output)
        x=self.norm_layer(x)
        return x
    
class blip(nn.Module):
   def __init__(self,clip,encoder,embed_dim):
       super().__init__()
       self.clip=clip
       self.encoder=clone(encoder,2)
       self.itm_head=nn.Sequential(nn.Linear(embed_dim,256),nn.ReLU(),nn.Linear(256,1))
       self.lm_head=nn.Linear(embed_dim,len(caption_dict))
   def forward(self,image_tensor,cap_pad_tensor):
       encoder_output=self.clip.vit.forward_features(image_tensor)
       ITC_img2cap_output,ITC_cap2img_output=self.clip(image_tensor,cap_pad_tensor)
       B=cap_pad_tensor.shape[0]
       encode_token=torch.full(size=(B,1),fill_value=caption_dict['Encode'],device=device)
       itm_cap_pad_tensor=torch.cat([encode_token,cap_pad_tensor],dim=1)
       itm_cap_pad_mask=Padding_Mask(itm_cap_pad_tensor)
       ITM_work_output=self.encoder[0](itm_cap_pad_tensor,itm_cap_pad_mask,encoder_output)
       ITM_output=ITM_work_output[:,0,:]
       ITM_output=self.itm_head(ITM_output)
       decode_token=torch.full(size=(B,1),fill_value=caption_dict['Decode'],device=device)
       lm_cap_pad_tensor=torch.cat([decode_token,cap_pad_tensor],dim=1)
       lm_cap_pad_tensor=lm_cap_pad_tensor[:,:-1]
       seq_len=lm_cap_pad_tensor.shape[1]
       lm_cap_pad_mask=Padding_Mask(lm_cap_pad_tensor)
       lm_caption_combined_mask=Causal_Mask(seq_len)*lm_cap_pad_mask
       LM_output=self.encoder[1](lm_cap_pad_tensor,lm_caption_combined_mask,encoder_output)
       LM_output=self.lm_head(LM_output)
       return ITC_img2cap_output,ITC_cap2img_output,ITM_output,LM_output
   
def BILP():
    embed_dim=768
    dropout=0.1
    hidden_dim=3072
    patch_embed=PatchEmbed(224,16,3,embed_dim)
    ff=MLP(embed_dim,hidden_dim,dropout)
    self_atten=MultiHeadAttention(8,embed_dim,dropout)
    block=Block(ff,self_atten,dropout,embed_dim)
    vit1=VisionTransformer(embed_dim,patch_embed,block,dropout=dropout)
    self_atten1=copy.deepcopy(self_atten)
    ff1=copy.deepcopy(ff)
    decoder_layer=DecoderLayer(self_atten1,ff1,embed_dim,dropout)
    decoder=Decoder(decoder_layer,12)
    clip1=clip(vit1,decoder)
    encoder_layer=ITMorLMEncoderLayer(self_atten,ff,embed_dim,dropout)
    encoder=ITMorLMEncoder(encoder_layer,12)
    model=blip(clip1,encoder,embed_dim)
    return model

def compute_recall(i2t_logits,t2i_logits,topk=(1,2,3)):
    B=i2t_logits.shape[0]
    results={}
    for k in topk:
        _,topk_idx=i2t_logits.topk(k=k,dim=-1)
        correct=torch.arange(B,device=device).view(-1,1)
        match=(topk_idx==correct).any(dim=-1)
        results[f'R@{k}_I2T']=match.float().mean().item()*100

    for k in topk:
        _,topk_idx=t2i_logits.topk(k=k,dim=-1)
        correct=torch.arange(B,device=device).view(-1,1)
        match=(topk_idx==correct).any(dim=-1)
        results[f'R@{k}_T2I']=match.float().mean().item()*100

    _,sorted_idx=i2t_logits.sort(dim=-1,descending=True)
    correct_rank=(sorted_idx==correct).nonzero(as_tuple=True)[1]
    results['Median_Rank_I2T']=correct_rank.median().item()+1
    
    _,sorted_indices=i2t_logits.sort(dim=-1,descending=True)
    correct_rank=(sorted_indices==correct).nonzero(as_tuple=True)[1]
    results['Median_Rank_T2I']=correct_rank.median().item()+1
    return results
    
def ITC_recall():
    all_logits1=[]
    all_logits2=[]
    with torch.no_grad():
        for img_tensor,cap_pad_tensor in get_val_dataloader():
            padding_mask=Padding_Mask(cap_pad_tensor)
            cls_mask = torch.ones(cap_pad_tensor.shape[0], 1, 1, 1, device=device, dtype=padding_mask.dtype)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=3)
            img_feat=model.clip.vit.forward(img_tensor)
            cap_feat=model.clip.decoder.forward(padding_mask,cap_pad_tensor)
            all_logits1.append(img_feat.cpu())
            all_logits2.append(cap_feat.cpu())
        all_logits1=torch.cat(all_logits1,dim=0).to(device)
        all_logits2=torch.cat(all_logits2,dim=0).to(device)
        score1=torch.matmul(all_logits1,all_logits2.transpose(-2,-1))
        score2=torch.matmul(all_logits2,all_logits1.transpose(-2,-1))
        results1=compute_recall(score1,score2)
        print(f"[Test R@1 I2T: {results1['R@1_I2T']} | R@1 T2I: {results1['R@1_T2I']}")
        print(f"Test R@2 I2T: {results1["R@2_I2T"]} | R@2 T2I: {results1["R@2_T2I"]}")
        print(f"Test R@3 I2T: {results1["R@3_I2T"]} | R@3 T2I: {results1["R@3_T2I"]}")
        print(f" Test Median_Rank_I2T: {results1['Median_Rank_I2T']}|Median_Rank_T2I: {results1['Median_Rank_T2I']}")
        model.train()

def BLIP_ITM_evaluate_recall(model):
    model.eval()
    with torch.no_grad():
     for img_tensor,cap_pad_tensor in get_val_dataloader():
        B=img_tensor.shape[0]
        rand_idx=torch.randperm(B,device=device)
        cap_neg_tensor=cap_pad_tensor[rand_idx]
        img_all=torch.cat([img_tensor,img_tensor],dim=0)
        cap_all=torch.cat([cap_pad_tensor,cap_neg_tensor],dim=0)
        itm_labels=torch.cat([torch.ones(B,device=device),torch.zeros(B,device=device)],dim=0).float().unsqueeze(-1)
        _,_,ITM_output,_=model(img_all,cap_all)
        match_prob=torch.sigmoid(ITM_output)
        pred_label=(match_prob>0.5).long()
        print("匹配原始分数:",ITM_output)
        print("匹配概率:",match_prob)
        print("预测标签(1=匹配,0=不匹配):",pred_label)
    itm_labels=itm_labels.cpu().numpy()
    match_prob=match_prob.cpu().numpy()
    pred_label=pred_label.cpu().numpy()
    auc=roc_auc_score(itm_labels,match_prob)
    acc=accuracy_score(itm_labels,pred_label)
    print(f"auc:{auc},acc:{acc}")

def compute_clip_score_for_lm(model, img_tensor, generated_ids):
    model.eval()
    with torch.no_grad():
        # 1. 图像编码（用 CLIP 的 ViT）
        img_feat = model.clip.vit.forward(img_tensor)  # [1, 768]
        
        # 2. 文本编码（用 CLIP 的 Decoder）
        # 注意：需要构造 padding mask
        generated_ids=torch.tensor(generated_ids,device=device)
        gen_ids_tensor = generated_ids.unsqueeze(0)  # [1, seq_len]
        padding_mask = Padding_Mask(gen_ids_tensor)
        cls_mask = torch.ones(gen_ids_tensor.shape[0], 1, 1, 1, device=device, dtype=padding_mask.dtype)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=3)
        txt_feat = model.clip.decoder(padding_mask, gen_ids_tensor)  # [1, 768]
        
        # 3. L2 归一化（CLIP 标准做法）
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        
        # 4. 余弦相似度（归一化后点积就是余弦相似度）
        cosine_sim = (img_feat * txt_feat).sum(dim=-1)  # 标量，范围 [-1, 1]
        
        # 5. 应用 CLIPScore 公式：w * max(cosine, 0)，w=2.5
        clip_score = 2.5 * torch.max(cosine_sim, torch.zeros_like(cosine_sim))
        
        return clip_score.item()

def evaluate_lm_clipscore(model, dataloader, max_samples=100):
    """
    用 CLIPScore 评估 LM 生成质量
    """
    model.eval()
    clip_scores = []
    
    with torch.no_grad():
        for i, (img_tensor, cap_pad_tensor) in enumerate(dataloader):
            if i >= max_samples: break  # 限制样本数，加速
            
            single_img = img_tensor[0:1]  # 取第一张图
            
            # 1. LM 生成描述（调用你已有的生成函数）
            generated_ids = BLIP_LM_evaluate(model, single_img,return_id=True)
            # generated_ids 是 token id 列表，如 [3, 56, 78, 2]
            
            # 2. 计算 CLIPScore
            score = compute_clip_score_for_lm(model, single_img, generated_ids)
            clip_scores.append(score)
    
    avg_clip = np.mean(clip_scores) if clip_scores else 0.0
    print(f"[LM CLIPScore] Average: {avg_clip:.4f}")


def BILP_train(optimizer,criterion,criterion2):
    model.train()
    total_loss=0
    total_correct=0
    total_target=0
    dataloader=get_train_dataloader()
    pbar=tqdm(dataloader,desc='Training')
    for batch in pbar:
         img_tensor,cap_pad_tensor=batch
         B=cap_pad_tensor.shape[0]
         ict_labels=torch.arange(B,device=device)
         ITC_img2cap_output,ITC_cap2img_output,ITM_output,LM_output=model(img_tensor,cap_pad_tensor)
         loss1=criterion(ITC_img2cap_output,ict_labels)+criterion(ITC_cap2img_output,ict_labels)
         rand_idx=torch.randperm(B,device=device)
         cap_neg_tensor=cap_pad_tensor[rand_idx]
         img_all=torch.cat([img_tensor,img_tensor],dim=0)
         cap_all=torch.cat([cap_pad_tensor,cap_neg_tensor],dim=0)
         itm_labels=torch.cat([torch.ones(B,device=device),torch.zeros(B,device=device)],dim=0).float().unsqueeze(-1)
         _,_,ITM_output,_=model(img_all,cap_all)
         loss2=criterion2(ITM_output,itm_labels)
         lm_labels=cap_pad_tensor
         loss3=criterion(LM_output.transpose(1,2),lm_labels)
         loss=0.2*loss1+0.2*loss2+0.6*loss3
         loss=loss/accum_steps
         loss.backward()
         if (pbar.n+1)% accum_steps==0:
            optimizer.step()
            optimizer.zero_grad()
         total_loss+=loss.item()

         pred_token=LM_output.argmax(dim=-1)
         mask=(cap_pad_tensor!=0)
         correct=(pred_token[mask]==lm_labels[mask]).sum().item()
         target=mask.sum().item()
         total_correct+=correct
         total_target+=target
    avg_acc=total_correct/total_target if total_target>0 else 0.0
    avg_loss=total_loss/len(dataloader)
    return avg_loss,avg_acc

def write_loss():
    num_epochs=20
    plot_loss_list=[]
    optimizer=optim.Adam(model.parameters(),lr=4e-5,weight_decay=1e-4)
    scheduler=CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion=nn.CrossEntropyLoss(ignore_index=0)
    criterion2=nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        print_loss_total,plot_loss_total=0.0,0.0
        start_time=time.time()
        avg_loss,avg_acc=BILP_train(optimizer,criterion,criterion2)
        print_loss_total+=avg_loss
        plot_loss_total+=avg_loss
        if epoch%1==0:
            print_avg_loss=print_loss_total/1
            print_loss_total=0.0
            print(f'轮次{epoch+1}  损失{print_avg_loss:.6f}  准确率{avg_acc:.6f}  时间:%d'%(time.time()-start_time))
        if epoch%1==0:
            plot_avg_loss=plot_loss_total/1
            plot_loss_list.append(plot_avg_loss)
            plot_loss_total=0.0
        
        checkpoint={'epoch':epoch+1,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),'plot_loss_list':plot_loss_list}
        save_path=f'./opencv/blip_{epoch+1}.checkpoint.pth'
        try:
           torch.save(checkpoint,save_path)
           print(f'模型已保存到{save_path}')
        except Exception as e:
           print(f'模型保存失败:{e}')

        if epoch%1==0:
            model.eval()
            ITC_recall()
            BLIP_ITM_evaluate_recall(model)
            evaluate_lm_clipscore(model,get_val_dataloader())
            model.train()
        scheduler.step()

    plt.figure(0)
    plt.plot(plot_loss_list)
    plt.savefig('./opencv/blip_plot_loss.png')
    plt.show()

model=BILP().to(device)
checkpoint=torch.load(f'./opencv/blip_1.checkpoint.pth',map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])      

def test_ITC_recall():
    all_logits1=[]
    all_logits2=[]
    model.eval()
    with torch.no_grad():
        for img_tensor,cap_pad_tensor in get_test_dataloader():
            padding_mask=Padding_Mask(cap_pad_tensor)
            cls_mask = torch.ones(cap_pad_tensor.shape[0], 1, 1, 1, device=device, dtype=padding_mask.dtype)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=3)
            img_feat=model.clip.vit.forward(img_tensor)
            cap_feat=model.clip.decoder.forward(padding_mask,cap_pad_tensor)
            all_logits1.append(img_feat.cpu())
            all_logits2.append(cap_feat.cpu())
        all_logits1=torch.cat(all_logits1,dim=0).to(device)
        all_logits2=torch.cat(all_logits2,dim=0).to(device)
        score1=torch.matmul(all_logits1,all_logits2.transpose(-2,-1))
        score2=torch.matmul(all_logits2,all_logits1.transpose(-2,-1))
        results1=compute_recall(score1,score2)
        print(f"[Test R@1 I2T: {results1['R@1_I2T']} | R@1 T2I: {results1['R@1_T2I']}")
        print(f"Test R@5 I2T: {results1["R@5_I2T"]} | R@5 T2I: {results1["R@5_T2I"]}")
        print(f"Test R@10 I2T: {results1["R@10_I2T"]} | R@10 T2I: {results1["R@10_T2I"]}")
        print(f" Test Median_Rank_I2T: {results1['Median_Rank_I2T']}|Median_Rank_T2I: {results1['Median_Rank_T2I']}")


def test_BLIP_ITM_evaluate_recall(model):
    model.eval()
    with torch.no_grad():
     for img_tensor,cap_pad_tensor in get_test_dataloader():
        B=img_tensor.shape[0]
        rand_idx=torch.randperm(B,device=device)
        cap_neg_tensor=cap_pad_tensor[rand_idx]
        img_all=torch.cat([img_tensor,img_tensor],dim=0)
        cap_all=torch.cat([cap_pad_tensor,cap_neg_tensor],dim=0)
        itm_labels=torch.cat([torch.ones(B,device=device),torch.zeros(B,device=device)],dim=0).float().unsqueeze(-1)
        _,_,ITM_output,_=model(img_all,cap_all)
        match_prob=torch.sigmoid(ITM_output)
        pred_label=(match_prob>0.5).long()
    itm_labels=itm_labels.cpu().numpy()
    match_prob=match_prob.cpu().numpy()
    pred_label=pred_label.cpu().numpy()
    auc=roc_auc_score(itm_labels,match_prob)
    acc=accuracy_score(itm_labels,pred_label)
    print(f"auc:{auc},acc:{acc}")


def BLIP_ITC_evaluate(model,img_tensor,cap_pad_tensor):
    model.eval()
    with torch.no_grad():
        ITC_img2cap_output,ITC_cap2img_output=model.clip(img_tensor,cap_pad_tensor)
        top_score,top_idx=torch.topk(ITC_img2cap_output,k=3,dim=-1)
    return top_score,top_idx

def use_BLIP_ITC_evaluate(img_tensor):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    image_tensor=transform(img_tensor).unsqueeze(0).to(device)
    cap_pad_tensor=['一匹马在草地上跑','小狗趴在地上','一个人在海边','飞机在天空飞翔','猫咪坐在椅子上']
    cap_pad_list=[]
    token_list=[]
    for sentence in cap_pad_tensor:
       texts=handle_caption(sentence)
       texts.append('EOS_token')
       texts=[caption_dict[word] for word in texts]
       texts=torch.tensor(texts,dtype=torch.long,device=device)
       token_list.append(texts)
    max_len=max(len(word) for word in token_list)
    for token in token_list:
        pad_math=max_len-len(token)
        pad_tensor=torch.tensor([0]*pad_math,dtype=torch.long,device=device)
        cap_pad_tensor1=torch.cat((token,pad_tensor),dim=0)
        cap_pad_list.append(cap_pad_tensor1)
    cap_pad_tensor2=torch.stack(cap_pad_list,dim=0)
    top_score,top_idx=BLIP_ITC_evaluate(model,image_tensor,cap_pad_tensor2)
    idx_tensor=top_idx[0].squeeze()
    pred_sentences=[cap_pad_tensor[idx.item()] for idx in idx_tensor]
    pred_sentences=" ".join(pred_sentences)
    print(f'预测结果为:{pred_sentences}')
       

def BLIP_ITM_evaluate(model,img_tensor,cap_pad_tensor):
    model.eval()
    with torch.no_grad():
        _,_,ITM_output,_=model(img_tensor,cap_pad_tensor)
        match_prob=torch.sigmoid(ITM_output)
        pred_label=(match_prob>0.5).long()
        print("匹配原始分数:",ITM_output)
        print("匹配概率:",match_prob)
        print("预测标签(1=匹配,0=不匹配):",pred_label)

def use_BLIP_ITM_evaluate(img_tensor1,img_tensor2,img_tensor3):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    image_tensor1=transform(img_tensor1).to(device)
    image_tensor2=transform(img_tensor2).to(device)
    image_tensor3=transform(img_tensor3).to(device)
    image_tensor=torch.stack([image_tensor1,image_tensor2,image_tensor3],dim=0)
    cap_pad_tensor=['一匹马在草地上跑','小狗趴在地上','一个人在海边']
    token_list=[]
    cap_pad_list=[]
    for sentence in cap_pad_tensor:
        texts=handle_caption(sentence)
        texts.append('EOS_token')
        texts=[caption_dict[word] for word in texts]
        texts=torch.tensor(texts,dtype=torch.long,device=device)
        token_list.append(texts)
    max_len=max(len(word) for word in token_list)+1
    for token in token_list:
        pad_math=max_len-len(token)
        pad_tensor=torch.tensor([0]*pad_math,dtype=torch.long,device=device)
        cap_pad_tensor1=torch.cat((token,pad_tensor),dim=0)
        cap_pad_list.append(cap_pad_tensor1)
    cap_pad_tensor2=torch.stack(cap_pad_list,dim=0)
    BLIP_ITM_evaluate(model,image_tensor,cap_pad_tensor2)

def test_BLIP_LM_evaluate():
    model.eval()
    with torch.no_grad():
        evaluate_lm_clipscore(model,get_val_dataloader())
        

def BLIP_LM_evaluate(model,img_tensor,temperature=0.7,return_id=False):
    model.eval()
    used_words=set()
    result_word_list=[]
    generated_id_list=[]
    with torch.no_grad():
        decoder_input=torch.tensor([[Decode]]).to(device)
        for _ in range(60):
            _,_,_,LM_output=model(img_tensor,decoder_input)
            last_word_logits=LM_output[:,-1,:]
            for word_id in used_words:
                last_word_logits[0,word_id]=-10
            last_word_logits=last_word_logits/temperature
            top_k_logits,top_k_index=torch.topk(last_word_logits,k=5,dim=-1)
            prob=F.softmax(top_k_logits,dim=-1)
            select_item=torch.multinomial(prob,num_samples=1)
            next_word_id=top_k_index[0,select_item.item()]
            if next_word_id.item()==EOS_token:
                result_word_list.append('EOS')
                break
            else:
                generated_id_list.append(next_word_id.item())
                next_word_id=next_word_id.unsqueeze(0).unsqueeze(0)
                result_word_list.append(cap_index2word[next_word_id.item()])
                used_words.add(next_word_id.item())
                decoder_input=torch.cat((decoder_input,next_word_id),dim=1)
        if return_id:
            return generated_id_list
        else:
            return result_word_list

def use_BLIP_LM_evaluate(image):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    image_tensor=transform(image).unsqueeze(0).to(device)
    res=BLIP_LM_evaluate(model,image_tensor)
    res=''.join(res) if isinstance(res,list) else res
    return res







if __name__ == '__main__':
    img_tensor1=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/69189650_6687da7280.jpg')
    img_tensor2=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/10815824_2997e03d76.jpg')
    img_tensor3=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/2079152458_40712c3b40.jpg')

    #use_BLIP_ITC_evaluate(img_tensor1)
    #use_BLIP_ITM_evaluate(img_tensor1,img_tensor2,img_tensor3)
    res=use_BLIP_LM_evaluate(img_tensor2)
    print(res)
    #write_loss()
    #test_BLIP_LM_evaluate()




    
