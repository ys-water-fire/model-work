import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import re
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import jieba
import random
import copy
import math
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

SOS_token=1
EOS_token=2
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
data_path='./Flickr8kCN/flickr8kcn/data/flickr8kzhc.caption.txt'
img_path='./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/'
img_caption_dict={}
caption_dict={"<pad>":0,"SOS_token":1,"EOS_token":2}


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
    def __init__(self,img_caption_dict,name_to_tensor,img_set=None):
        super().__init__()
        self.img_caption_dict=img_caption_dict
        self.name_to_tensor=name_to_tensor
        if img_set is not None:
            self.img_names=[x for x in img_caption_dict.keys() if x in img_set]
        else:
            self.img_names=list(self.img_caption_dict.keys())
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
        caption2index=torch.tensor(caption2index,dtype=torch.long,device=device)

        return img_tensor,caption2index
    
def get_dataloader():
    data_set=DataSet(img_caption_dict,name_to_tensor)
    dataloader=DataLoader(data_set,batch_size=9,shuffle=True,collate_fn=collate_fn)
    return dataloader

def get_train_loader():
    dataset = DataSet(img_caption_dict, name_to_tensor, train_img_names)
    return DataLoader(dataset, batch_size=9, shuffle=True, collate_fn=collate_fn)

def get_val_loader():
    dataset = DataSet(img_caption_dict, name_to_tensor, val_img_names)
    return DataLoader(dataset, batch_size=9, shuffle=False, collate_fn=collate_fn)

def get_test_loader():
    dataset = DataSet(img_caption_dict, name_to_tensor, test_img_names)
    return DataLoader(dataset, batch_size=9, shuffle=False, collate_fn=collate_fn)

def collate_fn(batch):
    img_batch=[item[0] for item in batch]
    cap_batch=[item[1] for item in batch]
    img_tensor=torch.stack(img_batch,dim=0)
    img_tensor=img_tensor.to(device)

    max_len=max(len(word) for word in cap_batch)
    cap_pad_list=[]
    for word in cap_batch:
        pad_math=max_len-len(word)
        padded_tensor=torch.tensor([0]*pad_math,dtype=torch.long,device=device)
        padded_word=torch.cat((word,padded_tensor),dim=0)
        cap_pad_list.append(padded_word)
    cap_pad_tensor=torch.stack(cap_pad_list,dim=0)
    cap_pad_mask=(cap_pad_tensor!=0).unsqueeze(-1).to(device,dtype=torch.long)
    return img_tensor,cap_pad_tensor,cap_pad_mask

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
        cls_features=x[:,0,:]
        return cls_features
    def forward(self,x):
        x=self.forward_features(x)
        proj_features=self.image_proj(x)
        proj_features=proj_features/torch.norm(proj_features,dim=-1,keepdim=True)
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
        self.text_proj=nn.Linear(decoder_layer.embed_dim,proj_dim)
        self.norm_layer=LayerNorm(decoder_layer.embed_dim)
        self.embed=Embedding(len(caption_dict),decoder_layer.embed_dim)
        self.position=PositionDecoder(decoder_layer.embed_dim,max_len=60)
    def forward(self,padding_mask,text,eos_token_id=2):
        x=self.embed(text)
        x=self.position(x)
        for layer in self.layers:
            x=layer(x,padding_mask)
        x=self.norm_layer(x)
        eos_pos=(text==eos_token_id).long().argmax(dim=1)
        eos_feature=x[torch.arange(x.shape[0],device=text.device),eos_pos]
        proj_feature=self.text_proj(eos_feature)
        proj_feature=proj_feature/torch.norm(proj_feature,dim=-1,keepdim=True)
        return proj_feature
        
class clip(nn.Module):
    def __init__(self,vit,decoder,init_temperature=0.17):
        super().__init__()
        self.vit=vit
        self.decoder=decoder
        self.logit_scale=nn.Parameter(torch.log(torch.tensor(1.0/init_temperature)))
    def forward(self,image,captions,train_pad_mask):
        image_features=self.vit(image)
        padding_mask=train_pad_mask
        captions_features=self.decoder(padding_mask,captions)
        logit_scale=self.logit_scale.exp()
        logits_per_image=logit_scale*torch.matmul(image_features,captions_features.transpose(-1,-2))
        logits_per_caption=logit_scale*torch.matmul(captions_features,image_features.transpose(-1,-2))
        return logits_per_image,logits_per_caption

def CLIP():
    embed_dim=768
    dropout=0.1
    hidden_dim=3072
    vocab_size=len(caption_dict)
    patch_embed=PatchEmbed(224,16,3,embed_dim)
    mlp=MLP(embed_dim,hidden_dim,dropout)
    atten=MultiHeadAttention(8,embed_dim,dropout)
    block=Block(mlp,atten,dropout,embed_dim)
    vit=VisionTransformer(embed_dim,patch_embed,block,dropout)
    self_atten=copy.deepcopy(atten)
    decoder_layer=DecoderLayer(self_atten=self_atten,ff=mlp,embed_dim=embed_dim,dropout=dropout)
    text_encoder=Decoder(decoder_layer,12)
    model=clip(vit,text_encoder)
    return model

def compute_recall(i2t_logits,t2i_logits,topk=(1,5,10)):
    B=i2t_logits.shape[0]
    results={}
    for k in topk:
        _,top_idx=i2t_logits.topk(k=k,dim=-1)
        correct=torch.arange(B,device=device).view(-1,1)
        match=(top_idx==correct).any(dim=-1)
        results[f'R@{k}_I2T']=match.float().mean().item()*100

    for k in topk:
        _,top_idx=t2i_logits.topk(k=k,dim=-1)
        correct=torch.arange(B,device=device).view(-1,1)
        match=(top_idx==correct).any(dim=-1)
        results[f'R@{k}_T2I']=match.float().mean().item()*100

    _,sorted_indices=i2t_logits.sort(dim=-1,descending=True)
    correct_rank=(sorted_indices==correct).nonzero(as_tuple=True)[1]
    results['Median_Rank_I2T']=correct_rank.median().item()+1

    _,sorted_indices=i2t_logits.sort(dim=-1,descending=True)
    correct_rank=(sorted_indices==correct).nonzero(as_tuple=True)[1]
    results['Median_Rank_T2I']=correct_rank.median().item()+1


    return results

def train_CLIP(model,dataloader,criterion,optimizer):
    total_loss=0.0
    model.train()
    pbar=tqdm(dataloader,desc='Training')
    for batch in pbar:
        img_tensor,cap_pad_tensor,cap_pad_mask=batch
        B=cap_pad_tensor.shape[0]
        optimizer.zero_grad()
        logits_per_image,logits_per_caption=model(img_tensor,cap_pad_tensor,cap_pad_mask)
        labels=torch.arange(B,device=device)
        loss1=criterion(logits_per_image,labels)
        loss2=criterion(logits_per_caption,labels)
        loss=loss1+loss2/2.0
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(dataloader)
    return avg_loss
        
def write_loss():
    plot_loss_list=[]
    nums_epochs=10
    model=CLIP().to(device)
    optimizer=optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    criterion=nn.CrossEntropyLoss()
    scheduler=CosineAnnealingLR(optimizer,T_max=nums_epochs,eta_min=1e-6)
    dataloader=get_train_loader()
    for epoch in range(nums_epochs):
        print_total_loss,plot_total_loss=0.0,0.0
        start_time=time.time()
        avg_loss=train_CLIP(model,dataloader,criterion,optimizer)
        print_total_loss+=avg_loss
        plot_total_loss+=avg_loss
        if epoch%1==0:
            avg_loss=print_total_loss/1.0
            print_total_loss=0.0
            print(f'轮次{epoch+1}  损失{avg_loss:.6f}  时间:%d'%(time.time()-start_time))
        if epoch%1==0:
            plot_avg_loss=plot_total_loss/1.0
            plot_loss_list.append(plot_avg_loss)
            plot_total_loss=0.0
        checkpoint={'epoch':epoch+1,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),'plot_loss_list':plot_loss_list}
        save_path=f'./opencv/clip_{epoch+1}.checkpoint.pth'
        try:
           torch.save(checkpoint,save_path)
           print(f'模型已保存到{save_path}')
        except Exception as e:
           print(f'模型保存失败:{e}')
        if epoch % 1 == 0:
            model.eval()
            all_logits=[]
            all_captions=[]
            with torch.no_grad():
                test_loader = get_val_loader()
                for img_tensor, cap_pad_tensor, cap_pad_mask in test_loader:
                    img_feat=model.vit(img_tensor)
                    cap_feat=model.decoder(cap_pad_mask,cap_pad_tensor)
                    all_logits.append(img_feat.cpu())
                    all_captions.append(cap_feat.cpu())
                sim_matrix=torch.cat(all_logits,dim=0).to(device)
                sim_matrix2=torch.cat(all_captions,dim=0).to(device)
                score1=torch.matmul(sim_matrix,sim_matrix2.t())
                score2=torch.matmul(sim_matrix2,sim_matrix.t())
                metrics = compute_recall(score1,score2)
                print(f"[Epoch {epoch+1}] Test R@1 I2T: {metrics['R@1_I2T']} | R@1 T2I: {metrics['R@1_T2I']}")
                print(f"Test R@5 I2T: {metrics["R@5_I2T"]} | R@5 T2I: {metrics["R@5_T2I"]}")
                print(f"Test R@10 I2T: {metrics["R@10_I2T"]} | R@10 T2I: {metrics["R@10_T2I"]}")
                print(f" Test Median_Rank_I2T: {metrics['Median_Rank_I2T']}|Median_Rank_T2I: {metrics['Median_Rank_T2I']}")
            model.train()
        scheduler.step()

    plt.figure(0)
    plt.plot(plot_loss_list)
    plt.savefig('./opencv/loss.png')
    plt.show()

def CLIP_evaluate(model,img_tensor,cap_pad_tensor):
    model.eval()
    all_logits=[]
    with torch.no_grad():
        padding_mask=(cap_pad_tensor!=0).unsqueeze(-1).to(device,dtype=torch.long)
        logits_per_image,logits_per_caption=model(img_tensor,cap_pad_tensor,padding_mask)
        top_score,top_idx=torch.topk(logits_per_image,k=3,dim=-1)
    return top_score,top_idx

model=CLIP().to(device)
checkpoint=torch.load(f'./opencv/clip_10.checkpoint.pth',map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


def test_CLIP():
    model.eval()
    with torch.no_grad():
        all_logits=[]
        all_captions=[]
        for img_tensor,cap_pad_tensor,cap_pad_mask in get_test_loader():
            img_feat=model.vit(img_tensor)
            cap_feat=model.decoder(cap_pad_mask,cap_pad_tensor)
            all_logits.append(img_feat.cpu())
            all_captions.append(cap_feat.cpu())
        sim_matrix=torch.cat(all_logits,dim=0).to(device)
        sim_matrix2=torch.cat(all_captions,dim=0).to(device)
        score1=torch.matmul(sim_matrix,sim_matrix2.t())
        score2=torch.matmul(sim_matrix2,sim_matrix.t())
        metrics = compute_recall(score1,score2)
        print(f"Test R@1 I2T: {metrics['R@1_I2T']} | R@1 T2I: {metrics['R@1_T2I']}")
        print(f"Test R@5 I2T: {metrics["R@5_I2T"]} | R@5 T2I: {metrics["R@5_T2I"]}")
        print(f"Test R@10 I2T: {metrics["R@10_I2T"]} | R@10 T2I: {metrics["R@10_T2I"]}")
        print(f" Test Median_Rank_I2T: {metrics['Median_Rank_I2T']}|Median_Rank_T2I: {metrics['Median_Rank_T2I']}")







def use_CLIP_evaluate(img_tensor,img_tensor2,img_tensor3):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    image_tensor=transform(img_tensor).to(device)
    image_tensor2=transform(img_tensor2).to(device)
    image_tensor3=transform(img_tensor3).to(device)
    image_tensor=torch.stack((image_tensor,image_tensor2,image_tensor3),dim=0)
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
    top_score,top_idx=CLIP_evaluate(model,image_tensor,cap_pad_tensor2)
    idx_tensor=top_idx.squeeze()
    for each_idx in idx_tensor:
      pred_sentences=[cap_pad_tensor[idx.item()] for idx in each_idx]
      pred_sentences=" ".join(pred_sentences)
      print(f'预测结果为:{pred_sentences}')

       



    
if __name__ == '__main__':
    img_tensor=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/69189650_6687da7280.jpg')
    img_tensor2=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/10815824_2997e03d76.jpg')
    img_tensor3=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/2079152458_40712c3b40.jpg')
    #use_CLIP_evaluate(img_tensor,img_tensor2,img_tensor3)
    test_CLIP()
