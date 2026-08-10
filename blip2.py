import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
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
import random
import math
import time
import copy
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from sklearn.metrics import roc_auc_score,accuracy_score
from bert_score import score



llm_model_name='Qwen/Qwen2-1.5B'
llm_tokenizer=AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.pad_token_id=llm_tokenizer.eos_token_id
llm_tokenizer.padding_side='right'
llm_model=AutoModelForCausalLM.from_pretrained(llm_model_name)
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
data_path='./Flickr8kCN/flickr8kcn/data/flickr8kzhc.caption.txt'
img_path='./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/'
img_caption_dict={}
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
    s3=re.sub(r'\s+',r' ',s3).strip()
    return s3

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

    
    name_to_tensor={item[0]:item[1] for item in image_list}
    return image_list,img_caption_dict,name_to_tensor
image_list,img_caption_dict,name_to_tensor=get_data()

random.seed(42)
all_img_names=list(img_caption_dict.keys())
random.shuffle(all_img_names)

n_train=int(0.7*len(img_caption_dict))
n_val=int(0.15*len(img_caption_dict))

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

        return img_tensor,select_cap
    
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

    token_out=llm_tokenizer(cap_batch,padding='longest',truncation=True,max_length=64,return_tensors='pt')
    input_ids=token_out['input_ids'].to(device).long()
    attention_mask=token_out['attention_mask'].to(device)

    return img_tensor,input_ids,attention_mask

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
        
    def forward(self,x):
        x=self.patch_embed(x)
        B=x.shape[0]
        cls_tokens=self.cls_token.expand(B,-1,-1)
        x=torch.cat([cls_tokens,x],dim=1)
        x=x+self.pos_embed
        for block in self.blocks:
            x=block(x)
        return x

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

    
class text_transformer_layer(nn.Module):
    def __init__(self,self_atten,cross_atten,ff,embed_dim,dropout_p):
        super().__init__()
        self.self_atten=self_atten
        self.cross_atten=cross_atten
        self.ff=ff
        self.embed_dim=embed_dim
        self.dropout=dropout_p
        self.sublayers=clone(SublayerConnection(embed_dim,dropout_p),3)
    def forward(self,x,combined_mask,encoder_output):
        x=self.sublayers[0](x,lambda x:self.self_atten(x,x,x,combined_mask))
        if encoder_output is not None:
            query=x[:,:32,:]
            query=self.sublayers[1](query,lambda q:self.cross_atten(q,encoder_output,encoder_output))
            x=torch.cat([query,x[:,32:,:]],dim=1)
        x=self.sublayers[2](x,lambda x:self.ff(x))
        return x

class text_transformer(nn.Module):
    def __init__(self,text_transformer_layer,N):
        super().__init__()
        self.layers=clone(text_transformer_layer,N)
        self.norm=LayerNorm(text_transformer_layer.embed_dim)
    def forward(self,x,encoder_output,combined_mask=None):
        for layer in self.layers:
            x=layer(x,combined_mask,encoder_output)
        x=self.norm(x)
        return x

class Qformer(nn.Module):
    def __init__(self,text_transform):
        super().__init__()
        self.text_transform=text_transform
        self.query=nn.Parameter(torch.randn(1,32,1536))
        self.text_pos_embed=PositionDecoder(embed_dim=1536, max_len=64)
        llm_hidden=llm_model.config.hidden_size
        self.llm_proj=nn.Linear(768,llm_hidden)
        self.query_pos_embed=nn.Parameter(torch.randn(32,1536))

    def forward_text(self,text_emb,text_mask,encoder_output,task_type='itm'):
        batch_size=text_emb.shape[0]
        query_tokens=self.query.expand(batch_size,32,1536)
        query_tokens=query_tokens+self.query_pos_embed[None,:,:]
        text=self.text_pos_embed(text_emb)
        x=torch.cat([query_tokens,text],dim=1)
        seq_len=x.shape[1]
        query_mask=torch.ones((batch_size,32),dtype=torch.long,device=text_emb.device)
        full_mask=torch.cat([query_mask,text_mask],dim=-1)
        if task_type=='itc':
           attn_mask=torch.ones((batch_size,1,seq_len,seq_len),dtype=torch.long,device=text_emb.device)
           attn_mask[:,:,:32,32:]=0
           attn_mask[:,:,32:,:32]=0
           pad_4d=full_mask[:,None,None,:].expand(batch_size,-1,seq_len,-1)
           attn_mask=(pad_4d==1)&(attn_mask==1)
        elif task_type=='itm':
           attn_mask=full_mask[:,None,None,:].expand(batch_size,-1,seq_len,-1)
        else:
            raise ValueError(f"Unknown task type {task_type} not implemented")
        x=self.text_transform(x,encoder_output,attn_mask)
        img_rep=x[:,:32,:]
        text_rep=x[:,32:,:]
        return img_rep,text_rep
    def forward_img(self,encoder_output=None):
        batch_size=encoder_output.shape[0]
        query_tokens=self.query.expand(batch_size,32,1536)
        query_tokens=query_tokens+self.query_pos_embed[None,:,:]
        x=self.text_transform(query_tokens,encoder_output)
        return x


class BLIP2(nn.Module):
    def __init__(self,vit,qformer):
        super().__init__()
        self.vit=vit
        self.qformer=qformer
        self.llm_tokenizer=llm_tokenizer
        self.llm=llm_model
        self.llm.config.pad_token_id=llm_tokenizer.eos_token_id
        llm_hidden=llm_model.config.hidden_size
        self.itc_proj=nn.Linear(1536,256)
        self.itm_head=nn.Linear(1536,2)
        self.llm_proj=nn.Linear(1536,llm_hidden)
        for param in self.llm.parameters():
            param.requires_grad=False
    def loss_itc(self,img_rep,text_rep,text_mask):
        img_global=img_rep.mean(dim=1)
        eos_idx=text_mask.sum(dim=-1)-1
        txt_global=text_rep[torch.arange(text_rep.shape[0],device=text_rep.device),eos_idx]
        img_emb=F.normalize(self.itc_proj(img_global),dim=-1)
        txt_emb=F.normalize(self.itc_proj(txt_global),dim=-1)
        sim=torch.matmul(img_emb,txt_emb.transpose(-1,-2))/0.07
        labels=torch.arange(sim.shape[0],device=sim.device)
        loss_i2t=F.cross_entropy(sim,labels)
        loss_t2i=F.cross_entropy(sim.transpose(-1,-2),labels)
        return (loss_i2t+loss_t2i)/2,sim
    def loss_itm(self,text_ids,text_mask,encoder_output):
        B=text_ids.shape[0]
        neg_idx=torch.randperm(B,device=device)
        neg_text_ids=text_ids[neg_idx]
        neg_text_mask=text_mask[neg_idx]
        all_text_ids=torch.cat([text_ids,neg_text_ids],dim=0)
        all_text_mask=torch.cat([text_mask,neg_text_mask],dim=0)
        all_encoder_output=torch.cat([encoder_output,encoder_output],dim=0)
        all_text_emb=self.llm.get_input_embeddings()(all_text_ids)
        all_img_rep,_=self.qformer.forward_text(all_text_emb,all_text_mask,all_encoder_output,task_type='itm')
        img_global=all_img_rep.mean(dim=1)
        logits=self.itm_head(img_global)
        labels=torch.cat([torch.ones(B,device=device),torch.zeros(B,device=device)],dim=0).long()
        return F.cross_entropy(logits,labels),logits
    def loss_lm(self,query_features,input_ids,attention_mask):
        B=input_ids.shape[0]
        query_llm=self.llm_proj(query_features)
        text_emb=self.llm.get_input_embeddings()(input_ids)
        inputs_embeds=torch.cat([query_llm,text_emb],dim=1)
        query_mask=torch.ones((B,32),dtype=torch.long,device=text_emb.device)
        full_mask=torch.cat([query_mask,attention_mask],dim=-1)
        labels=input_ids.clone()
        prefix_labels=torch.full((B,32),-100,dtype=torch.long,device=input_ids.device)
        full_labels=torch.cat([prefix_labels,labels],dim=1)
        inputs_embeds=inputs_embeds.to(torch.bfloat16)
        outputs=self.llm(inputs_embeds=inputs_embeds,attention_mask=full_mask,labels=full_labels,return_dict=True)
        loss=outputs.loss
        return loss,outputs.logits
    def forward(self,image_tensor,input_ids,attention_mask):
        vit_features=self.vit(image_tensor)
        text_emb=self.llm.get_input_embeddings()(input_ids)
        img_rep,text_rep=self.qformer.forward_text(text_emb,attention_mask,vit_features,task_type='itc')
        loss_itc,sim=self.loss_itc(img_rep,text_rep,attention_mask)
        loss_itm,_=self.loss_itm(input_ids,attention_mask,vit_features)
        query_feat=self.qformer.forward_img(vit_features)
        loss_lm,logits=self.loss_lm(query_feat,input_ids,attention_mask)
        total_loss=loss_itc+loss_itm+loss_lm
        return total_loss,loss_itc,loss_itm,loss_lm,logits,sim
    
def BLIP2_model():
    embed_dim=1536
    dropout=0.1
    hidden_dim=3072
    patch_embed=PatchEmbed(224,16,3,embed_dim)
    ff=MLP(embed_dim,hidden_dim,dropout)
    self_atten=MultiHeadAttention(8,embed_dim,dropout)
    block=Block(ff,self_atten,dropout,embed_dim)
    vit1=VisionTransformer(embed_dim,patch_embed,block,dropout=dropout)
    self_atten1=copy.deepcopy(self_atten)
    ff1=copy.deepcopy(ff)
    text_transformer1_layer=text_transformer_layer(self_atten,self_atten1,ff1,embed_dim,dropout)
    text_transformer1=text_transformer(text_transformer1_layer,12)
    qformer=Qformer(text_transformer1)
    model=BLIP2(vit1,qformer)
    return model

model=BLIP2_model().to(device)

def compute_recall(i2t_logits,t2i_logits,topk=(1,2,3)):
    B=i2t_logits.shape[0]
    results={}
    for k in topk:
        _,topk_idx=i2t_logits.topk(k=k,dim=-1)
        correct=torch.arange(B,device=i2t_logits.device).view(-1,1)
        match=(topk_idx==correct).any(dim=-1)
        results[f'R@{k}_I2T']=match.float().mean().item()*100

    for k in topk:
        _,topk_idx=t2i_logits.topk(k=k,dim=-1)
        correct=torch.arange(B,device=t2i_logits.device).view(-1,1)
        match=(topk_idx==correct).any(dim=-1)
        results[f'R@{k}_T2I']=match.float().mean().item()*100

    _,sorted_idx=i2t_logits.sort(dim=-1,descending=True)
    correct_rank=(sorted_idx==correct).nonzero(as_tuple=True)[1]
    results['Median_Rank_I2T']=correct_rank.median().item()+1

    _,sorted_idx=t2i_logits.sort(dim=-1,descending=True)
    correct_rank=(sorted_idx==correct).nonzero(as_tuple=True)[1]
    results['Median_Rank_T2I']=correct_rank.median().item()+1
    return results

def itc_recall():
    all_logits1=[]
    all_logits2=[]
    with torch.no_grad():
        for batch in get_val_dataloader():
            img_tensor,input_ids,attention_mask=batch
            B=img_tensor.shape[0]
            vit_features=model.vit(img_tensor)
            text_emb=llm_model.get_input_embeddings()(input_ids)
            img_rep,text_rep=model.qformer.forward_text(text_emb,attention_mask,vit_features,task_type='itc')
            img_global=img_rep.mean(dim=1)
            eos_idx=attention_mask.sum(dim=-1)-1
            txt_global=text_rep[torch.arange(text_rep.shape[0],device=text_rep.device),eos_idx]
            all_logits1.append(img_global.cpu())
            all_logits2.append(txt_global.cpu())
        all_logits1=torch.cat(all_logits1,dim=0)
        all_logits2=torch.cat(all_logits2,dim=0)
        img_emb=F.normalize(all_logits1,dim=-1)
        txt_emb=F.normalize(all_logits2,dim=-1)
        score1=torch.matmul(img_emb,txt_emb.transpose(-1,-2))
        score2=torch.matmul(txt_emb,img_emb.transpose(-1,-2))
        results=compute_recall(score1,score2)
        print(f"[Test R@1 I2T: {results['R@1_I2T']} | R@1 T2I: {results['R@1_T2I']}")
        print(f"Test R@2 I2T: {results["R@2_I2T"]} | R@2 T2I: {results["R@2_T2I"]}")
        print(f"Test R@3 I2T: {results["R@3_I2T"]} | R@3 T2I: {results["R@3_T2I"]}")
        print(f" Test Median_Rank_I2T: {results['Median_Rank_I2T']}|Median_Rank_T2I: {results['Median_Rank_T2I']}")
        

def BLIP2_ITM_recall():
    model.eval() 
    with torch.no_grad():
        for img_tensor,input_ids,attention_mask in get_val_dataloader():
            B=img_tensor.shape[0]
            vit_features=model.vit(img_tensor)
            loss,logits=model.loss_itm(input_ids,attention_mask,vit_features)
            itm_labels=torch.cat([torch.ones(B,device=device),torch.zeros(B,device=device)],dim=0).float().unsqueeze(-1)
            logits=F.softmax(logits,dim=-1)
            match_prob=logits[:,1]
            pred_label=(match_prob>0.5).long()
        itm_labels=itm_labels.cpu().numpy()
        match_prob=match_prob.cpu().numpy()
        pred_label=pred_label.cpu().numpy()
        auc=roc_auc_score(itm_labels,match_prob)
        acc=accuracy_score(itm_labels,pred_label)
        print(f"ROC AUC: {auc:.6f}")
        print(f"Accuracy: {acc:.6f}")

def calc_bert_score():
    pred_texts=[]
    ref_texts=[]
    model.eval()
    with torch.no_grad():
        for img_tensor,input_ids,attention_mask in get_val_dataloader():
            output_ids=BLIP2_LM_evaluate(model,img_tensor)
            pred_texts.append(output_ids)
            one_img_refs=[]
            for token_seq in input_ids:
                ref_sent=llm_tokenizer.decode(token_seq)
                one_img_refs.append(ref_sent)
            ref_texts.append(one_img_refs)
            P, R, F1 = score(pred_texts, ref_texts, model_type='bert-base-chinese',verbose=True)
            mean_P=P.mean()
            mean_R=R.mean() 
            mean_F1=F1.mean()
            print(f'P:{mean_P:.6f}  R:{mean_R:.6f}  F1:{mean_F1:.6f}')

def BILP2_train(optimizer):
    model.train()
    total_loss=0
    total_correct=0
    total_target=0
    dataloader=get_train_dataloader()
    pbar=tqdm(dataloader,desc='Training')
    for batch in pbar:
        img_tensor,input_ids,attention_mask=batch
        B=img_tensor.shape[0]
        total_loss,loss_itc,loss_itm,loss_lm,logits,sim=model(img_tensor,input_ids,attention_mask)
        loss=total_loss/accum_steps
        loss.backward()
        if (pbar.n+1)% accum_steps==0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss+=loss.item()*accum_steps

        pred_token=logits.argmax(dim=-1)
        labels=input_ids.clone()
        prefix_labels=torch.full((B,32),-100,dtype=torch.long,device=input_ids.device)
        full_labels=torch.cat([prefix_labels,labels],dim=1)
        labels=full_labels
        mask=(labels!=-100)
        correct=(pred_token[mask]==labels[mask]).sum().item()
        total_correct+=correct
        total_target+=mask.sum().item()
    avg_acc=total_correct/total_target
    avg_loss=total_loss/len(dataloader)
    return avg_loss,avg_acc

def write_BILP2_loss():
    num_epochs=20
    plot_loss_list=[]
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=4e-5,weight_decay=1e-4)
    scheduler=CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    for epoch in range(num_epochs):
        print_loss_total,plot_loss_total=0.0,0.0
        start_time=time.time()
        avg_loss,avg_acc=BILP2_train(optimizer)
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
        save_path=f'./opencv/blip2_{epoch+1}.checkpoint.pth'
        try:
           torch.save(checkpoint,save_path)
           print(f'模型已保存到{save_path}')
        except Exception as e:
           print(f'模型保存失败:{e}')

        if epoch%1==0:
            model.eval()
            itc_recall()
            BLIP2_ITM_recall()
            calc_bert_score()
            model.train()

        scheduler.step()

    plt.figure(0)
    plt.plot(plot_loss_list)
    plt.savefig('./opencv/blip2_plot_loss.png')
    plt.show()

for param in model.vit.parameters():
    param.requires_grad=False
checkpoint=torch.load(f'./opencv/blip2_1.checkpoint.pth',map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


def test_itc():
    all_logits1=[]
    all_logits2=[]
    with torch.no_grad():
        for batch in get_test_dataloader():
            img_tensor,input_ids,attention_mask=batch
            B=img_tensor.shape[0]
            vit_features=model.vit(img_tensor)
            text_emb=llm_model.get_input_embeddings()(input_ids)
            img_rep,text_rep=model.qformer.forward_text(text_emb,attention_mask,vit_features,task_type='itc')
            img_global=img_rep.mean(dim=1)
            eos_idx=attention_mask.sum(dim=-1)-1
            txt_global=text_rep[torch.arange(text_rep.shape[0],device=text_rep.device),eos_idx]
            all_logits1.append(img_global.cpu())
            all_logits2.append(txt_global.cpu())
        all_logits1=torch.cat(all_logits1,dim=0)
        all_logits2=torch.cat(all_logits2,dim=0)
        img_emb=F.normalize(all_logits1,dim=-1)
        txt_emb=F.normalize(all_logits2,dim=-1)
        score1=torch.matmul(img_emb,txt_emb.transpose(-1,-2))
        score2=torch.matmul(txt_emb,img_emb.transpose(-1,-2))
        results=compute_recall(score1,score2)
        print(f"[Test R@1 I2T: {results['R@1_I2T']} | R@1 T2I: {results['R@1_T2I']}")
        print(f"Test R@2 I2T: {results["R@2_I2T"]} | R@2 T2I: {results["R@2_T2I"]}")
        print(f"Test R@3 I2T: {results["R@3_I2T"]} | R@3 T2I: {results["R@3_T2I"]}")
        print(f" Test Median_Rank_I2T: {results['Median_Rank_I2T']}|Median_Rank_T2I: {results['Median_Rank_T2I']}")

def BLIP2_ITC_evaluate(model,img_tensor,input_ids,attention_mask):
    model.eval()
    with torch.no_grad():
        vit_features=model.vit(img_tensor)
        img_query=model.qformer.forward_img(vit_features)
        img_global=img_query.mean(dim=1)
        img_global=F.normalize(img_global,dim=-1)
        text_emb=model.llm.get_input_embeddings()(input_ids)
        _,text_rep=model.qformer.forward_text(text_emb,attention_mask,encoder_output=None,task_type='itc')
        text_global=text_rep.mean(dim=1)
        text_global=F.normalize(text_global,dim=-1)
        score=torch.matmul(img_global,text_global.transpose(-2,-1))
        top_score,top_idx=torch.topk(score,k=3,dim=-1)
    return top_score,top_idx


def use_BLIP2_ITC_evaluate(img_tensor):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    image_tensor=transform(img_tensor).unsqueeze(0).to(device)
    cap_pad_tensor=['一匹马在草地上跑','小狗趴在地上','一个人在海边','飞机在天空飞翔','猫咪坐在椅子上']
    token_out=llm_tokenizer(cap_pad_tensor,return_tensors='pt',padding='longest', truncation=True,max_length=32)
    input_ids=token_out['input_ids'].to(device).long()
    attention_mask=token_out['attention_mask'].to(device)
    top_score,top_idx=BLIP2_ITC_evaluate(model,image_tensor,input_ids,attention_mask)
    top_idx=top_idx[0].squeeze()
    pred_sentences=[cap_pad_tensor[idx.item()] for idx in top_idx]
    pred_sentences=" ".join(pred_sentences)
    print(f'预测结果为:{pred_sentences}')

def test_BLIP2_ITM():
    model.eval() 
    with torch.no_grad():
        for img_tensor,input_ids,attention_mask in get_test_dataloader():
            B=img_tensor.shape[0]
            vit_features=model.vit(img_tensor)
            loss,logits=model.loss_itm(input_ids,attention_mask,vit_features)
            itm_labels=torch.cat([torch.ones(B,device=device),torch.zeros(B,device=device)],dim=0).float().unsqueeze(-1)
            logits=F.softmax(logits,dim=-1)
            match_prob=logits[:,1]
            pred_label=(match_prob>0.5).long()
            print("匹配原始分数:",logits)
            print("匹配概率:",match_prob)
            print("预测标签(1=匹配,0=不匹配):",pred_label)
        itm_labels=itm_labels.cpu().numpy()
        match_prob=match_prob.cpu().numpy()
        pred_label=pred_label.cpu().numpy()
        auc=roc_auc_score(itm_labels,match_prob)
        acc=accuracy_score(itm_labels,pred_label)
        print(f"ROC AUC: {auc:.6f}")
        print(f"Accuracy: {acc:.6f}")


def BLIP2_ITM_evaluate(model,img_tensor,input_ids,attention_mask):
    model.eval()
    with torch.no_grad():
        vit_features=model.vit(img_tensor)
        text_emb=model.llm.get_input_embeddings()(input_ids)
        img_rep,_=model.qformer.forward_text(text_emb,attention_mask,vit_features,task_type='itm')
        global_features=img_rep.mean(dim=1)
        logits=model.itm_head(global_features)
        print(f'原始二维logits(不匹配/匹配分数):\n{logits}')
        match_score=logits[:,1]
        print(f'图文原始匹配分数:{match_score}')
        match_prob=torch.softmax(logits,dim=-1)[:,1]
        print(f'图文匹配分数:{match_prob}')
        pred_label=(match_prob>0.5).long()
        print("预测标签(1=匹配,0=不匹配):",pred_label)


def use_BLIP2_ITM_evaluate(img_tensor1,img_tensor2,img_tensor3):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    image_tensor1=transform(img_tensor1).to(device)
    image_tensor2=transform(img_tensor2).to(device)
    image_tensor3=transform(img_tensor3).to(device)
    image_tensor=torch.stack([image_tensor1,image_tensor2,image_tensor3],dim=0)
    cap_pad_tensor=['一匹马在草地上跑','小狗趴在地上','一个人在海边']
    token_out=llm_tokenizer(cap_pad_tensor,return_tensors='pt',padding='longest', truncation=True,max_length=32)
    input_ids=token_out['input_ids'].to(device).long()
    attention_mask=token_out['attention_mask'].to(device)
    BLIP2_ITM_evaluate(model,image_tensor,input_ids,attention_mask)

def test_calc_bert_score():
    pred_texts=[]
    ref_texts=[]
    model.eval()
    with torch.no_grad():
        for img_tensor,input_ids,attention_mask in get_test_dataloader():
            output_ids=BLIP2_LM_evaluate(model,img_tensor)
            pred_texts.append(output_ids)
            one_img_refs=[]
            for token_seq in input_ids:
                ref_sent=llm_tokenizer.decode(token_seq)
                one_img_refs.append(ref_sent)
            ref_texts.append(one_img_refs)
            P, R, F1 = score(pred_texts, ref_texts, model_type='bert-base-chinese',verbose=True)
            mean_P=P.mean()
            mean_R=R.mean() 
            mean_F1=F1.mean()
            print(f'P:{mean_P:.6f}  R:{mean_R:.6f}  F1:{mean_F1:.6f}')

def BLIP2_LM_evaluate(model,img_tensor,return_id=False):
    model.eval()
    model.to(torch.bfloat16)
    img_tensor=img_tensor.to(torch.bfloat16)
    with torch.no_grad():
        vit_features=model.vit(img_tensor)
        img_query=model.qformer.forward_img(vit_features)
        llm_input_emb=model.llm_proj(img_query)
        b,seq_len,_=llm_input_emb.shape
        attention_mask=torch.ones((b,seq_len),device=device,dtype=torch.long)
        gen_cfg=GenerationConfig(
            max_new_tokens=60,
            temperature=0.7,
            repetition_penalty=1.05,
            eos_token_id=llm_tokenizer.eos_token_id,
            pad_token_id=llm_tokenizer.pad_token_id,
        )
        output_ids=model.llm.generate(inputs_embeds=llm_input_emb,attention_mask=attention_mask,generation_config=gen_cfg)
        pred_sentences=llm_tokenizer.decode(output_ids[0],skip_special_tokens=True)
        if return_id:
            return output_ids
        else:
            return pred_sentences

def use_BLIP2_LM_evaluate(img_tensor):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    image_tensor=transform(img_tensor).unsqueeze(0).to(device)
    res=BLIP2_LM_evaluate(model,image_tensor)
    res=''.join(res) if isinstance(res,list) else res
    return res



if __name__ == '__main__':
    img_tensor1=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/69189650_6687da7280.jpg')
    img_tensor2=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/10815824_2997e03d76.jpg')
    img_tensor3=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/2079152458_40712c3b40.jpg')
    #use_BLIP2_ITC_evaluate(img_tensor1)
    #use_BLIP2_ITM_evaluate(img_tensor1,img_tensor2,img_tensor3)
    res=use_BLIP2_LM_evaluate(img_tensor1)
    print(res)
    #write_BILP2_loss()
    
