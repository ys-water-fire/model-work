import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import re
import time 
import random 
import numpy as np
from typing import List,Tuple
from transformers import AutoTokenizer,AutoModelForCausalLM,CLIPVisionModel,GenerationConfig
from peft import LoraConfig,get_peft_model
from bert_score import score
import matplotlib.pyplot as plt
import torch.nn.functional as F

img_caption_dict={}
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
llm_tokenizer=AutoTokenizer.from_pretrained('./opencv/Qwen',local_files_only=True)

class Config:
    vision_dim=768
    hidden_dim=2048
    llm_dim=1536

    base_size=224
    aspect_threshold=1.5
    max_tiles=4

    lora_r=8
    lora_alpha=16
    lora_drop=0.05
    lora_targets=['q_proj','v_proj','k_proj','o_proj']

    num_epochs=30
    lr=2e-5
    weight_decay=1e-4
    accum_steps=2
    micro_batch=2
    max_text_len=64
    grad_clip=1.0

    data_path='./Flickr8kCN/flickr8kcn/data/flickr8kzhc.caption.txt'
    img_path='./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/'




class AnyResProcessor:
    def __init__(self,base_size=Config.base_size,aspect_threshold=Config.aspect_threshold):
        self.base_size = base_size
        self.aspect_threshold = aspect_threshold
        self.transform=transforms.Compose([transforms.Resize((base_size,base_size)),transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485,0.486,0.466],std=[0.229,0.224,0.225])])
    def __call__(self,img_name:str)->List[torch.Tensor]:
        full_path=os.path.join(Config.img_path,img_name)
        img = Image.open(full_path).convert("RGB")
        W,H=img.size
        aspect=max(W,H)/min(W,H)
        tiles=[]
        if aspect>self.aspect_threshold and W>H:
            tiles.append(img.crop((0,0,W//2,H)))
            tiles.append(img.crop((W//2,0,W,H)))
        elif aspect>self.aspect_threshold and W<H:
            tiles.append(img.crop((0,0,W,H//2)))
            tiles.append(img.crop((0,H//2,W,H)))
        else:
            tiles.append(img)
        tiles.append(img.resize((self.base_size,self.base_size)))
        if len(tiles)>Config.max_tiles:
            tiles=tiles[:Config.max_tiles]
        tiles_list=[self.transform(tile) for tile in tiles]
        return torch.stack(tiles_list,dim=0)

processor=AnyResProcessor()

def handle_caption(caption):
    s1=caption.strip()
    s2=re.sub(r'([，。！？])',' \1',s1)
    s3=re.sub(r'[^，。！？\u4e00-\u9fa5]+',r' ',s2)
    s3=re.sub(r'\s+',r' ',s3).strip()
    return s3

def get_data():
    with open(Config.data_path,'r',encoding='utf-8') as f:
        data=f.read().strip().split('\n')
    image_list=[]
    for pair in [text.split(maxsplit=1) for text in data]:
        img_name=pair[0].split('#')[0]
        image_list.append([img_name,processor(img_name),handle_caption(pair[1])])
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
    dataloader=DataLoader(data_set,batch_size=Config.micro_batch,shuffle=True,collate_fn=collate_fn)
    return dataloader

def get_train_dataloader():
    data_set=DataSet(img_caption_dict,name_to_tensor,train_img_names)
    dataloader=DataLoader(data_set,batch_size=Config.micro_batch,shuffle=True,collate_fn=collate_fn)
    return dataloader

def get_val_dataloader():
    data_set=DataSet(img_caption_dict,name_to_tensor,val_img_names)
    dataloader=DataLoader(data_set,batch_size=Config.micro_batch,shuffle=False,collate_fn=collate_fn)
    return dataloader

def get_test_dataloader():
    data_set=DataSet(img_caption_dict,name_to_tensor,test_img_names)
    dataloader=DataLoader(data_set,batch_size=Config.micro_batch,shuffle=False,collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    img_batch=[item[0] for item in batch]
    cap_batch=[item[1] for item in batch]
    max_tiles=Config.max_tiles
    img_pad_list=[]
    vis_mask=[]
    for tile in img_batch:
        num_tiles=tile.shape[0]
        pad_len=max_tiles-num_tiles
        if pad_len>0:
            pad_tensor=torch.stack([torch.zeros_like(tile[0]) for _ in range(pad_len)],dim=0)
            pad_tile=torch.cat([tile,pad_tensor],dim=0)
            img_pad_list.append(pad_tile)
            mask=(torch.cat([torch.ones(num_tiles,dtype=torch.long,device=device),torch.zeros(pad_len,dtype=torch.long,device=device)]).to(device))
            vis_mask.append(mask)
        else:
            pad_tile=tile
            img_pad_list.append(pad_tile)
            mask=torch.ones(num_tiles,dtype=torch.long,device=device)
            vis_mask.append(mask)
    img_pad_tensor=torch.stack(img_pad_list,dim=0).to(device)
    vis_mask=torch.stack(vis_mask,dim=0).to(device)
    token_out=llm_tokenizer(cap_batch,padding='longest',truncation=True,max_length=Config.max_text_len,return_tensors='pt')
    input_ids=token_out['input_ids'].to(device).long()
    attention_mask=token_out['attention_mask'].to(device)
    return img_pad_tensor,vis_mask,input_ids,attention_mask


def load_vision_encoder():
    print(f'加载视觉编码器')
    model_path='./opencv/openai/'
    model=CLIPVisionModel.from_pretrained(model_path,local_files_only=True)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad=False
    return model

def get_vision_tokens(vision_encoder,img_tensor,vis_mask):
    B,max_tiles,C,H,W=img_tensor.shape
    all_tiles=img_tensor.reshape(B*max_tiles,C,H,W)
    all_tokens=vision_encoder(all_tiles).last_hidden_state[:,1:,:]
    all_tokens=all_tokens.reshape(B,max_tiles,-1,Config.vision_dim)
    vis_tokens_list=[]
    for i in range(B):
        real_tokens=all_tokens[i][vis_mask[i]==1]
        vis_tokens_list.append(real_tokens.reshape(-1,Config.vision_dim))
    return vis_tokens_list

class MLPProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(Config.vision_dim,Config.hidden_dim)
        self.linear2=nn.Linear(Config.hidden_dim,Config.llm_dim)
        self.act=nn.GELU()
    def forward(self,x):
        x=self.linear1(x)
        x=self.act(x)
        x=self.linear2(x)
        return x
        
def load_llm_with_lora():
    print(f'加载LLM模型')
    model_path='./opencv/Qwen/'
    llm_model=AutoModelForCausalLM.from_pretrained(model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='cuda:0')
    lora_cfg = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_drop,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=Config.lora_targets
    )
    llm = get_peft_model(llm_model,lora_cfg)
    trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in llm.parameters())
    print(f'可训练参数{trainable},总参数{total}')
    return llm

class LLaVA_next(nn.Module):
    def __init__(self,vision_encoder,mlp_projector,llm):
        super().__init__()
        self.vision_encoder=vision_encoder
        self.llm=llm
        self.mlp_projector=mlp_projector
    def forward(self,img_tensor,vis_mask,input_ids,attention_mask,labels=None):
        B=input_ids.shape[0]
        vis_tokens_list=get_vision_tokens(self.vision_encoder,img_tensor,vis_mask)
        vis_embeds_list=[self.mlp_projector(tokens) for tokens in vis_tokens_list]

        text_embeds=self.llm.get_input_embeddings()(input_ids)
        inputs_embeds_list,full_attention_list,full_labels_list=[],[],[]

        for i in range(B):
            vis_emb=vis_embeds_list[i]
            vis_len=vis_emb.shape[0]
            combined_emb=torch.cat([vis_emb,text_embeds[i]],dim=0)
            vis_mask_seq=torch.ones(vis_len,dtype=torch.long,device=device)
            combined_mask=torch.cat([vis_mask_seq,attention_mask[i]],dim=0)
            inputs_embeds_list.append(combined_emb)
            full_attention_list.append(combined_mask)
            if labels is not None:
                vis_labels=torch.full((vis_len,),-100,dtype=torch.long,device=device)
                full_labels=torch.cat([vis_labels,labels[i]],dim=0)
                full_labels_list.append(full_labels)
        max_len=max([x.shape[0] for x in inputs_embeds_list])
        inputs_embeds=torch.zeros(B,max_len,Config.llm_dim,dtype=torch.bfloat16,device=device)
        full_mask=torch.zeros(B,max_len,dtype=torch.long,device=device)
        full_label=torch.full((B,max_len),-100,dtype=torch.long,device=device)if labels is not None else None

        for i in range(B):
            seq_len=inputs_embeds_list[i].shape[0]
            inputs_embeds[i,:seq_len]=inputs_embeds_list[i]
            full_mask[i,:seq_len]=full_attention_list[i]
            if labels is not None:
                full_label[i,:seq_len]=full_labels_list[i]
        outputs=self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_label,
            use_cache=False
        )
        return outputs

def LLAVA_next():
    vision_encoder=load_vision_encoder()
    llm=load_llm_with_lora()
    mlp_projector=MLPProjector().to(device).to(dtype=torch.bfloat16)
    model=LLaVA_next(
        vision_encoder=vision_encoder,
        mlp_projector=mlp_projector,
        llm=llm
    )
    return model

model=LLAVA_next()

def calc_bert_score():
    pred_texts=[]
    ref_texts=[]
    model.eval()
    with torch.no_grad():
        for img_tensor,vis_mask,input_ids,attention_mask in get_val_dataloader():
            output_ids=llava_evalate(model,img_tensor,vis_mask)
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


def train_LLAVA_next(optimizer):
    model.train()
    total_loss=0
    total_correct=0
    total_target=0
    dataloader=get_train_dataloader()
    pbar= tqdm(dataloader,desc='Training')
    for batch in pbar:
        img_tensor,vis_mask,input_ids,attention_mask=batch
        labels=input_ids.clone()
        B=input_ids.shape[0]
        with autocast(device_type='cuda',dtype=torch.bfloat16):
            vis_tokens=get_vision_tokens(model.vision_encoder,img_tensor,vis_mask)
            outputs=model(img_tensor,vis_mask,input_ids,attention_mask,labels=labels)
            loss=outputs.loss/Config.accum_steps
            loss.backward()
            if (pbar.n+1)% Config.accum_steps==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),Config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            total_loss+=loss.item()*Config.accum_steps

    avg_loss=total_loss/len(dataloader)
    return avg_loss
                
def write_llava_loss():
    plot_loss_list=[]
    trainable=[param for param in model.parameters() if param.requires_grad]
    optimizer=optim.AdamW(trainable,lr=2e-5,weight_decay=1e-4)
    scheduler=CosineAnnealingLR(optimizer, T_max=Config.num_epochs, eta_min=1e-5)
    for epoch in range(Config.num_epochs):
        print_loss_total,plot_loss_total=0.0,0.0
        start_time=time.time()
        avg_loss=train_LLAVA_next(optimizer)
        print_loss_total+=avg_loss
        plot_loss_total+=avg_loss
        if epoch%1==0:
            print_avg_loss=print_loss_total/1
            print_loss_total=0.0
            print(f'轮次{epoch+1}  损失{print_avg_loss:.6f} 时间:%d'%(time.time()-start_time))
        if epoch%1==0:
            plot_avg_loss=plot_loss_total/1
            plot_loss_list.append(plot_avg_loss)
            plot_loss_total=0.0
        
        checkpoint={'epoch':epoch+1,'model_state_dict':model.state_dict()}
        save_path=f'./opencv/llava1_next_{epoch+1}.checkpoint.pth'
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        try:
           torch.save(checkpoint,save_path)
           print(f'模型已保存到{save_path}')
        except Exception as e:
           print(f'模型保存失败:{e}')
        if (epoch+1)%30==0:
            model.eval()
            calc_bert_score()
            model.train()
        scheduler.step()

    plt.figure(0)
    plt.plot(plot_loss_list)
    plt.savefig('./opencv/llava_next_plot_loss.png')
    plt.show()

checkpoint=torch.load(f'./opencv/llava1_next_1.checkpoint.pth',map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'],strict=False)

def test_calc_bert_score():
    pred_texts=[]
    ref_texts=[]
    model.eval()
    with torch.no_grad():
        for img_tensor,vis_mask,input_ids,attention_mask in get_test_dataloader():
            output_ids=llava_evalate(model,img_tensor,vis_mask)
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


def llava_evalate(model,img_tensor,vis_mask,question=None,history=None):
    model.eval()
    img_tensor=img_tensor.to(dtype=torch.bfloat16)
    with torch.no_grad():
        patch_tokens_list=get_vision_tokens(model.vision_encoder,img_tensor,vis_mask)
        patch_tokens=[patch_tokens.to(dtype=torch.bfloat16) for patch_tokens in patch_tokens_list]
        val_tokens=torch.cat(patch_tokens,dim=0)
        vis_tokens = model.mlp_projector(val_tokens)
        seq_len=vis_tokens.shape[0]
        message=[]
        if question is not None:
          message=[{"role":"system","content":"你是一个多模态助手,只能根据图片内容回答问题，不知道的内容就说不知道，不能编造内容。"}]
          for user_msg, asst_msg in history:
                message.append({"role": "user", "content": user_msg})
                message.append({"role": "assistant", "content": asst_msg})
          message.append({"role": "user", "content": question})
          input=llm_tokenizer.apply_chat_template(message,add_generation_prompt=True,return_tensors='pt').to(device)
          question_embed=model.llm.get_input_embeddings()(input['input_ids']).to(dtype=torch.bfloat16)
          question_mask=input['attention_mask'].to(device)
          vis_mask1=torch.ones((1,seq_len),dtype=torch.long,device=device)
          val_tokens=vis_tokens.unsqueeze(0)
          vis_tokens=torch.cat([val_tokens,question_embed],dim=1)
          attention_mask=torch.cat([vis_mask1,question_mask],dim=1)
        else:
          attention_mask=torch.ones((1,seq_len),dtype=torch.long,device=device)
        gen_cfg=GenerationConfig(
                max_new_tokens=60,
                use_cache=False,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.05,
                eos_token_id=llm_tokenizer.eos_token_id,
                pad_token_id=llm_tokenizer.pad_token_id,
            )
        output_ids=model.llm.generate(inputs_embeds=vis_tokens,generation_config=gen_cfg,attention_mask=attention_mask)
        pred_sentences=llm_tokenizer.decode(output_ids[0],skip_special_tokens=True)
    return pred_sentences

def use_LLAVA_evaluate(img_name):
    img_tensor=processor(img_name).unsqueeze(0).to(device)
    num_tiles=img_tensor.shape[0]
    vis_mask=torch.ones((num_tiles),dtype=torch.long)
    history=[]
    print('图片已经加载，开始输入问题:(输入exit退出)')
    while True:
        user_q=input('请输入问题:').strip()
        if user_q.lower()=='exit':
            print('程序退出')
            break
        if not user_q:
            continue
        answer=llava_evalate(model,img_tensor,vis_mask,question=user_q,history=history)
        answer=''.join(answer) if isinstance(answer,list) else answer
        print(answer)
        history.append((user_q,answer))
        if len(history) > 5:
           history = history[-5:]




if __name__ == '__main__':
    #write_llava_loss()
    img_name1='69189650_6687da7280.jpg'
    img_name2='10815824_2997e03d76.jpg'
    use_LLAVA_evaluate(img_name1)
