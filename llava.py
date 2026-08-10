import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com/'
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
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM,AutoTokenizer,CLIPVisionModel,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
import bitsandbytes as bnb
from transformers import GenerationConfig
from bert_score import score

llm_model_name='Qwen/Qwen2-1.5B'
llm_tokenizer=AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.pad_token_id=llm_tokenizer.eos_token_id
llm_tokenizer.padding_side='right'
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
data_path='./Flickr8kCN/flickr8kcn/data/flickr8kzhc.caption.txt'
img_path='./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/'
img_caption_dict={}
accum_steps=1
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


class Config:
    vision_model='openai/clip-vit-base-patch32'
    llm_model='Qwen/Qwen2-1.5B'

    vision_dim=768
    hiddeb_dim=2048
    llm_dim=1536

    lora_r=8
    lora_alpha=16
    lora_drop=0.05
    lora_targets=['q_proj','k_proj','v_proj','o_proj']


class MLPProjector(nn.Module):
    def __init__(self,embed_dim=Config.vision_dim,hidden_dim=Config.hiddeb_dim,out_dim=Config.llm_dim):
        super().__init__()
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.linear1=nn.Linear(embed_dim,hidden_dim)
        self.act=nn.GELU()
        self.linear2=nn.Linear(hidden_dim,out_dim)
    def forward(self,x):
        x=self.linear1(x)
        x=self.act(x)
        x=self.linear2(x)
        return x
    
def load_vision_encoder():
    print(f'加载视觉编码器{Config.vision_model}')
    model=CLIPVisionModel.from_pretrained(Config.vision_model)
    model.eval()
    model=model.to(device)
    for param in model.parameters():
        param.requires_grad=False
    return model

def get_visual_tokens(vision_encoder,pixel_values):
    with torch.no_grad():
        output=vision_encoder(pixel_values)
        tokens=output.last_hidden_state[:,1:,:]
    return tokens

def load_llm_with_lora():
    print(f'加载语言模型{Config.llm_model}')
    llm_model = AutoModelForCausalLM.from_pretrained(
        Config.llm_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,   
        device_map='cuda:0',            
        local_files_only=True          
    )

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


class LLaVA(nn.Module):
    def __init__(self,vision_encoder,llm,mlp_projector):
        super().__init__()
        self.vision_encoder=vision_encoder
        self.llm=llm
        self.mlp_projector=mlp_projector
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
    # ViT冻结，无梯度前向
       patch_tokens = get_visual_tokens(self.vision_encoder, pixel_values)
       patch_tokens = patch_tokens.to(dtype=torch.bfloat16)
    
    # 视觉特征投影到LLM维度
       vis_tokens = self.mlp_projector(patch_tokens)
       text_embeds = self.llm.get_input_embeddings()(input_ids)
       inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)

       B, N = vis_tokens.shape[:2]
       vis_mask = torch.ones((B, N), dtype=torch.long, device=inputs_embeds.device)
       full_mask = torch.cat([vis_mask, attention_mask], dim=1)

       full_labels = None
       if labels is not None:
          vis_labels = torch.full((B, N), -100, dtype=torch.long, device=inputs_embeds.device)
          full_labels = torch.cat([vis_labels, labels], dim=1)

       output = self.llm(
          inputs_embeds=inputs_embeds,
          attention_mask=full_mask,
          labels=full_labels,
          use_cache=False
        )
       return output


def LLAVA():
    vision_encoder=load_vision_encoder()
    llm=load_llm_with_lora()
    mlp_projector=MLPProjector().to(device).to(dtype=torch.bfloat16)
    model=LLaVA(
        vision_encoder=vision_encoder,
        mlp_projector=mlp_projector,
        llm=llm
    )
    return model

model=LLAVA()

def calc_bert_score():
    pred_texts=[]
    ref_texts=[]
    model.eval()
    with torch.no_grad():
        for img_tensor,input_ids,attention_mask in get_val_dataloader():
            output_ids=llava_evalate(model,img_tensor)
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

def train_llava(optimizer):
    model.train()
    total_loss=0
    total_correct=0
    total_target=0
    dataloader=get_train_dataloader()
    pbar=tqdm(dataloader,desc='Training')
    for batch in pbar:
        img_tensor,input_ids,attention_mask=batch
        labels=input_ids.clone()
        B=img_tensor.shape[0]
        output=model(img_tensor,input_ids,attention_mask,labels=labels)
        loss=output.loss/accum_steps
        loss.backward()
        if (pbar.n+1)% accum_steps==0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss+=loss.item()*accum_steps
    
        pred_token=output.logits.argmax(dim=-1)
        prefix_labels=torch.full((B,49),-100,dtype=torch.long,device=device)
        full_labels=torch.cat([prefix_labels,labels],dim=1)
        labels=full_labels
        mask=(labels!=-100)
        correct=(pred_token[mask]==labels[mask]).sum().item()
        total_correct+=correct
        total_target+=mask.sum().item()
    avg_acc=total_correct/total_target
    avg_loss=total_loss/len(dataloader)
    return avg_loss,avg_acc
           

def write_llava_loss():
    num_epochs=10
    plot_loss_list=[]
    trainable=[param for param in model.parameters() if param.requires_grad]
    optimizer=optim.AdamW(trainable,lr=2e-5,weight_decay=1e-4)
    scheduler=CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    for epoch in range(num_epochs):
        print_loss_total,plot_loss_total=0.0,0.0
        start_time=time.time()
        avg_loss,avg_acc=train_llava(optimizer)
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
        save_path=f'./opencv/llava_{epoch+1}.checkpoint.pth'
        try:
           torch.save(checkpoint,save_path)
           print(f'模型已保存到{save_path}')
        except Exception as e:
           print(f'模型保存失败:{e}')
        if epoch%5==0:
            model.eval()
            calc_bert_score()
            model.train()
        scheduler.step()

    plt.figure(0)
    plt.plot(plot_loss_list)
    plt.savefig('./opencv/llava_plot_loss.png')
    plt.show()

checkpoint=torch.load(f'./opencv/llava_10.checkpoint.pth',map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

def test_calc_bert_score():
    pred_texts=[]
    ref_texts=[]
    model.eval()
    with torch.no_grad():
        for img_tensor,input_ids,attention_mask in get_test_dataloader():
            output_ids=llava_evalate(model,img_tensor)
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


def llava_evalate(model,img_tensor):
    model.eval()
    img_tensor=img_tensor.to(dtype=torch.bfloat16)
    with torch.no_grad():
        patch_tokens=get_visual_tokens(model.vision_encoder,img_tensor)
        patch_tokens=patch_tokens.to(dtype=torch.bfloat16)
        vis_tokens = model.mlp_projector(patch_tokens)
        b,seq_len,_=vis_tokens.shape
        gen_cfg=GenerationConfig(
                max_new_tokens=60,
                temperature=0.7,
                repetition_penalty=1.05,
                eos_token_id=llm_tokenizer.eos_token_id,
                pad_token_id=llm_tokenizer.pad_token_id,
            )
        output_ids=model.llm.generate(inputs_embeds=vis_tokens,generation_config=gen_cfg)
        pred_sentences=llm_tokenizer.decode(output_ids[0],skip_special_tokens=True)
    return pred_sentences

def use_LLAVA_evaluate(img_tensor):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    image_tensor=transform(img_tensor).unsqueeze(0).to(device)
    res=llava_evalate(model,image_tensor)
    res=''.join(res) if isinstance(res,list) else res
    return res


if __name__ == '__main__':
    img_tensor1=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/69189650_6687da7280.jpg')
    img_tensor2=Image.open('./Flickr8kCN/Flickr8k_Dataset/Flicker8k_Dataset/10815824_2997e03d76.jpg')
    res=use_LLAVA_evaluate(img_tensor2)
    print(res)
    #write_llava_loss()
    #test_calc_bert_score()
