import os
import torch
from parameter.configs import load_config
from torchvision import transforms
from PIL import Image
import re
import random
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
import math
import numpy as np
import json
import cv2


config=load_config()
llm_tokenizer=AutoTokenizer.from_pretrained(config['data']['llm_model_path'],local_files_only=True)
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
data_path=config['data']['caption_path']
img_path=config['data']['image_path']
img_caption_dict={}
accum_steps=config['training']['accum_steps']
micro_batch=config['training']['micro_batch']

IMAGE_TOKEN = '<image>'
if IMAGE_TOKEN not in llm_tokenizer.get_vocab():
    llm_tokenizer.add_special_tokens({'additional_special_tokens': [IMAGE_TOKEN]})
IMAGE_TOKEN_INDEX = llm_tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
IGNORE_INDEX = -100


def select_best_resolution(orginal_size,possible_resolution):#主要作用就是选择一个合适的画布
    ow,oh=orginal_size
    best_fit,max_eff,min_waste=None,0,float('inf')
    for w,h in possible_resolution:
        scale=min(w/ow,h/oh)
        dw,dh=int(ow*scale),int(oh*scale)
        eff=min(dw*dh,ow*oh)#就是算一下有效面积
        waste=w*h-eff
        if eff>max_eff or(eff==max_eff and waste<min_waste):
            max_eff,min_waste,best_fit=eff,waste,(w,h)
    return best_fit

def resize_and_pad_image(img,target_resolution):#主要作用就是将图片放大或者缩小并填充到画布居中位置
    tw,th=target_resolution
    w,h=img.size
    scale=min(tw/w,th/h)
    nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
    img=img.resize((nw,nh),Image.BICUBIC)
    canvas=Image.new('RGB',(tw,th),(0,0,0))
    canvas.paste(img,((tw-nw)//2,(th-nh)//2))
    return canvas

def divide_to_patches(img,patch_size):
    w,h=img.size
    return [img.crop((j,i,min(j+patch_size,w),min(i+patch_size,h)))
            for i in range(0,h,patch_size) for j in range(0,w,patch_size)]

siglip_transform=transforms.Compose([
    transforms.Resize((config['training']['img_size'],config['training']['img_size'])),
    transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

class AnyResProcessor:
    def __call__(self,img):
        S,P=config['training']['img_size'],config['training']['patch_size']
        grid_pinpoints=[(i*S,j*S) for i in range(1,7) for j in range(1,7)]
        best=select_best_resolution(img.size,grid_pinpoints)
        padded=resize_and_pad_image(img,best)
        tiles=divide_to_patches(padded,S)
        if len(tiles)>config['training']['max_tiles']:
            tiles=tiles[:config['training']['max_tiles']]
        tiles=[img.resize((S,S),Image.BICUBIC)]+tiles
        n=len(tiles)
        T=config['training']['num_patches']**2
        s=config['training']['num_patches']
        if n*T>config['training']['image_token_budget']:
            s=max(1,int(math.sqrt(config['training']['image_token_budget']/n)))
        tensor=torch.stack([siglip_transform(t) for t in tiles],dim=0)
        meta={'scenario':'image','num_block':n,'pool_side':s,'token_per_block':[n*(s**2+1)]}
        return tensor,meta

class MultiImageProcessor:
    def __call__(self,imgs):
        S=config['training']['img_size']
        tiles,hw=[],[]
        for im in imgs[:config['training']['max_num_images']]:
            w,h=im.size
            scale=S/max(w,h)
            nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
            im=im.resize((nw,nh),Image.BICUBIC)
            canvas=Image.new('RGB',(S,S),(0,0,0))
            canvas.paste(im,(0,0))
            tiles.append(siglip_transform(canvas))
            pw=max(1,round(nw/S*config['training']['num_patches']))
            ph=max(1,round(nh/S*config['training']['num_patches']))
            hw.append((ph,pw))
        tensor=torch.stack(tiles,dim=0)
        meta={'scenario':'multi_image','grid_hw':hw,'token_per_block':[pw*ph+1 for pw,ph in hw]}
        return tensor,meta

class VideoProcessor:
    def __call__(self,frames):
        S=config['training']['img_size']
        idx=np.linspace(0,len(frames)-1,config['training']['video_frames']).astype(int)\
          if len(frames)>config['training']['video_frames'] else range(len(frames))
        fs=[frames[i] for i in idx]
        tensor=torch.stack([siglip_transform(f.resize((S,S),Image.BICUBIC)) for f in fs],dim=0)
        s=int(math.sqrt(config['training']['video_token_per_block']))
        meta = {'scenario': 'video', 'num_block': len(fs), 'pool_side': s,
                'token_per_block':[s * s+1]}
        return tensor,meta

anyres_proc, multi_proc, video_proc = AnyResProcessor(), MultiImageProcessor(), VideoProcessor()


def handle_img(img_name):
    full_path=os.path.join(img_path,img_name)
    img=Image.open(full_path).convert("RGB")
    pixel,meta=anyres_proc(img)
    return pixel,meta

def handle_caption(caption):
    s1=caption.strip()
    s2=re.sub(r'([，。？！])',r' \1',s1)
    s3=re.sub(r'[^，。！？\u4e00-\u9fa5]+',r' ',s2)
    s3=re.sub(r'\s+',r' ',s3).strip()
    return s3

PROMPTS = [
    f'{IMAGE_TOKEN}\n请简要描述这张图片。',
    f'{IMAGE_TOKEN}\n用一句话描述图片内容。',
    f'{IMAGE_TOKEN}\n这张图片里有什么?',
    f'{IMAGE_TOKEN}\n请描述这张图片的主要内容。',
]
def get_data():
    with open(data_path,'r',encoding='utf-8') as f:
        data=f.read().strip().split('\n')
    subset=config['training'].get('subset_size',-1)     
    if isinstance(subset,int) and subset>0:
        data=data[:subset]
    img_caption_dict={}
    for text in data:
        pair=text.split(maxsplit=1)
        if len(pair)<2:                                   
            continue
        img_name=pair[0].split('#')[0]
        if not os.path.exists(os.path.join(img_path,img_name)):
            continue
        # 只存字符串 caption，不存 meta
        img_caption_dict.setdefault(img_name,[]).append(handle_caption(pair[1]))#setdefault(key,default)如果key不存在，返回default，否则返回key对应的值
    return img_caption_dict

img_caption_dict=get_data()
    

random.seed(42)
all_img_names=list(img_caption_dict.keys())
random.shuffle(all_img_names)

n_train=int(0.7*len(img_caption_dict))
n_val=int(0.15*len(img_caption_dict))

train_img_names=all_img_names[:n_train]
val_img_names=all_img_names[n_train:n_train+n_val]
test_img_names=all_img_names[n_train+n_val:]


class DataSet(Dataset):
    def __init__(self,img_caption_dict,type_set=None,cache_size=128):
        super().__init__()
        self.img_caption_dict=img_caption_dict
        if type_set is not None:
            self.img_names=[x for x in img_caption_dict.keys() if x in type_set]
        else:
            self.img_names=list(img_caption_dict.keys())
        self.sample_len=len(self.img_names)
        self._cache={}                 # 进程内小缓存，避免同一张图反复解码
        self._cache_size=cache_size

    def __len__(self):
        return self.sample_len

    def _load(self,img_name):
        hit=self._cache.get(img_name)
        if hit is not None:
            return hit
        out=handle_img(img_name)
        if len(self._cache)>=self._cache_size:
            self._cache.pop(next(iter(self._cache)))   # FIFO 淘汰
        self._cache[img_name]=out
        return out

    def __getitem__(self,item):
        r_index=min(max(item,0),self.sample_len-1)
        img_name=self.img_names[r_index]
        img_tensor,meta=self._load(img_name)
        select_cap=random.choice(self.img_caption_dict[img_name])
        prompt=random.choice(PROMPTS)
        return img_tensor,meta,prompt,select_cap
    
class MultiImageJson(DataSet):
    def __init__(self):
        self.items=[json.loads(l) for l in open(config['MULTI_JSONL']['train'],encoding='utf-8')if l.strip()]
        self.sample_len=len(self.items)
    def __len__(self):
        return self.sample_len
    def __getitem__(self,i):
        r_index=min(self.sample_len-1,max(i,0))
        it=self.items[r_index]
        imgs=[Image.open(os.path.join(img_path,p)).convert("RGB") for p in it['images']]
        pixel,meta=multi_proc(imgs)
        return pixel,meta,it['question'],it['answer']

class VideoJson(DataSet):
    def __init__(self):
      self.items=[json.loads(l) for l in open(config['VIDEO_JSONL']['train'],encoding='utf-8')if l.strip()]
      self.sample_len=len(self.items)
    def __len__(self):
        return self.sample_len
    def __getitem__(self,i):
        r_index=min(self.sample_len-1,max(i,0))
        it=self.items[r_index]
        frames=[]
        cap=cv2.VideoCapture(it['video'])
        while True:
            ok,f=cap.read()
            if not ok:
                break
            f=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(f))
        cap.release()
        pixel,meta=video_proc(frames)
        return pixel,meta,it['question'],it['answer']

def build_ids(prompt,answer,token_per_block):#将问题和答案转换为token_ids，并且将占位符根据token_per_block进行填充，最后构造标签
    p_ids=llm_tokenizer(prompt,add_special_tokens=False)['input_ids']
    a_ids=llm_tokenizer(answer,add_special_tokens=False)['input_ids']+[llm_tokenizer.eos_token_id]
    input_ids,ci=[],0
    for t in p_ids:
        if t==IMAGE_TOKEN_INDEX:
            input_ids+=[IMAGE_TOKEN_INDEX]*token_per_block[ci]
            ci+=1
        else:
            input_ids.append(t)
    labels=[IGNORE_INDEX]*len(input_ids)+a_ids
    input_ids=input_ids+a_ids
    return input_ids,labels



def get_dataloader():
    data_set=DataSet(img_caption_dict)
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=True,collate_fn=collate_fn)
    return dataloader

def get_train_dataloader():
    data_set=DataSet(img_caption_dict,train_img_names)
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=True,collate_fn=collate_fn)
    return dataloader

def get_val_dataloader():
    data_set=DataSet(img_caption_dict,val_img_names)
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=False,collate_fn=collate_fn)
    return dataloader

def get_test_dataloader():
    data_set=DataSet(img_caption_dict,test_img_names)
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=False,collate_fn=collate_fn)
    return dataloader

def get_mutli_train_dataloader(dataset):
    data_set=dataset()
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=True,collate_fn=collate_fn)
    return dataloader

def get_mutli_val_dataloader(dataset):
    data_set=dataset()
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=False,collate_fn=collate_fn)
    return dataloader

def get_mutli_test_dataloader(dataset):
    data_set=dataset()
    dataloader=DataLoader(data_set,batch_size=micro_batch,shuffle=False,collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    img_batch=[item[0] for item in batch]
    meta_batch=[item[1] for item in batch]
    max_n=max(p.shape[0] for p in img_batch)
    C,H,W=img_batch[0].shape[1:]
    pad_pixels,vis_mask=[],[]
    for p in img_batch:#图片补齐
        n=p.shape[0]
        if n<max_n:
            p=torch.cat([p,torch.zeros(max_n-n,C,H,W)],dim=0)
            m=torch.cat([torch.ones(n,dtype=torch.long),torch.zeros(max_n-n,dtype=torch.long)])
        else:
            m=torch.ones(n,dtype=torch.long)
        pad_pixels.append(p)
        vis_mask.append(m)

    input_ids_list,labels_list=[],[]
    for _,meta,prompt,answer in batch:#文字的补齐，构造一个固定容量的容器去存储文字和标签
        ids,lab=build_ids(prompt,answer,meta['token_per_block'])
        input_ids_list.append(ids)
        labels_list.append(lab)
    L=max(len(x) for x in input_ids_list)#input_ids_list是双层列表，每个元素是一个列表，表示一个样本的token_ids[[1,5,6],[4,3,2,1,0,9]]
    input_ids=torch.full((len(batch),L),llm_tokenizer.pad_token_id,dtype=torch.long)
    labels=torch.full((len(batch),L),IGNORE_INDEX,dtype=torch.long)
    attn_mask=torch.zeros((len(batch),L),dtype=torch.long)
    for i,(ids,label) in enumerate(zip(input_ids_list,labels_list)):
        input_ids[i,:len(ids)]=torch.tensor(ids,dtype=torch.long)
        labels[i,:len(label)]=torch.tensor(label,dtype=torch.long)
        attn_mask[i,:len(ids)]=1
    return (torch.stack(pad_pixels,dim=0),torch.stack(vis_mask,dim=0),meta_batch,input_ids,labels,attn_mask)

