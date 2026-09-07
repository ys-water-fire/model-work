import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, re, math, time, random, gc, json
import torch
import torch.nn as nn
from parameter.configs import load_config
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer,AutoModelForCausalLM,SiglipVisionModel,SiglipImageProcessor,GenerationConfig,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from models.common import load_data
from torch.amp import autocast


config=load_config()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
llm_tokenizer=AutoTokenizer.from_pretrained(config['data']['llm_model_path'],local_files_only=True)
if load_data.IMAGE_TOKEN not in llm_tokenizer.get_vocab():
    llm_tokenizer.add_special_tokens({'additional_special_tokens': [load_data.IMAGE_TOKEN]})


def load_vision_encoder():
    print('加载SigLip-SO400M视觉编码器')
    model=SiglipVisionModel.from_pretrained(config['data']['vision_model_path'],local_files_only=True)
    model=model.to(device)
    for param in model.parameters():
        param.requires_grad=False
    model.eval()
    return model

class MlpProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(config['model']['vision_embed_dim'],config['model']['hidden_dim'])
        self.linear2=nn.Linear(config['model']['hidden_dim'],config['model']['caption_embed_dim'])
        self.act=nn.GELU()
    def forward(self,x):
        x=self.linear1(x)
        x=self.act(x)
        x=self.linear2(x)
        return x

def load_llm_with_lora():
    print(f'加载Qwen+lora模型')
    llm_model=AutoModelForCausalLM.from_pretrained(config['data']['llm_model_path'],
            local_files_only=True,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='cuda:0')
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    lora_cfg = LoraConfig(
        r=config['lora']['lora_r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=config['lora']['lora_targets']
    )
    llm = get_peft_model(llm_model,lora_cfg)
    trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in llm.parameters())
    print(f'可训练参数{trainable},总参数{total}')
    for n, p in llm.named_parameters():     
        if p.requires_grad:
            p.data = p.data.to(torch.float32)#p.data->拿到权重数字本身，不带梯度追踪
    return llm

def bilinear_pool(tokens,side):#token超过预算的时候，经过AnyResProcessor算出的side,将vision_model处理好的token拼接回一张图片,然后进行bilinear池化，使得token的维度为side*side*llm_dim
    if side==config['training']['num_patches']:
        return tokens
    N,D=tokens.shape
    g=int(math.sqrt(N))
    x=tokens.view(1,g,g,D).permute(0,3,1,2)
    x=F.interpolate(x,(side,side),mode='bilinear',align_corners=False)
    return x.permute(0,2,3,1).reshape(side*side,D)


class LLaVAOneVision(nn.Module):
    def __init__(self,vision_encoder,mlp,llm):
        super().__init__()
        self.vision_encoder=vision_encoder
        self.mlp=mlp
        self.llm=llm
        self.image_newline=nn.Parameter(torch.randn(config['model']['caption_embed_dim'])*0.02)
        base=self.llm.get_base_model()
        base.config.use_cache=False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.llm.gradient_checkpointing_enable()

    @torch.no_grad()
    def encode_vision(self,pixel):
        out=self.vision_encoder(pixel,output_hidden_states=True)#开output_hidden_states=True,是为了不拿到归一化后的结果，因为做的不是图文对比，所以不需要归一化，归一化后效果反倒差
        return out.hidden_states[-1]

    def get_vision_feature(self,pixel,meta):
        B, n, C, H, W = pixel.shape
        flat = pixel.reshape(B * n, C, H, W)    
        hs=self.encode_vision(flat).to(torch.bfloat16)
        hs=hs.reshape(B, n, hs.shape[1], hs.shape[2])
        B=hs.shape[0]
        feats=[]
        for i in range(B):
            m=meta[i]
            block=[]
            if m['scenario']=='multi_image':#多图块先拼成一张完整图片，取出有效图片面积之后，再重新切分成小块(14*14),有27*27个小块
                for j,(ph,pw) in enumerate(m['grid_hw']):
                    t=hs[i,j].view(config['training']['num_patches'],config['training']['num_patches'],config['model']['vision_embed_dim'])
                    block.append(t[:ph,:pw,:].reshape(-1,config['model']['vision_embed_dim']))
            else:
                side=m['pool_side']
                for j in range(m['num_block']):#因为是单图，每个图片切出的tile数量不一样,算出来要池化的side也不一样，所以得一个个遍历出来去池化，不能一个batch直接送进去
                    block.append(bilinear_pool(hs[i,j],side))#[tile,tile2,tile3,...]
            seqs=[]
            for b in block:
                e=self.mlp(b)
                seqs.append(torch.cat([e,self.image_newline[None,None,:].expand(1,1,-1).squeeze(0)],dim=0))
            feats.append(torch.cat(seqs,dim=0))#把每个tile的拼接起来，得到一整条序列
        return feats

    def _ce_chunk(self,hidden,labels,lm_head_W):
            logits=F.linear(hidden.float(),lm_head_W.float())
            return F.cross_entropy(logits,labels,reduction='none')

    def _chunked_ce(self,hidden,labels,lm_head_W):
            hidden=hidden.reshape(-1,hidden.shape[-1])
            mask=labels!=load_data.IGNORE_INDEX
            h=hidden[mask]
            y=labels[mask]
            n=y.numel()
            if n==0:
                return hidden.sum()*0.0,0
            total=0.0
            for s in range(0,n,config['training']['loss_chunk']):
                hc,yc=h[s:s+config['training']['loss_chunk']],y[s:s+config['training']['loss_chunk']]
                total=total+torch.utils.checkpoint.checkpoint(self._ce_chunk,hc,yc,lm_head_W,use_reentrant=False).sum()
            return total/n,n

    def forward(self,pixel,vis_mask,meta,input_ids,labels,attention_mask):#把<image>替换为视觉特征,并计算损失
            visual=self.get_vision_feature(pixel,meta)
            base=self.llm.get_base_model()
            emb_layer=base.get_input_embeddings()
            embeds_list=[]
            for i in range(input_ids.shape[0]):
                e=emb_layer(input_ids[i])
                m=(input_ids[i]==load_data.IMAGE_TOKEN_INDEX)
                v = visual[i]
                if m.sum() != v.shape[0]:                            # 容错：截断对齐
                    k = int(m.sum())
                    v = v[:k] if v.shape[0] > k else F.pad(v, (0, 0, 0, k - v.shape[0]))
                e=e.masked_scatter(m.unsqueeze(-1),v.to(e.dtype))
                embeds_list.append(e)
            inputs_embeds=torch.stack(embeds_list,dim=0)
            hidden = base.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            use_cache=False).last_hidden_state     # 跳过 lm_head，省 B*T*V
            sh = hidden[:, :-1].contiguous()
            sl = labels[:, 1:].contiguous()
            loss, ntok = self._chunked_ce(sh, sl.reshape(-1), base.lm_head.weight)
            return loss, ntok

    @torch.no_grad()
    def generation(self,pixel,meta,prompt,max_new_tokens):
        self.eval()
        with autocast(device_type='cuda',dtype=torch.bfloat16):
            visual=self.get_vision_feature(pixel,meta)
        base=self.llm.get_base_model()
        emb_layer=base.get_input_embeddings()
        ids_list,emb_list=[],[]
        for i,p in enumerate(prompt):
            ids=llm_tokenizer(p,add_special_tokens=False)['input_ids']
            expanded,ci=[],0
            num_img_tokens=sum(1 for t in ids if t==load_data.IMAGE_TOKEN_INDEX)
            for t in ids:
                if t==load_data.IMAGE_TOKEN_INDEX:
                    expanded+=[load_data.IMAGE_TOKEN_INDEX]*(visual[i].shape[0]//max(1,num_img_tokens))
                    ci+=1
                else:
                    expanded.append(t)
            ids=torch.tensor(expanded,device=device)
            e=emb_layer(ids)
            m = (ids == load_data.IMAGE_TOKEN_INDEX)
            v = visual[i]
            if m.sum() != v.shape[0]:
                k = int(m.sum()); v = v[:k] if v.shape[0] > k else F.pad(v, (0, 0, 0, k - v.shape[0]))
            emb_list.append(e.masked_scatter(m.unsqueeze(-1),v.to(e.dtype)))
        L = max(e.shape[0] for e in emb_list)
        D = emb_list[0].shape[-1]
        emb = torch.zeros(len(emb_list), L, D, dtype=emb_list[0].dtype, device=device)
        attn = torch.zeros(len(emb_list), L, dtype=torch.long, device=device)
        for i, e in enumerate(emb_list):
            emb[i, :e.shape[0]] = e; attn[i, :e.shape[0]] = 1

        gc = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=llm_tokenizer.pad_token_id or llm_tokenizer.eos_token_id,
                              eos_token_id=llm_tokenizer.eos_token_id)#把本轮生成要用的超参数打包成一个配置对象，后面传给generate函数
        out = base.generate(inputs_embeds=emb, attention_mask=attn, generation_config=gc,
                            use_cache=True)                      
        return llm_tokenizer.batch_decode(out, skip_special_tokens=True)


def build_model():
    ve = load_vision_encoder()
    llm = load_llm_with_lora()
    proj = MlpProjection().to(device=device)
    model = LLaVAOneVision(ve, proj, llm).to(device)
    return model
model = build_model()

def build_optimizer(model):
    params=[{'params':[p for n,p in model.mlp.named_parameters()], 'lr':config['training']['lr_proj']},
             {'params':[model.image_newline], 'lr':config['training']['lr_proj']},
             ]
    llm_train = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    if llm_train:
        params.append({'params': llm_train, 'lr': config['training']['lr']})
    return optim.AdamW(params, weight_decay=config['training']['weight_decay'])

def train_one_epoch(model,optimizer):
    model.train()
    dl=load_data.get_train_dataloader()
    pbar=tqdm(dl,desc='training')
    loss_sum,tok_sum,seen=0.0,0,0
    optimizer.zero_grad(set_to_none=True)
    for step, (pixel, vis_mask, metas, input_ids, labels, attn) in enumerate(pbar, 1):
        pixel, vis_mask = pixel.to(device, non_blocking=True), vis_mask.to(device, non_blocking=True)
        input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            loss, ntok = model(pixel, vis_mask, metas, input_ids, labels, attn)
            (loss/config['training']['accum_steps']).backward()
        loss_sum += loss.item() * ntok; tok_sum += ntok; seen += 1
        if step % config['training']['accum_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], config['training']['grad_clip'])
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f'{loss_sum / max(1, tok_sum):.4f}', mem=f'{torch.cuda.max_memory_allocated()/1e9:.1f}G')
        del loss
        torch.cuda.empty_cache()#把没有用的内存释放掉
    return loss_sum / max(1, tok_sum)

@torch.no_grad()
def evaluate_bert(model, max_batches=40):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type='./opencv/bert-base-chinese',lang='zh',num_layers=12)
    model.eval()
    dl = load_data.get_test_dataloader()
    preds, refs = [], []
    for bi, (pixel, vis_mask, metas, input_ids, labels, attn) in enumerate(dl):
        if bi >= max_batches:
            break
        prompts = [random.choice(load_data.PROMPTS) for _ in range(pixel.shape[0])]
        outs = model.generation(pixel.to(device), metas,
                              [p.replace('这张图片', load_data.IMAGE_TOKEN).replace('图片', load_data.IMAGE_TOKEN)
                               if load_data.IMAGE_TOKEN not in p else p for p in prompts],max_new_tokens=50)
        preds.extend([o.strip() for o in outs])#[预测句子1，预测句子2，...]
        refs.extend([llm_tokenizer.decode([x for x in t if x != load_data.IGNORE_INDEX],
                                          skip_special_tokens=True) for t in labels])#[参考句子1，参考句子2，...]
    P, R, F1 = scorer.score(preds, refs,verbose=False)
    print(f'BERTScore  P:{P.mean():.4f}  R:{R.mean():.4f}  F1:{F1.mean():.4f}')
    model.train()
    return F1.mean()

def save_ckpt(model, epoch):
    """只存 LoRA + projector + newline，几百 MB -> 几 MB/几十 MB"""
    os.makedirs(config['data']['save_path'], exist_ok=True)#创建保护文件夹
    path = os.path.join(config['data']['save_path'], f'{epoch+1}.checkpoint.pth')
    payload = {
        'epoch': epoch,
        'projector': model.mlp.state_dict(),
        'image_newline': model.image_newline.detach().cpu(),
    }
    model.llm.save_pretrained(os.path.join(config['data']['save_path'], f'lora_{epoch+1}'))#专门保存LoRA参数
    torch.save(payload, path)
    print(f'已保存 -> {path}')

def load_ckpt(model, epoch):
    path = os.path.join(config['data']['save_path'], f'{epoch+1}.checkpoint.pth')
    ck = torch.load(path, map_location='cpu')
    model.mlp.load_state_dict(ck['projector'])
    with torch.no_grad():
        model.image_newline.copy_(ck['image_newline'].to(model.image_newline.device))
    model.llm.load_adapter(os.path.join(config['data']['save_path'], f'lora_{epoch+1}'),adapter_name='default')#专门加载LoRA参数
    print(f'已加载 <- {path}')

def main():
    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=config['training']['lr'] * 0.1)
    hist = []
    for ep in range(config['training']['epochs']):
        t0 = time.time()
        avg = train_one_epoch(model, optimizer)
        scheduler.step()
        print(f'轮次 {ep+1}/{config["training"]["epochs"]}  损失 {avg:.6f}  用时 {time.time()-t0:.0f}s  '
              f'峰值 {torch.cuda.max_memory_allocated()/1e9:.2f}G')
        hist.append(avg)
        save_ckpt(model, ep + 1)
        if (ep + 1) % 3 == 0:
            evaluate_bert(model, max_batches=40)
        gc.collect(); torch.cuda.empty_cache()#清除没有用的内存和缓存
    plt.figure(); plt.plot(hist); plt.xlabel('epoch'); plt.ylabel('loss')
    plt.savefig(os.path.join(config['data']['save_path'], 'llava_loss.png')); plt.close()

# ============================== 推理接口 ==============================
@torch.no_grad()
def load_for_inference(checkpoint_epoch):
    """加载训练好的模型，返回 model + tokenizer"""
    load_ckpt(model, checkpoint_epoch)
    model.eval()
    return model

@torch.no_grad()
def infer_image(model, image_path, prompt):
    """单图推理"""
    if prompt is None:
        prompt = random.choice(load_data.PROMPTS)
    img = Image.open(image_path).convert('RGB')
    pixel, meta = load_data.anyres_proc(img)
    pixel = pixel.unsqueeze(0).to(device)           # (1, n, 3, 384, 384)
    vis_mask = torch.ones(1, pixel.shape[1], dtype=torch.long, device=device)
    metas = [meta]
    
    prompt = prompt.replace('图片', load_data.IMAGE_TOKEN).replace('这张图片', load_data.IMAGE_TOKEN)
    result = model.generation(pixel, metas, [prompt], max_new_tokens=64)
    return prompt,result[0]

@torch.no_grad()
def infer_multi_image(model, image_paths, question):
    """多图推理"""
    imgs = [Image.open(p).convert('RGB') for p in image_paths]
    pixel, meta = load_data.multi_proc(imgs)
    pixel = pixel.unsqueeze(0).to(device)
    vis_mask = torch.ones(1, pixel.shape[1], dtype=torch.long, device=device)
    metas = [meta]
    
    question = question.replace('图片', load_data.IMAGE_TOKEN)
    result = model.generation(pixel, metas, [question], max_new_tokens=64)
    return question,result[0]

@torch.no_grad()
def infer_video(model, video_path, question):
    """视频推理"""
    from torchvision.io import read_video
    vframes, _, _ = read_video(video_path, pts_unit='sec')
    frames = [Image.fromarray(f.numpy()) for f in vframes]
    pixel, meta = load_data.video_proc(frames)
    pixel = pixel.unsqueeze(0).to(device)
    metas = [meta]
    
    question = question.replace('视频', load_data.IMAGE_TOKEN)
    result = model.generation(pixel, metas, [question], max_new_tokens=64)
    return question,result[0]


if __name__ == '__main__':
    main()
    model = load_for_inference(checkpoint_epoch=5)
    
    # 单图
    prompt, out = infer_image(model, config['evaluation']['test_image3'], '请描述这张图片。')
    print(f'输入: {prompt}  输出单图输出: {out}')
    
    # 多图
    question, out = infer_multi_image(model, [config['evaluation']['test_image2'], config['evaluation']['test_image3']], '这两张图片有什么不同？')
    print(f'输入: {question}  输出多图输出: {out}')
    
    # 视频
    #out = infer_video(model, './test.mp4', '视频里发生了什么？')
    #print(f'视频输出: {out}')