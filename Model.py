# -*- coding: utf-8 -*-

import sys
import logging
import torch
import math
import numpy as np
import glob
from Data import Vocab

def numparameters(model):
  npars = 0 #pars
  nbytes = 0 #bytes
  for name, param in model.named_parameters():
    if param.requires_grad: #learnable parameters only
      npars += param.numel()
      nbytes += param.numel() * param.data.element_size() #returns size of each parameter
      logging.debug("{} => {} = {} x {} bytes".format(name, list(param.data.size()), param.data.numel(), param.data.element_size()))

  name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
  if nbytes == 0:
    i = 0
  else:
    i = int(math.floor(math.log(nbytes, 1024)))
    p = math.pow(1024, i)
    nbytes /= p
  size = "{:.2f}{}".format(nbytes, name[i])

  return npars, size

def save_checkpoint(suffixmodel, optimizer, step, keep_last_n):
  checkpoint = { 'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict() }
  torch.save(checkpoint, "{}.checkpoint_{:08d}.pt".format(suffix,step))
  logging.info('Saved {}.checkpoint_{:08d}.pt'.format(suffix,step))
  files = sorted(glob.glob(suffix + '.checkpoint_????????.pt')) 
  while keep_last_n > 0 and len(files) > keep_last_n:
    f = files.pop(0)
    os.remove(f) ### first is the oldest
    logging.debug('Removed checkpoint {}'.format(f))

def load_checkpoint_or_initialise(suffix, model, optimizer):
  step = 0
  files = sorted(glob.glob("{}.checkpoint_????????.pt".format(suffix))) ### I check if there is one model
  if len(files) == 0:
    for p in model.parameters():
      if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
    logging.info('No model found [network initialised]')
    return step, model, optimizer

  file = files[-1] ### last is the newest
  checkpoint = torch.load(file)
  step = checkpoint['step']
  ### assert checkpoint['model'] has same options than model
  model.load_state_dict(checkpoint['model'])
  if optimizer is None:
    logging.info('Loaded model step={} from {}'.format(step,file))
    return step, model, optimizer ### this is for inference

  optimizer.load_state_dict(checkpoint['optimizer'])
  logging.info('Loaded model/optimizer step={} from {}'.format(step,file))
  return step, model, optimizer ### this is for learning

##############################################################################################################
### Endcoder_Decoder #########################################################################################
##############################################################################################################
class Encoder_Decoder(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout, src_voc_size, tgt_voc_size, idx_pad): 
    super(Encoder_Decoder, self).__init__()
    self.idx_pad = idx_pad
    self.src_emb = torch.nn.Embedding(src_voc_size, emb_dim, padding_idx=idx_pad)
    self.tgt_emb = torch.nn.Embedding(tgt_voc_size, emb_dim, padding_idx=idx_pad)
    self.pos_enc = PositionalEncoding(emb_dim, dropout, max_len=5000)
    self.stacked_encoder = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.stacked_decoder = Stacked_Decoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.generator = Generator(emb_dim, tgt_voc_size)
    logging.debug('n_layers={}'.format(n_layers))
    logging.debug('ff_dim={}'.format(ff_dim))
    logging.debug('n_heads={}'.format(n_heads))
    logging.debug('emb_dim={}'.format(emb_dim))
    logging.debug('qk_dim={}'.format(qk_dim))
    logging.debug('v_dim={}'.format(v_dim))
    logging.debug('Vs={}'.format(src_voc_size))
    logging.debug('Vt={}'.format(tgt_voc_size))

  def forward(self, src, tgt):
    src = torch.LongTensor(src) #[bs,ls]
    tgt = torch.LongTensor(tgt) #[bs,lt]
    msk_src = (src != self.idx_pad).unsqueeze(-2) #[bs,1,ls] (False where <pad> True otherwise)
    msk_tgt = (tgt != self.idx_pad).unsqueeze(-2) & (1 - torch.triu(torch.ones((1, tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1)).bool() #[bs,lt,lt]

    src = self.pos_enc(self.src_emb(src)) #[bs,ls,ed]
    tgt = self.pos_enc(self.tgt_emb(tgt)) #[bs,lt,ed]
    z_src = self.stacked_encoder(src, msk_src) #[bs,ls,ed]
    z_tgt = self.stacked_decoder(z_src, tgt, msk_src, msk_tgt) #[bs,lt,ed]
    y = self.generator(z_tgt) #[bs, lt, Vt]
    return y

##############################################################################################################
### Stacked_Encoder ##########################################################################################
##############################################################################################################
class Stacked_Encoder(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout): 
    super(Stacked_Encoder, self).__init__()
    self.encoderlayers = torch.nn.ModuleList([Encoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])

  def forward(self, src, msk):
    for i,encoderlayer in enumerate(self.encoderlayers):
      src = encoderlayer(src, msk) #[bs, ls, ed]
    return src 

##############################################################################################################
### Stacked_Decoder ##########################################################################################
##############################################################################################################
class Stacked_Decoder(torch.nn.Module):
  def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout): 
    super(Stacked_Decoder, self).__init__()
    self.decoderlayers = torch.nn.ModuleList([Decoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])

  def forward(self, z_src, tgt, msk_src, msk_tgt):
    for i,decoderlayer in enumerate(self.decoderlayers):
      tgt = decoderlayer(z_src, tgt, msk_src, msk_tgt)
    return tgt 

##############################################################################################################
### Encoder ##################################################################################################
##############################################################################################################
class Encoder(torch.nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Encoder, self).__init__()
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.multihead_attn = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.norm = LayerNorm(emb_dim)

  def forward(self, src, msk):
    #logging.info('begin Encoder fwd')
    tmp = self.norm(self.multihead_attn(q=src, k=src, v=src, msk=msk) + src) #[bs, ls, ed] # self-attention to src embeddings
    z = self.norm(self.feedforward(tmp) + tmp) #[bs, ls, ed]  
    #logging.info('end Encoder fwd')
    return z

##############################################################################################################
### Decoder ##################################################################################################
##############################################################################################################
class Decoder(torch.nn.Module):
  def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(Decoder, self).__init__()
    self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
    self.multihead_attn = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
    self.norm = LayerNorm(emb_dim)

  def forward(self, z_src, tgt, msk_src, msk_tgt):
    #add causal mask to msk_tgt
    tmp = self.norm(self.multihead_attn(q=tgt, k=tgt, v=tgt, msk=msk_tgt) + tgt) #[bs, lt, ed] causal attention to previous tgt words (decoder attention)
    tmp = self.norm(self.multihead_attn(q=tmp, k=z_src, v=z_src, msk=msk_src) + tmp) #[bs, ls, ed] self-attention to src embeddings (encoder attention)
    z = self.norm(self.feedforward(tmp) + tmp)
    return z

##############################################################################################################
### MultiHead_Attn ###########################################################################################
##############################################################################################################
class MultiHead_Attn(torch.nn.Module):
  def __init__(self, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(MultiHead_Attn, self).__init__()
    self.nh = n_heads
    self.ed = emb_dim
    self.qd = qk_dim
    self.kd = qk_dim
    self.vd = v_dim
    self.WQ = torch.nn.Linear(emb_dim, qk_dim*n_heads)
    self.WK = torch.nn.Linear(emb_dim, qk_dim*n_heads)
    self.WV = torch.nn.Linear(emb_dim,  v_dim*n_heads)
    self.WO = torch.nn.Linear(v_dim*n_heads, emb_dim)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, q, k, v, msk):
    #q is [bs, slq, ed]
    #k is [bs, slk, ed]
    #v is [bs, slv, ed]
    bs = q.shape[0]
    slq = q.shape[1] 
    slk = k.shape[1]
    slv = v.shape[1]
    ed = q.shape[2]
    assert self.ed == q.shape[2] == k.shape[2] == v.shape[2]
    assert slk == slv #when applied in decoder both may be be referred to the seq-length of the source-side while slq is referred to the target-side

    Q = self.WQ(q.contiguous().view([bs*slq,self.ed])) #[bs*slq,qd*nh]
    K = self.WK(k.contiguous().view([bs*slk,self.ed])) #[bs*slk,kd*nh]
    V = self.WV(v.contiguous().view([bs*slv,self.ed])) #[bs*slv,vd*nh]

    Q = Q.contiguous().view([bs,slq,self.qd,self.nh]).permute(3,0,1,2) #=> [bs,slq,qd,nh] => [nh,bs,slq,qd]
    K = K.contiguous().view([bs,slk,self.kd,self.nh]).permute(3,0,1,2) #=> [bs,slk,kd,nh] => [nh,bs,slk,kd]
    V = V.contiguous().view([bs,slv,self.vd,self.nh]).permute(3,0,1,2) #=> [bs,slv,vd,nh] => [nh,bs,slv,vd]

    z = torch.cat([self.Attn(Q[n], K[n], V[n], msk) for n in range(self.nh)], dim=2) #cancats the nh 3d-matrices in the last (d=2) dimension [bs, slk, vd*nh]
    z = self.WO(z) #[bs, slk, ed]
    return self.dropout(z)

  def Attn(self, Q, K, V, msk): 
    #Q is [bs, slq, qd]
    #K is [bs, slk, kd]
    #V is [bs, slv, vd]
    #msk is [bs, 1, ls] or [bs, lt, lt]
    ### implements scaled dot-product attention for q, k, v
    s = Q.bmm(K.transpose(1, 2)) #[bs, slq, qd] x [bs, kd, slk] = [bs, slq, slk] (qd must be equal to kd)
    s = s.masked_fill(msk == 0, -1e9) #[bs, slq, slk] (very low score for <pad> words)
    w = torch.nn.functional.softmax(s / Q.shape[2]**0.5, dim=2) #[bs, slq, slk]
    z = w.bmm(V) #[bs, slq, slk] X [bs, slv, vd] = [bs, slq, vd]
    return z

##############################################################################################################
### FeedForward ##############################################################################################
##############################################################################################################
class FeedForward(torch.nn.Module):
  def __init__(self, emb_dim, ff_dim, dropout):
    super(FeedForward, self).__init__()
    self.FF_in = torch.nn.Linear(emb_dim, ff_dim)
    self.FF_out = torch.nn.Linear(ff_dim, emb_dim)
    self.dropout = torch.nn.Dropout(dropout) #this regularization is not used in the original model

  def forward(self, x):
    return self.FF_out(self.dropout(torch.nn.functional.relu(self.FF_in(x))))

##############################################################################################################
### LayerNorm ################################################################################################
##############################################################################################################
class LayerNorm(torch.nn.Module):
  def __init__(self, dim, eps=1e-6):
    super(LayerNorm, self).__init__()
    self.a_2 = torch.nn.Parameter(torch.ones(dim))
    self.b_2 = torch.nn.Parameter(torch.zeros(dim))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

##############################################################################################################
### PositionalEncoding #######################################################################################
##############################################################################################################
class PositionalEncoding(torch.nn.Module):
  def __init__(self, emb_dim, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = torch.nn.Dropout(dropout)
    pe = torch.zeros(max_len, emb_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe) #register_buffer are for params saved&restored in state_dict not trained 

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

##############################################################################################################
### Generator ################################################################################################
##############################################################################################################
class Generator(torch.nn.Module):
  def __init__(self, emb_dim, voc_size):
    super(Generator, self).__init__()
    self.proj = torch.nn.Linear(emb_dim, voc_size)

  def forward(self, x):
    return torch.nn.functional.log_softmax(self.proj(x), dim=-1)


