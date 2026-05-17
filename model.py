
from dataclass import dataclass
import torch
from torch import nn
from torch.nn  import Functional as F
import inspect



@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50257
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768

class MLP(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
    self.gelu = nn.GELU(approximate="tanh")
    self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
  def forward(self,X):
    X = self.c_fc(X)
    X = self.gelu(X)
    X = self.c_proj(X)

    return X

class Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self,x):
      x = x + self.attn(self.ln_1(x))
      x = x + self.mlp(self.ln_2(x))
      return x


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0

    self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
    self.c_proj = nn.Linear(config.n_embd,config.n_embd)


    self.n_head = config.n_head
    self.n_embd = config.n_embd

    self.c_proj.NANOGPT_SCALE_INIT = 1

    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

  def forward(self, X):
    B,T,C = X.size()


    qkv = self.c_attn(X)


    q,k,v = qkv.split(self.n_embd,dim=2)

    q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
    k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
    v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2)


    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y


class GPT2(nn.Module):

  def __init__(self,config):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size,config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd)
    ))

    self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

    self.transformer.wte.weight = self.lm_head.weight

    self.apply(self.init_weight)

  def init_weight(self,module):
      if isinstance(module, nn.Linear):
          std = 0.02
          if hasattr(module, 'NANOGPT_SCALE_INIT'):
              std *= (2 * self.config.n_layer) ** -0.5
          torch.nn.init.normal_(module.weight, mean=0.0, std=std)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


  @classmethod
  def from_pretrained(cls,model_type):
    assert model_type in {"gpt2","gpt2-medium","gpt-large","gpt-xl"}

    from transformers import GPT2LMHeadModel

    config_args = {
     "gpt2": dict(n_layer=12, n_head=12, n_embd=768) ,
     'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
     'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
     'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_type]

    config_args['vocab_size'] = 50257
    config_args['block_size'] = 1024

    config = GPTConfig(**config_args)

    model = GPT2(config)

    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = sd_hf.keys()

    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']


    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    print("Custom keys:", sd_keys[:5])
    print("HF keys:", sd_keys_hf[:5])
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):

            assert sd_hf[k].shape[::-1] == sd[k].shape, f"mismatched shape: {k}: {sd_hf[k].shape} {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:

            assert sd_hf[k].shape == sd[k].shape, f"mismatched shape: {k}: {sd_hf[k].shape} {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model


  def forward(self,idx,target=None):
    B,T = idx.size()

    assert T <= self.config.block_size, f"Cannot forward seq of length {T}, should less than {self.config.block_size}"

    pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
    pos_emb = self.transformer.wpe(pos)
    tok_emb = self.transformer.wte(idx)

    x = tok_emb + pos_emb #(B,T,E)

    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)


    logits = self.lm_head(x) #(B,T,C)

    loss = None
    if target is not None:
      loss = F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
    return logits,loss

  def configure_optimizer(self,weight_decay,learning_rate,device_type):
    param_dict = {pn: p for pn,p in self.model.parameters()}
    param_dict = {pn: p for pn,p in param_dict if p.requires_grad}

    decay_parameters = [p for n,p in param_dict.items() if p.dim() >= 2]
    non_decay_parameters = [p for n,p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_parameters, 'weight_decay': weight_decay},
        {'params': non_decay_parameters, 'weight_decay': 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_parameters)
    num_nodecay_params = sum(p.numel() for p in non_decay_parameters)


    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"

    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer