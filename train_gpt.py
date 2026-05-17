import torch
from data_loader import DataLoaderLite
from model import GPTConfig,GPT2
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import os 

ddp = int(os.environ.get('RANK',-1)) != -1

if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

else :
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 4
T = 1024

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total batch size is divisible by batch"

accumulation_steps = total_batch_size // (B * T * ddp_world_size)

train_loader = DataLoaderLite(B=B,T=T)

model = GPT2(GPTConfig(vocab_size=50304))
model = model.to(device)
try:
    model = torch.compile(model)
    print("Compiled model")
except:
    print("Compile failed")

if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_step = 715
max_steps = 19073

def get_lr(step):
  if step < warmup_step:
    return max_lr * (step+1) / warmup_step
  if step > max_steps:
    return min_lr
  decay_ratio = (step - warmup_step) / (max_steps - warmup_step)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5*(1.0 + math.cos(math.pi*decay_ratio))

  return min_lr + coeff * (max_lr - min_lr)


optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


for step in range(max_steps):

  model.train()

  optimizer.zero_grad()
  start.record()

  last_step = step == max_steps-1

  step_loss = 0.0
  for micro_step in range(accumulation_steps):

    model.require_backward_grad_sync = (
      micro_step == accumulation_steps - 1
    )
    x,y = train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    logits,loss = model(x,y)


    loss = loss / accumulation_steps
    step_loss += loss.item()


    loss.backward()

    
  norm = torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=1.0)
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()

  end.record()
  torch.cuda.synchronize()
  time = start.elapsed_time(end)
  print(
      f"iter time: {time:.4f}s |"
      f"Norm: {norm}"
  )
  print(f"step {step}, loss:{step_loss}")

if ddp:
  destroy_process_group()








