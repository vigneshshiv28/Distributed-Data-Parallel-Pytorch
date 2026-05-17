import os 
import tiktoken
import torch

dataset_path = os.getenv("DATASET_PATH")


if not dataset_path:
    raise ValueError("DATASET_PATH is not set")


class DataLoaderLite():
  def __init__(self,B,T):
    self.B = B
    self.T = T

    with open(dataset_path,'r') as f:
      text = f.read()

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)

    print(f"loaded {len(self.tokens)} tokens")
    self.current_postion = 0
  def next_batch(self):
    B,T = self.B,self.T
    buf = self.tokens[self.current_postion:self.current_postion+(B*T)+1]
    x = buf[:-1].view(B,T)
    y = buf[1:].view(B,T)

    self.current_postion += B * T
    if self.current_postion + (B*T+1) > len(self.tokens):
      self.current_postion = 0
    return x,y
