from turtle import forward
import torch
import torch.nn as nn

""" 
All code in this example can be found here:
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/03_understanding-buffers/understanding-buffers.ipynb
 """

class CausalAttentionWithoutBuffers(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Old code without buffer
        #self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # new code with buffer
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
d_in = inputs.shape[1]
d_out = 2

ca_without_buffer = CausalAttentionWithoutBuffers(d_in, d_out, context_length, 0.0)

""" 

Code below confirms we are using cuda
 """
has_cuda = torch.cuda.is_available()
has_mps = torch.backends.mps.is_available()
if has_mps:
    device = torch.device("mps")   # Apple Silicon GPU (Metal)
elif has_cuda:
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # CPU fallback

#print(f"Using device: {device}")
#print("Machine has GPU:", has_cuda or has_mps)

batch = batch.to(device)
ca_without_buffer = ca_without_buffer.to(device)
ca_without_buffer.mask = ca_without_buffer.mask.to(device)

with torch.no_grad():
    context_vecs = ca_without_buffer(batch)

#print(context_vecs)

""" 
In the above we noticed that we had to move the mask to a device to avoid an error
Remembering to move individual tensors can be tedious

See modification to the code above

PyTorch buffers get included in the a model's state_dict
A state_dict is useful when saving and loading trained PyTorch models

 """

print(ca_without_buffer.state_dict())