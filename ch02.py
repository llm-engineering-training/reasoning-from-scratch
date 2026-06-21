from importlib.metadata import version
import torch
from torch.cpu import is_available
from reasoning_from_scratch.qwen3 import download_qwen3_small
from pathlib import Path
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B, KVCache
import warnings
import time

""" 
used_libraries = [
    "reasoning_from_scratch",
    "torch",
    "tokenizers"
]

for lib in used_libraries:
    print(f"{lib} version: {version(lib)}")

 """

""" 

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDS/ROCm GPU: {torch.cuda.get_device_name(0)}")

 """
""" 
#download the tokenizer files for the Qwen3 base LLM
download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")

 """
#download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")
tokenizer_path = Path("qwen3") / "tokenizer-base.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

prompt = "Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)

"""# The output below is an example of a BPE (Byte Pair Encoding)
BPE can represent both common and rare words using a mix of full words and
subword units. Note the Qwen3Tokenizer has a vocabulary of about 151,000 tokens
Which is relatively large
A larger vocabulary in a language model increases the model’s size because the embed-
ding and output layers must store more token representations
"""
""" for i in input_token_ids_list:
    print(f"{i} --> {tokenizer.decode([i])}") """

def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDS/ROCm GPU: {torch.cuda.get_device_name(0)}")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2,9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

device = get_device()
#download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")

model_path = Path("qwen3") / "qwen3-0.6B-base.pth"
model = Qwen3Model(QWEN_CONFIG_06_B) #instantial a Qwen3 model with random weights as placeholders
model.load_state_dict(torch.load(model_path)) #load the pretrained weights into the model
model.to(device=device) # transfer the model to the designated device

""" Uncomment to see model architecture
print(model)
 """

""" 
The following provides an example of the autoregressive text generation of LLMs

 """
promt ="Explain large language models."
input_token_ids_list = tokenizer.encode(prompt)
print(f"Number of input tokens: {len(input_token_ids_list)}")

input_tensore = torch.tensor(input_token_ids_list) # convert python list into PyTorch tensore
input_tensore_fmt = input_tensore.unsqueeze(0).to(device) # Add an additional dimension.

""" 
generate the output
 """
with torch.inference_mode():
    output_tensor = model(input_tensore_fmt)
output_tensor_fmt = output_tensor.squeeze(0) # remove the extra dimension we added earlier
#print(f"Formatted Output tensor shape: {output_tensor_fmt.shape}")
""" 
The above will return
Number of input tokens: 6
Formatted Output tensor shape: torch.Size([6, 151936])
 """

""" 
The following provides some additional information on the dimensions we added to the
tensor prior to text generation

The results will be
tensor([1, 2, 3])
tensor([[1, 2, 3]])
 """
""" 
example = torch.tensor([1,2,3])
print(example)
print(example.unsqueeze(0))
 """


""" 
last_token = output_tensor_fmt[-1] gets the last row of the 
torch.Size([6, 151936])  6 x 151,935 dimensional matrix
 """

last_token = output_tensor_fmt[-1]

#print(last_token)
#print(torch.argmax(last_token, dim=-1, keepdim=True))

""" 
torch.max() - returns the largest value in a tensore
torch.argmax() - returns the index of the largest value
When we use keepdim=True with torch.argmax() we keep the shape consistent by retaining the reduced dimension

The example below will produce
tensor(3)
tensor(2)
tensor([2])

 """

""" 
example = torch.tensor([-2, 1, 3, 1])
print(torch.max(example))
print(torch.argmax(example))
print(torch.argmax(example, keepdim=True))
 """
@torch.inference_mode() # disables gradient tracking for speed and memory efficiency
def generate_text_basic_stream(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    """ switch model to evaluation mode to enable deterministic behavior - this is best practive  """ 
    model.eval()
    """ We added the followin to inclide KV caching in the text generation process  """
    cache = KVCache(n_layers=model.cfg["n_layers"])  # New
    model.reset_kv_cache()                           # New 
    out = model(token_ids, cache=cache)[:, -1] # out in for originally moved here
    for _ in range(max_new_tokens):
        """ get the scores of the last token update removed since we are now implementing KV caching """
        #out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        """ stop if we encounter an end-of-sequence token """
        if(eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break
        """ Yield each token as it's generated """
        yield next_token
        """ Append the newly predicted token to the sequence - update removed since we are now implementing KV caching """
        #token_ids = torch.cat([token_ids, next_token], dim=1)
        out = model(next_token, cache=cache)[:, -1] # New

prompt = "Explain large language models in a single sentence."
input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device
).unsqueeze(0)
max_new_tokens = 100


""" This is a utility function that measures the runtime of the text generation process """
def generate_stats(
    output_token_ids,
    tokenizer,
    start_time,
    end_time
):
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")
    for name, backend in (("CUDA", getattr(torch, "cuda", None)),("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )
            if hasattr(backend, "synchronize"):
                backend.synchronize()
            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()

start_time = time.time()
generated_ids = []

""" 
for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=max_new_tokens,
    eos_token_id=tokenizer.eos_token_id  # Use EOS token try also by commenting this out
):
    token_id = token.squeeze(0).tolist() # convert output token IDs from PyTorch tensor to list
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True # Deactivates buffering so tokens are printed live
    )
    next_token_id = token.squeeze(0)
    generated_ids.append(next_token_id)
end_time = time.time()
output_token_ids_tensor = torch.cat(generated_ids, dim=0)
generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)

 """


""" 
The results from our generate_stats() is
Time: 2.40 sec
17 tokens/sec
Max CUDA memory allocated: 1.48 GB

 """

""" 
In neural networks context, inference, is specifically, taking a model whose parameters are already
learned and fixed, running a forward pass and producing predictions. This is neural network inference,
which is just the foward application of a trained model, since the model is a fixed function

In statistics, inference involves estimating parameters, quantifying uncertainty or testing hypotheses

In what follows we will consider KV caching and model compilation to speed up our inference

KV (key and values used in the model's attention mechanism) caching involves storing intermediate values computed in each iteration
in a cache. In the above each new token generated the network was concatenated to the entire input sequence and fed back into the model
repeatedly. By using a KV cache we avoid redundant computation

In the generate_text_basic_stream() with KV caching added the generate_stats()
returns
generate_text_basic_stream 
Time: 1.73 sec
23 tokens/sec
Max CUDA memory allocated: 1.46 GB

Note: model is initialized and loaded on line 83
For understanding and loading KV cache see: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms

 """

major, minor = map(int, torch.__version__.split(".")[:2])
if (major, minor) >= (2, 8):
    # This avoids retriggering model recompilations 
    # in PyTorch 2.8 and newer
    # if the model contains code like self.pos = self.pos + 1
    torch._dynamo.config.allow_unspec_int_on_nn_module = True

model_compiled = torch.compile(model)

for i in range(3):

    start_time = time.time()
    generated_ids = []
    
    for token in generate_text_basic_stream(
        model=model_compiled,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )
    
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id)  # Collect generated tokens
    
    end_time = time.time()
    

    if i == 0:
        print("\n\nWarm-up run")
    else:
        print(f"\n\nTimed run {i}:")

    output_token_ids_tensor = torch.cat(generated_ids, dim=0)
    generate_stats(output_token_ids_tensor, tokenizer, start_time, end_time)

    print(f"\n{30*'-'}\n")





