from pathlib import Path
from pprint import pprint
import time
from ch02 import get_device
from ch03 import (
    load_model_and_tokenizer, render_prompt, load_math500_test
)
from ch05 import heuristic_score
import torch
import platform

""" Ex 5.1 
Our goal is to use the heuristic scorer in chapter 5 as a tie-breaker in self consistency


 """
device = get_device()
model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)
""" 
math_data = load_math500_test();
pprint(math_data[0])
 """
used_libraries = [
    "torch",
]