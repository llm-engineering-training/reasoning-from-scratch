"""
* start 06-28-2026
Our main tasks will be
* prompting an LLM to explain its reasoning to improve accuracy
* Modifying the text generation function to produce diverse responses
* Improve reasoning reliability by sampling multiple responses

 We want to discuss techniques to improve model reasoning without model training
 In machine learning (ML) and AI, "compute" refers to the computational resources required to train or run a model
 Two main strategies to improve reasoning are:
 1 - Increase training compute
 2 - Increase inference compute (inference-time scaling) - increasing computation at answer-generation time, that is the model does more work per question
    Chain-of-thought - prompt the model to explain reasoning, simple and effective
    self-consistency or Parallel Sampling -  model generates multiple responses and selects the most frequent one
    Self-refinement - model reviews and improves its own reasoning across multiple steps
(4.12)
 """
import torch
import matplotlib.pyplot as plt
from torch.cuda import temperature
from collections import Counter
from ch02 import (
    get_device,
    generate_text_basic_stream
)
from ch03 import(
    load_model_and_tokenizer, render_prompt, extract_final_candidate
)
from reasoning_from_scratch.qwen3 import KVCache
device = get_device()
model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False,
)

raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)

#print(prompt)

""" 
We want to compare several inference-time scaling strategies
To do this we'll modify the text_generation wrapper to let us swap in
different generation functions and settings without  changing the surrounding
prompt handling and output code
(4.2)
 """

def generate_text_stream_concat_flex(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False, 
    generate_func=None,
    **generate_kwargs
):
    """ In the function definition we added a parameter to accept a text generation and additional arguments """
    if generate_func is None:
        generate_func = generate_text_basic_stream
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)
    generated_ids = []
    for token in generate_func(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)

""" 
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_basic_stream  # NEW
)
 """

""" 
The generated response from the above is:
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
 \boxed{20}% 
 Which is incorrect
 We are going to attempt chain-of-thought prompting to improve accuracy
Which forces the model to produce the result step-by-step
 """

 #Uncomment below to see output
""" 

prompt_cot = prompt + " \n\nExplain step by step."
response = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=True,
)
 """


""" 
With chain-of-thought shows that the model expends many more tokens
Note: Not all problems can benefit from chain-of-thought. On simple problems it can degrade the models performance
Reasoning models will not benefit from chain-of-thought prompting
Chain-of-thought prompting changes how the model uses existing knowledge, often this can lead to 
more reliable answers especially for math, code, logic and other multistep problems
Specifically, chain-of-thought is a sequential technique because it extends the number of
next-token prediction steps
For the remainder of this chapter we'll implement a self-consistency (Parallel sampling) 
The technique is formally described in the following Google Research paper: https://arxiv.org/abs/2203.11171
* end 06-28-2026
* start 07-04-2026
To implement self-consistency sampling we need to extend the text generation function to produce
different answers for the same prompt. We implement two techniques that allow the model
to sample different responses: "temprature scaling" and "top-p filtering"
 """

""" 
Understanding the process of selecting the next token
 """


""" 
The example below illustrates something interesting, we left max_new_token large, and we found out that the LLM
would just continue to repeat itself until it runs out of tokens.
 """

ex_prompt = "The capital of Germany is"
""" 
response = generate_text_stream_concat_flex(
    model, tokenizer, ex_prompt, device,
    max_new_tokens=1, verbose=True,
)
 """


""" 
Below we illustrate the steps the model takes to generate the answer 
"Berlin" from the above 

1) convert the input into token ids
Output: tensor([[ 785, 6722,  315, 9856,  374]])
 """


input_token_ids = torch.tensor(
    tokenizer.encode(ex_prompt), device=device
).unsqueeze(0)

#print(input_token_ids)


""" 
2) get scores for the next token
Note: The LLM generates one output token for each input token, but we are only
interested in the last token which we select via [:,-1] tensore indexing

Output:torch.Size([1, 151936])
where 151936 is the vocabulary size of the tokenizer model...all  the unique tokens
the tokenizer can handle and the LLM can generate 
 """



with torch.inference_mode():
    next_token_logits = model(input_token_ids)[:,-1]

""" 
#Output: torch.Size([1, 151936])
print(next_token_logits.shape)
In the following we still need the next_token_logits property

 """


""" 
3) we find the vocabulary index associated with the largest score
 """

max_token_id = torch.argmax(next_token_logits)
#print(f"Token ID: {max_token_id}")
#print(f"Decoded token: '{tokenizer.decode([max_token_id])}'")


""" 
To take a closer look at the score distribution of the next_token_logits we 
create the following function after importing matplotlib.pyplot as plt
(4.3)
 """

def plot_scores_bar(
    next_token_id, start=19_800, end=19_900,
    arrow=True, ylabel="Logit value"
):
    x = torch.arange(start, end)
    """ .cpu() is short for to(torch.device("cpu")) """
    logits_section = next_token_logits[0, start:end].float().cpu()
    """ Plot the logits """
    plt.bar(x,logits_section)
    plt.xlabel("Vocabulary index")
    plt.ylabel(ylabel)
    """ Highlight the max logit """
    if arrow:
        max_idx = torch.argmax(logits_section)
        plt.annotate(
            "Berlin",
            xy=(x[max_idx], logits_section[max_idx]),
            xytext=(x[max_idx] - 25, logits_section[max_idx] - 2),
            arrowprops={
                "facecolor": "black", "arrowstyle": "->", "lw": 1.5
            },
            fontsize=10,
        )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("bar_new_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

#plot_scores_bar(next_token_logits)


""" 
We want to rescale the next-token logits with a temperature parameter
before using them for sampling. We adjust the magnitude of the scores
to become more or less sensitive to the differences in scores
In the function below, a temperature < 1 makes the distribution sharper and makes the model more confident
whereas a temperature > 1 flattens the distribution and makes the sampling more diverse
(4.4)
 """
def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature



""" 
Below we create another plotting function to use the temperature parameter in (4.4)
Term "temperature" comes from physics where temperature controls how much randomness
or movement there is in a system.
(4.5)
 """
def plot_logits_with_temperature(
    next_token_logits, start=19_800, end=19_900,
    temps=(0.5, 5.0),
):
    x = torch.arange(start, end)
    logits_orig = next_token_logits[0, start:end].float().cpu()

    # Apply temperature scaling
    logits_scaled = [
        scale_logits_by_temperature(logits_orig, T) for T in temps
    ]
    # Plot logits
    plt.plot(x, logits_orig, label="Original logits", lw=2)
    plt.plot(
        x, logits_scaled[0],
        label=f"T={temps[0]} (sharper)", ls="--", lw=1
    )
    plt.plot(
        x, logits_scaled[1],
        label=f"T={temps[1]} (flatter)", ls=":", lw=3
    )

    # Highlight max logit
    max_idx = torch.argmax(logits_orig)
    plt.annotate(
        "Berlin",
        xy=(x[max_idx], logits_orig[max_idx]),
        xytext=(x[max_idx] - 25, logits_orig[max_idx] + 2),
        arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 1.5},
        fontsize=12,
    )

    plt.xlabel("Vocabulary index")
    plt.ylabel("Logit value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("temperatur_logits_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

""" plot_logits_with_temperature(
    next_token_logits,
    temps=(0.5, 5.0)
) """

""" 
Next we want to be able to convert the rescaled logits into probability scores
Converting rescaled logits into probability scores can be done with torch.softmax
(4.6)
 """
rescaled_logits = scale_logits_by_temperature(next_token_logits, 5.0)
next_token_probas = torch.softmax(
    rescaled_logits, dim=-1
)

""" 
plot_scores_bar(
    next_token_probas, arrow=False, ylabel="Probability value"
)
 """


"""  
torch.softmax normalizes the logit values. In essence

softmax(z(i)) = exp(z(i)) / sum_over(j)exp(z(j))
i = index of the current element in (1 <= i <- n)
j is the index used to sum over all elements (1 <= j <- n)
z = [z(1), z(2),...,z(n)]
The above produces a normalized probability for each z, such that the sum of
the softmax over all a is equal to one
The purpose of converting the probabilities is to make them more interpretable when we use
torch.multinomial function 
"""
torch.manual_seed(123)


""" 
print(
    "Sampled token:",
    torch.multinomial(next_token_probas.cpu(), num_samples=1)
)
print(tokenizer.decode([65094]))

 """

""" 
The above will output: Sampled token: tensor([[65094]])
65094 corresponds to the word mistress
Note: torch.multinomial function samples vocabulary indices in proportion to their
probabilities. The following function allows us to sample multiple next-token candidates.
samples token indices from a probability distribution and counts how often each token is
drawn
(4.7)
 """
def count_samples(probas, num_samples=1000, threshold=1,
    tokenizer=None
):
    """ draw samples according to probability, this is in lieu of always selecting the most likely token
    known, as greedy decoding...the variability allows us to generate multiple candidate answers for
    self consistency later
     """
    samples = torch.multinomial(
        probas.cpu(), num_samples=num_samples, replacement=True
    )
    """ count how often each index was selected """
    counts = torch.bincount(samples.squeeze(0), minlength=1)
    """ Print frequently samples vocabulary indices (entries) """
    for i, c in enumerate(counts):
        if c > threshold:
            if tokenizer is None:
                print(f"Vocab index {i}: {c.item()}x")
            else:
                print(f"'{tokenizer.decode([i])}': {c.item()}x")


""" 
count_samples(next_token_probas, tokenizer=tokenizer)
 """


""" 
The output is:
'}': 2x
' your': 2x
' "<': 2x
' bitter': 2x
' rejects': 2x
'�': 2x
' trẻ': 2x
None of the sampled tokens appeared more than twice
All are nonsense tokens in the context of our prompt which is
"The capital of Germany is" -  the reason for this is that the temperature
we used was too high

 """
probas_lowT =  torch.softmax(
    scale_logits_by_temperature(next_token_logits, 0.35), dim=-1
)

""" end 07-04-2026 Uncomment below to generate the results discussed in the comment-block below """
#count_samples(probas_lowT, tokenizer=tokenizer)

""" start 07-05-2026  
In probas_lowT we reduced the temperature from 5 to 0.35 and the results now make much more sense.
Output with probas_lowT is:
': 2x
' __': 119x
' Berlin': 499x
' ____': 165x
' ______': 184x
' Munich': 6x
' Hamburg': 7x
' _____': 16x
We're intentionally adding an additional layer of randomness with the multinomial sampling
to help the model explore alternative responses -  this variability is useful for creative or open-ended
tasks where they may be multiple valid completions. Below we are going to add the temperature-scaling 
modification to the next generation function to more readily generate new tokens with the model
(4.8) - original name was: generate_text_temp_stream_cache
"""
@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.,
    top_p=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    """ Step 3.1 above, get the logits """
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        """ Start new code """
        orig_device = token_ids.device
        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        else:
            """ step 3.2 above apply temperature scaling to logits """
            logits = scale_logits_by_temperature(out, temperature)
            """ step 3.3 above convert to probabilities """
            probas = torch.softmax(logits, dim=1)
            """ step 4 outlined below - apply top-p filter to probabilities and rename method to 
            generate_text_top_p_stream_cache """
            probas = top_p_filter(probas, top_p)
            """ step 3.4 above sample the tokens according to probabilities """
            next_token =  torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)
        """ End new code """ 
        if (eos_token_id is not None
            and torch.all(next_token == eos_token_id)):
            break
        yield next_token
        out = model(next_token, cache=cache)[:,-1]   

""" 
Note: the above is similar to the generate_text_basic_stream function we wrote in ch02.py after we added KVCache
but now we added temperature scaling and sampling.
If we use:
raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
and we re-tun the generate_text_stream_concat_flex with generate_text_temp_stream_cache above
(renamed to: generate_text_top_p_stream_cache)we still get a wrong answer of 11, but this was only meant as a demonstration.
Temperature selection depends on the goal. Temperature = 0.0  is gready decoding
since we always pick the highest probability token. Values 0.3-0.8 are useful when we
want more diversity without making the output too erratic. Higher values make the model
explore more broadly.
Our goal now is to improve the sampling process by adding a top-p filter so that
low-confidence tokens are not sampled by accident. Our focus will be the "nucleus sampling"
technique. Which will have the following steps
(4.1) - sort sample in descending order
(4.2) -  compute cumulative sum
(4.3) -  apply top-p threshold
(4.4) -  renormalize
 """

""" 

response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_temp_stream_cache, #the new code above use new name: generate_text_top_p_stream_cache
    temperature=1.1
)

 """


"""
In the following we are going to review some of the steps we discussed earlier
(4.9)
"""
""" step 3.1 get the logits (these are toy logits for practive) - example values for the next-token logit scores """
toy_logits = torch.tensor(
    [-0.7, -3.0, 0.1, -1.2, 2.0, -1.0, -0.5, -2.0, 0.3, 1.5]
)

""" step 3.2 apply temperature scaling """
toy_logits_scaled = scale_logits_by_temperature(toy_logits, 1.0)

""" step 3.3 convert to probabilities """
toy_probas = torch.softmax(toy_logits_scaled, dim=-1) 

""" 
#Uncomment to generate plot
plt.bar(
    torch.arange(len(toy_logits_scaled)), toy_probas,
    alpha=0.5
)
plt.ylim([0, 1])
plt.xlabel("Vocabulary index")
plt.ylabel("Probability")
plt.savefig("toy_probas.png")

 """


""" 
Now we'll add the first two top-p filter steps
(4.10)
 """

""" 4.1 sort by descending probability """
sorted_probas, sorted_idx = torch.sort(toy_probas, descending=True)

""" 4.2 compute cumulative sum 
the torch.cumsum computes the cumulative sum of elements along a given dimension
"""
cumsum = torch.cumsum(sorted_probas, dim=-1)


""" 
#Uncomment to generate the cumulative sum plot
plt.bar(
    torch.arange(len(sorted_probas)), sorted_probas, 
        alpha=0.5
)
plt.step(
    torch.arange(len(cumsum)), cumsum, 
    where="mid", color="C1", label="Cumulative sum"
)

plt.ylim([0, 1])
plt.xlabel("Token rank (sorted by probability)")
plt.ylabel("Probability")
plt.savefig("cum_toy_probas.png")

 """


""" 
Now we'll implement the core top-p filtering step
(4.11)
 """

""" 4.3.1 apply top-p threshold e.g., keep tokens until cumulative mass > 0.8
(4.11)
"""
top_p = 0.8

""" 

keep_mask = cumsum <= top_p
n_kept = torch.sum(keep_mask).item()
print("Cumulative sum:", cumsum)
print("Tokens kept:", n_kept)

 """

""" 
The above will  output
Cumulative sum: tensor([0.4538, 0.7290, 0.8119, 0.8798, 0.9170, 0.9475, 0.9701, 0.9886, 0.9969,
        1.0000])
Tokens kept: 2
the more common variant is to include the token that exceeds the threshold
The output from the refactord code below will be:
Tokens kept: 3
(4.12) - we'll skip (4.13) because it is another plot to illustrate the refactored top-p filter
"""
keep_mask = (cumsum - sorted_probas) < top_p
n_kept = keep_mask.sum().item()
#print("Tokens kept:", n_kept)

""" 4.3.2 sero out beyond the cutoff
(4.14)
"""
kept_sorted = torch.where(
    keep_mask, sorted_probas,
    torch.zeros_like(sorted_probas)
)

"""  Step 4.3.3: Map back to original order 
filtered below looks like:
tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.4538, 0.0000, 0.0000, 0.0000, 0.0829,
        0.2752])
"""
filtered = torch.zeros_like(toy_probas).scatter(0, sorted_idx, kept_sorted)

#print(filtered)

"""  Step 4.4: Renormalize to sum to 1 """

""" 
denom = torch.sum(filtered).clamp_min(1e-12)
renormalized = filtered / denom
print(renormalized)

 """


"""
we assemble the steps into a single convenient function
(4.15)
"""
def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas
    """ 4.1 sort by descending probability """
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True) 
    """ 4.2 cumulative sum """
    cumprobas = torch.cumsum(sorted_probas, dim=1)
    """ 4.3.1 keep tokens where prefix cumulative mass is < top_p """
    prefix = cumprobas - sorted_probas
    keep = prefix < top_p
    """ always keep atleast one """
    keep[:, 0] = True 
    """ 4.3.2 zero out beyond the cutoff """
    kept_sorted = torch.where(
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )
    """ 4.3.3 map bsck to original order """
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)
    """ 4.4 renormalize to sum to 1 """
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom


#Uncomment to see output below
""" 

probas_lowT_filtered = top_p_filter(probas_lowT, top_p=0.8)
count_samples(probas_lowT_filtered, threshold=1, tokenizer=tokenizer)

 """
"""
Using top_p_filter above on  the probas_lowT defined above in the chapter we now
get output:
 ' Berlin': 596x
' ____': 183x
' ______': 221x

Now we return to the math query and we'll add the top-p filter to the function in (4.8)
we add the top_p parameter since the renamed generate_text_top_p_stream_cache can now 
accept a top_p which defaults to None
 \boxed{18}% 
 """

""" end 07-05-2026 """
""" 
print(prompt)

response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.5,
    top_p=0.8, 
)
 """


"""
start 07-15-2026
ex (4.2) Modify evaluate_math500_stream from (3.15) by adding temperature scaling and top-p
to see if the base model accuracy changes.
Improving response accuracy with self-consistency
Now we are going to implement self-consistency sampling, which was introduced in a Google Research paper 
here: https:/ /arxiv.org/abs/2203.1117
- In simple term it is a form of majority voting -  We use temperature scaling and top-p filtering to 
generate multiple answers, and then select the most frequent one
Note, this is generally a time-scaling technique since we don't update the model itself
Process:
    1 - generate multiple answers using a temperature > 0 and top-p filtering,
    2 - extract the final answer from each generated solution
    3 - select the most frequently extracted answer as the final result
(4.17)
"""

def self_consistency_vote(
    model, tokenizer, prompt, device,
    num_samples=10, temperature=0.8, top_p=0.9, max_new_tokens=2048,
    show_progress=True, show_long_answer=False, seed=None,
):
    full_answers, short_answers = [],[]
    """ 1 - sample multiple answers """
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)
        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )
        """ 2 - extract the final (short) answer from each answer """
        short = extract_final_candidate(
            answer, fallback="number_then_full"
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] -> {short!r}")
    """ 3 - choose the most frequent final answer (self-consistency vote) """
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)
    mc = counts.most_common()
    if not mc:
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        final_answer = mc[0][0] if len(majority_winners) == 1 else None
    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }
"""
Our implementation above uses a for loop to generate answers sequentially,
in practive we use different devices to parallelize the sampling 
"""

""" 
results = self_consistency_vote(
    model,
    tokenizer,
    prompt,
    device=device,
    num_samples=5,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    seed=123,
    show_progress=True,
)
print(results["final_answer"])

"""

"""
The above will output:
[Sample 1/5] -> '83'
[Sample 2/5] -> '22'
[Sample 3/5] -> '54'
[Sample 4/5] -> '83'
[Sample 5/5] -> '66'
83
ex (4.3) modify evaluate_math500_stream from 3.15 to test whether self-consistency improves the MATH-500 accuracy
of the base model with a sampling size of 3 and temperature and top-p of 0.9.
Implement a tiebreaking rule so that ties are resolved by the first appearing answer, so that if the results are 13, 15, 13, 15, 16
the chosen result will be 13.

ex (4.4) Implement an early-stopping version so that self-consistency ends sampling once more than half of the answers agree

In practice temperature values around 0.5 to 0.9 anf top-p around 0.7 and 0.8 are reasonable starting points

The long answer can be accessed with: print(results["full_answers"][0])

Now when we ask the model to explain step by step and generate longer responses with the following call
"""

results = self_consistency_vote(
    model,
    tokenizer,
    prompt + "\n\nExplain step by step.",
    device=device,
    num_samples=5,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    seed=123,
    show_progress=True,
)
"""
Out put this time with chain-of-thought 
[Sample 1/5] -> '83'
[Sample 2/5] -> '83'
[Sample 3/5] -> '83'
[Sample 4/5] -> '83'
[Sample 5/5] -> '3'
"""



