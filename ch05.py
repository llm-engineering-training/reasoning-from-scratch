""" start 07-15-2026 """
"""  
We will cover another inference-scaling technique, self-refinement
which focuses on iteratively refining a single answer to correct potential
mistakes. The model iteratively improves its own answers
In Chapter 4 we looked at chain-of-thought and self-consistency.
A downside of self-consistency is that it requires short answers that can be compared
Before implemting a self-refinement model we are first going to implement scoring functions
to compare and rank answers
The general approach will be
1 - Load pretrained LLM
2 - Build a simple rule-based score
3 - Compute token probabilities and log probabilities
4 - use the probabilities as part of the self-refinement method
"""
import math
import torch
import matplotlib.pyplot as plt
from ch02 import get_device
from ch03 import (
    load_model_and_tokenizer, render_prompt
)
from ch04 import (
    extract_final_candidate,
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache
)



"""
Loading the tokenizer and base model
(5.1)  
"""
device = get_device()
device = torch.device("cpu")

model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)

"""  
The following is to ensure that the model is loaded correctly
(5.2)
"""
raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)

raw_cot = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
    "Explain step by step."
)
#prompt_cot = prompt + "\n\nExplain step by step."
prompt_cot = render_prompt(raw_cot)


"""  
torch.manual_seed(0)
response_1 = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=False,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9, 
)

torch.manual_seed(3)
response_2 = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=False,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9, 
)
 """



"""
print("Response 1 characters:", len(response_1))
print("Response 1 tokens:", len(tokenizer.encode(response_1)))
print("\nResponse 2 characters:", len(response_2))
print("Response 2 tokens:", len(tokenizer.encode(response_2)))

 """

"""
The output from the test above is:
Response 1 characters: 495
Response 1 tokens: 220

Response 2 characters: 384
Response 2 tokens: 149  

Shorter responses are cheaper because they require fewer tokens to be generated and are therefore preferred
Our main focus will now turn to developing a simple rule-based scoring function to compare LLM responses
end 07-15-2026
"""

"""  
start 07-16-2026

We begin with a heuristic, rule-based scorer, which will provide a baseline for the probability-based
scorer we'll implement later. Note below is a scorer, not a verifier as in ch03. We're assuming the answer
is unknown
(5.3)
"""

def heuristic_score(
    answer,
    prompt=None,
    brevity_bonus=500,
    boxed_bonus=2.0,
    extract_bonus=1.0,
    fulltext_bonus=0.0,
):
    score =0.0
    """ Reward answers that have a final boxed answer """
    cand = extract_final_candidate(answer, fallback="none")
    if cand:
        score += boxed_bonus
        """ Give weaker reward if answer doesn't have a boxed value """
    else:
        cand = extract_final_candidate(answer, fallback="number_then_full")
        if cand:
            score += fulltext_bonus
    """ Add brevity reward that decays with text length """
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score
    
"""  
Plotting the brevity penalty score
For simplicity the brevity bonus is computed
using the number of characters rather than the number of tokens
(5.4)
"""
def plot_brevity_curve(brevity_bonus, max_len=2048):
    lengths = torch.arange(1, max_len)
    scores = 1.5 * torch.exp(-lengths / brevity_bonus)
    plt.figure(figsize=(4, 3))
    plt.plot(lengths, scores)
    plt.xlabel("Text length (number of characters)")
    plt.ylabel("Score contribution")
    plt.tight_layout()
    plt.savefig("brevity_curve.png")
    plt.show()
#plot_brevity_curve(500)

""" Now we apply the heuristic_socre to the long and short answers 
response_1 and response_2 from above 
ex (5.1) Extend the self-consistency implementation(self_consistency_vote) in (4.17) so that it handles
ties among candidate answers - This can be achieved by passing the ties to the hueristic_score method

ex (5.2) Modify the self-consistency implementation so that the final answer is chosen using the 
hueristic_score method rather than the majority voting. Generate N candidate answers for each problem (N>=2)
score each candidate using the hueristic_score method and select the one with the highest score
The hueristic_score method has several parameters based on intuition, but we should keep in mind the following
rules of thumb
1 - Increase the extraction related bonus is the scorer too often prefers final answers that are hard to parse
2 - Increase the brevity pressure if the scorer keeps favoring long answers and conversely if the scorer keeps
    prefering short candidates
"""

""" Uncomment responses above before uncommenting this section

print(round(heuristic_score(response_1), 3))
print(round(heuristic_score(response_2), 3))

 """
"""  
We now turn our attention t building a scorer based on the models own confidence -  where confidence is a
probability that the model assigns.
At each position the model will distribute the probability mass over the possible next tokens
If the tokens in a proposed answer consistenly receive high probability, then the answer is more 
compatible with the model's own internal preferences.
We'll not illustrate the concept of token probabilities - referred to in the literature as next-token probabilities, 
per-token probabilities, swquence likelihoods or more loosely token-level likelihoods.
We'll revisit the logits where we looked up the vocabulary entry associated to the highest score, in this
setting we want to use the scores to quantify the models confidence in a candidate answer - we're only inspecting the scores
Note: The model's confidence does not automatically imply correctness

Rather than working with raw logits we'll convert them to probabilities, which are comparable across inputs and form
the basis for the logprob scoring method. torch.softmax converts logits into probabilities - our interest here is simply the
probability assigned to specific tokens
Note: the probability of the last token is taken as a conditional probability, whereas for the entire sequence it is a joint 
probability
We are not interested in generating next text, instead this is retrospective scoring not a generation step
(5.5)
"""
@torch.inference_mode()
def calc_next_token_probas(model, tokenizer, prompt, device, show=True):
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    """ Get logits and probabilities similar to the text generation functions 
    logits runs the models forward pass to predict the next tokens
    where squeeze(0) removes the batch dimension to return to a 2D tensor
    and output shape is (sequence_length, vocabulary_size)
    """
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    """ convert unnormalized model scores into actual probability distributions """
    all_probas = torch.softmax(logits, dim=1)
    """ Positions we score here:all - creates a sequence of indices to align prompts with targets"""
    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    """ Since we have the text, we know the true next tokens - isolates the true targets the  model is supposed to predict
    we use slicing to shift the entire prompt forward by one token
    """
    next_ids = token_ids[1:]
    """ Get probabilites for each next token 
    extracts the specific probabilities assigned to the correct words - uses advanced indexing
    to pull values from all_probas, matches each prompt position t_idx, with the actual next token ID next_ids
    """
    next_token_probas = all_probas[t_idx, next_ids]
    """ Likelihood of the sequence is the product of the probabilitity scores """
    prod_next_token_probas = torch.prod(next_token_probas)
    if show:
        print("Next-token probabilities:", next_token_probas)
        print("Joint Probability:", prod_next_token_probas)
    else:
        return next_token_probas, prod_next_token_probas

""" 

torch.set_printoptions(precision=4, sci_mode=True)
calc_next_token_probas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Berlin"
)

 """

""" end 07-17-2026 """

""" start 07-28-2026 
Output from the above is:
Next-token probabilities: tensor([5.4598e-05, 4.3164e-01, 1.5747e-02, 7.5000e-01, 1.7969e-01],
       dtype=torch.bfloat16)
Joint Probability: tensor(5.0291e-08, dtype=torch.bfloat16)
The values are extremely small since we are multiplying probabilities
We'll move to working with log probabilities which will avoid the numerical issues
Note: Likelihood referes to how well a specific model explains observed data.
The token probabilities calculated above can be used as a scoring function to rank different responses.
"""

torch.set_printoptions(precision=4, sci_mode=False)
""" (5.6) """

def plots_soft_log():
    plt.figure(figsize=(9, 4))
    # Logits
    plt.subplot(1, 3, 1)
    logits = torch.linspace(-2, 2, steps=7)
    plt.bar(range(len(logits)), logits, color="C0", alpha=0.7)
    plt.title("Logits")
    plt.xlabel("Token index")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)

    # Softmax
    plt.subplot(1, 3, 2)
    probas = torch.softmax(logits, dim=-1)
    plt.bar(range(len(probas)), probas, color="C1", alpha=0.7)
    plt.title("torch.softmax(logits)")
    plt.xlabel("Token index")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    # Log-softmax
    plt.subplot(1, 3, 3)
    log_probas = torch.log_softmax(logits, dim=-1)
    plt.bar(range(len(log_probas)), log_probas, color="C2", alpha=0.7)
    plt.title("torch.log_softmax(logits)")
    plt.xlabel("Token index")
    plt.ylabel("Log-probability")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("logits_softmax_log_softmax.png")
    plt.show()

""" 
plots_soft_log()

(5.7)
 """

@torch.inference_mode()
def calc_next_token_logprobas(
    model, tokenizer, prompt, device, show=True
):
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    # We now use log_softmax
    all_logprobas = torch.log_softmax(logits, dim=-1)
    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:]
    next_token_logprobas = all_logprobas[t_idx, next_ids]
    # We replace the product with a sum
    sum_next_token_logprobas = torch.sum(next_token_logprobas)
    if show:
        print("Next-token log-probabilities:", next_token_logprobas)
        print("Joint log-probability:", sum_next_token_logprobas)
    else:
        return next_token_logprobas, sum_next_token_logprobas


#torch.set_printoptions(precision=4, sci_mode=True)

""" 
calc_next_token_logprobas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Berlin"
)

calc_next_token_logprobas(
    model, tokenizer, device=device,
    prompt="The capital of Germany is Bridge"
)

"""


"""
The followin is an illustration on how we can extract just the answer portion and
calculate an average log probability - also known as the per-token log likelihood 
output:
Next-token logprobas: tensor([    -0.4512,     -0.3418,     -8.2500,     -0.4258,     -3.7500,
            -2.9375,     -1.1641,      0.0000,     -0.0232,      0.0000,
            -0.0078,     -0.0679,     -0.1582], dtype=torch.bfloat16)
Joint log-probability: tensor(-17.6250, dtype=torch.bfloat16)
7
last seven tokens tensor([-1.1641,  0.0000, -0.0232,  0.0000, -0.0078, -0.0679, -0.1582],
       dtype=torch.bfloat16)
Mean of last seven tokens tensor(-0.2031, dtype=torch.bfloat16)
⚡ main ~/reasoning-from-scratch 
"""

""" 
example_prompt = "What is the capital of Germany?"
example_answer = " The capital of Germany is Berlin."
next_token_logprobas, sum_next_token_logprobas = calc_next_token_logprobas(
    model, tokenizer, device=device,
    prompt=example_prompt+example_answer, show=False
)

print("Next-token logprobas:", next_token_logprobas)
print("Joint log-probability:", sum_next_token_logprobas)

print(len(tokenizer.encode(example_answer)))
last_7 = next_token_logprobas[-7:]
print("last seven tokens", last_7)
print("Mean of last seven tokens", torch.mean(last_7))

 """


""" 
We are going to use the exercise above to define a new function
(5.8)
"""
@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=1)

    """ Index range for the positions corresponding to the answer tokens """
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    return torch.mean(next_token_logps)

""" 
score_1 = avg_logprob_answer(
    model, tokenizer,
    prompt="What is the capital of Germany?",
    answer=" The capital of Germany is Berlin.",
    device=device
)
print(score_1)

score_2 = avg_logprob_answer(
    model, tokenizer,
    prompt="What is the capital of Germany?",
    answer=" The capital of Germany is Bridge.",
    device=device
)
print(score_2)

 """
""" 

torch.manual_seed(0)
response_1 = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=False,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9, 
)
print("Response 1 characters:", len(response_1))
torch.manual_seed(3)
response_2 = generate_text_stream_concat_flex(
    model, tokenizer, prompt_cot, device,
    max_new_tokens=2048, verbose=False,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9,
    top_p=0.9, 
)
print("Response 2 characters:", len(response_2))
avg_res1 =avg_logprob_answer(
    model, tokenizer,
    prompt=prompt_cot,
    answer=response_1,
    device=device
)
print(avg_res1)
avg_res2 =avg_logprob_answer(
    model, tokenizer,
    prompt=prompt_cot,
    answer=response_2,
    device=device
)
print(avg_res2)

 """

""" end 07-28-2026 """

""" start 08-18-2026 """
""" 
We've spent a lot of time looking at the concept of next token probability and log scoring.
1 -  We'll use log-probability scoring when discussing self-refinement
2 - The concept of log probabilities will also be relevant when implementing reinforcement 
    learning with verifiable rewards training.
Note: logprob scoring is not a panacea. It is a useful tool for scoring signal amongst many

ex (5.3) - Use the logprb scorer and extend the self-consistency inplementation from listing (4.17)
so that it can handle ties among candidate answers. Then run the two implementations on a subset of 
the MATH-500 dataset to see which tiebreaking method performs better.




ex (5.4) Extend the self-consistency implementation so that the final answer is chosen using the
logprob scorer rather than the hueristic scorer or majority voting


Self-refinement through iterative feedback - self-refinement is the core inference-scaling technique we 
want to explore. It is a technique where the LLM analyzes and refines its own answers. The LLM starts with
an initial answer to a prompt, the it critiques the answer and refines it. It is in essence a sequential application
of the text generation function to different prompts and inputs. The following illustration will skip 
chain-of-thought prompting, however, in practice we combine the two when working with a base model
Steps
1 - prompt half the value of 3x - 9 is x + 37, what is the value of x?
2 - initial answer - \boxed{18}
3 - critique prompt
4 - LLM writes critiques of initial answer
5 - refine prompt
raw_prompt and prompt are defined on lines 48 and 52
"""

raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)

prompt = render_prompt(raw_prompt)

torch.manual_seed(123)

""" 
initial_response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7, top_p=0.9,
)
"""

#print("Initial response:", initial_response)


"""
LLM response is \boxed{18}  correct answer is 83 - uncomment above to see results
(5.10)
"""
def make_critique_prompt(raw_prompt, draft):
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\n"
        "Critique:"
    )

""" 
critique_prompt = make_critique_prompt(raw_prompt, initial_response)
torch.manual_seed(123)
critique = generate_text_stream_concat_flex(
    model, tokenizer, critique_prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7, top_p=0.9,
)

print("critique:", critique)
"""

"""
output:   The solution is incomplete. The equation is not properly solved.

Fix plan: 1. Start with the equation: $3x - 9 = 2(x + 37)$
2. Distribute 2 on the right side: $3x - 9 = 2x + 74$
3. Subtract $2x$ from both sides: $x - 9 = 74$
4. Add 9 to both sides: $x = 83$
5. Check the solution: $3(83) - 9 = 2(83 + 37)$critique:  The solution is incomplete. The equation is not properly solved.

Fix plan: 1. Start with the equation: $3x - 9 = 2(x + 37)$
2. Distribute 2 on the right side: $3x - 9 = 2x + 74$
3. Subtract $2x$ from both sides: $x - 9 = 74$
4. Add 9 to both sides: $x = 83$
5. Check the solution: $3(83) - 9 = 2(83 + 37)$
 \boxed{83}

Note: the critique contains a factual error, since the question does not contain a typo. The critique is still useful
even if parts of it are wrong. Also, to get the output above we had to switch from CPU to GPU. CPU did not return as detailed a response
(5.11)
"""
def make_refine_prompt(raw_prompt, draft, critique):
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revised answer:"
    )

""" 
refine_prompt = make_refine_prompt(raw_prompt, initial_response, critique)
torch.manual_seed(123)
revised_answer = generate_text_stream_concat_flex(
    model, tokenizer, refine_prompt, device,
    max_new_tokens=2048, verbose=True,
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.7, top_p=0.9,
)
print("Revised answer:", revised_answer)

 """

""" 
output: Answer:
 \boxed{83}Revised answer:  \boxed{83} -  uncomment above to see output

 We're now going to create a function for the self-refinement loop we illustrated above.
 The function will also support plugging in the scoring methods we created earlier to compute
 a score for each answer.
 Note: Scoring does not always improve results. Whether a scorer should be used in self-refinement
 depends on the LLM. You an determine this by experimenting on a benchmark dataset like
 MATH-500.
 In the function below, when score_fn=None, we always accept the most recent answer
 (5.12)
"""
def self_refinement_loop(
    model,
    tokenizer,
    raw_prompt,
    device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=None,
    prompt_renderer=render_prompt,
    prompt_suffix="",
    verbose=False,
    temperature=0.7,
    top_p=0.9,
):
    steps = []
    prompt = prompt_renderer(raw_prompt) + prompt_suffix
    """ initial response draft """
    current_full =  generate_text_stream_concat_flex(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_response_tokens,
        verbose=False,
        generate_func=generate_text_top_p_stream_cache,
        temperature=temperature,
        top_p=top_p,
    )
    current_extracted = extract_final_candidate(
        current_full, fallback="number_then_full"
    )
    if score_fn:
        current_score = score_fn(answer=current_full, prompt=prompt)
    else:
        current_score = 0.0
    """ Run for one or more iterations """
    for it in range(iterations):
        draft_before_full = current_full
        draft_before_extracted = current_extracted
        score_before = current_score
        critique_prompt = make_critique_prompt(
            raw_prompt, draft_before_full
        )
        critique_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=critique_prompt,
            device=device,
            max_new_tokens=max_critique_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )
        """ Refine the response """
        refine_prompt = make_refine_prompt(
            raw_prompt, draft_before_full, critique_full
        )
        revised_full = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=refine_prompt,
            device=device,
            max_new_tokens=max_response_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )
        revised_extracted = extract_final_candidate(
            revised_full, fallback="number_then_full"
        )
        if score_fn:
            revised_score = score_fn(
                answer=revised_full, prompt=prompt
            )
        else:
            revised_score = 0.0
        """ Log the result """
        step = {
            "iteration": it + 1,
            "draft_full": draft_before_full,
            "draft_extracted": draft_before_extracted,
            "critique": critique_full,
            "revised_full": revised_full,
            "revised_extracted": revised_extracted,
            "score_before": score_before,
            "score_after": revised_score,
        }
        steps.append(step)
        if verbose:
            print(
                f"[Refinement {it+1}/{iterations}]"
                f"\nCurrent: {draft_before_extracted}"
                f"\nRevised: {revised_extracted}"
                f"\nScore before: {score_before:.3f}"
                f"\nScore after: {revised_score:.3f}"
                f"\n{'=' * 25}"
            )
        """ Accept revised response if it's not worse """
        if revised_score >= current_score:
            current_full = revised_full
            current_extracted = revised_extracted
            current_score =  revised_score
        
    return {
        "final_full": current_full,
        "final_extracted": current_extracted,
        "steps": steps
    }

"""  
To use the avg_logprob_answer in the refinement loop we need to use Python's built-in partial function to specify
the remaining answers
(5.13)
"""
from functools import partial
avg_logprob_score = partial(
    avg_logprob_answer,
    model=model,
    tokenizer=tokenizer,
    device=device
)

torch.manual_seed(1)

""" 
results_logprob = self_refinement_loop(
    model=model,
    tokenizer=tokenizer,
    raw_prompt=raw_prompt,
    device=device,
    iterations=2,
    max_response_tokens=2048,
    max_critique_tokens=256,
    score_fn=avg_logprob_score,
    verbose=True,
    temperature=0.7,
    top_p=0.9,
)
"""


""" end 08-18-2026 
Output: 
[Refinement 1/2]
Current: 10
Revised: 83
Score before: -0.879
Score after: -0.367
=========================
[Refinement 2/2]
Current: 83
Revised: 83
Score before: -0.367
Score after: -1.359
=========================

Summary:
Created a rule-based scorer that ranks model output by rewarding extractable final answers and more
economical completions
We looked at next token scoring to quantify model confidence by converting logits into normalized
probabilities and combining these into sequence-level likelihood
Log probabilities replace raw probabilities to avoid numerical underflow
We walked through the self-refinement process and created a reusable self-refinement function capable
of using score-based acceptance to keep only revisions that do not degrade the computed answer
"""

""" start 08-18-2026 """