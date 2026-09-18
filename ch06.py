""" start 08-19-2026 """

""" 
Our focus will be on reinforcement learning (RL), the most commonly used training method for reasoning models
Chapter 5 focused on inference-time scaling, i.e., spending more resources during inference to improve accuracy
Training-time scaling, our current focus, invest additional computation during training
In practice the two are usually combined
RL focuses on how models learn from sequences of actions and their outcomes. RL is applied as a post-training stage
on top pf the retrained language model.
Two common RL stages for LLM: reasoning training, and preference training where preference tuning follows reasoning training
Reasoning training: train LLM to solve complex math or code problems
Preference tuning: train LLM on human preferences

RLHF - Reinforcement Learning with human feedback. Most popular method for implementing preference-tuing. Optimizes for how humans rank
and evaluate LLMs
RLVR - Reinforcement Learning with Verifiable rewards. 

RLHF has two main steps: 
    1 -  train a remard LLM Model -> The LLM generates multiple responses per prompt using the temprature scaling or top-p
        ask human annotators to rank the answers from best to worst - convert ranks to targets for for the model using statistical 
        preference model that maps pairwise comparisons to relative scores
    2 - use the reward LLM model to score the target LLM responses and fine-tune it
In RLVR the two-step RLHF pipeline collapses into a single training loop, because we use a deterministic verifier that checks answer
correctness. The correctness label is used as a reward signal within an RL objective to update the model
GRPO - Group Relative Policy Optimization, here "policy" is jargon that refers to the LLM we want to train.
RLVR determines what signal is available, GRPO determines how the signal is used to update the model weights

PPO - Proximal Policy Optimization - policy gradient algorithm for LLMs. GRPO is more resource friendly because it does not require a
separate value model to estimate the value function 

""" 
""" end 08-19-2026 pg189 """


""" start 08-20-2026 """
""" 
Kullback-Leibier (KL) loss term

High-level GRPO Approach

1 - Prompt - input for the LLM  
    Rollouts - We'll use temperature scaling and top-p filtering from Chapter 4 to get the LLM generate multiple responses
2 - Rewards - We compute a correctness and format reward using a math verifier from CHapter 3
3 - Advantages - We compute advantages from the list of reward values
4 - Logprobs - We compute the log probability of each rollout as in Chapter 5
5 - Policy gradient loss - We elementwise-multiply advantages and logprobs and average the resulting values to calculate the loss
6 - We calculate the total loss or loss gradient using the LLM we are training and a reerence LLM to calculate the Kullback-Leibier loss and combine with the
    the policy gradien loss in -5 -  to update the model weights


start 08-25-2026
As we progress, we'll refer back to page 190 for detailed diagram of GRPO process






 """
import torch
import json
import requests
import time
from pathlib import Path
from pprint import pprint


from ch02 import get_device
from ch03 import(
    load_model_and_tokenizer, 
    render_prompt,
    extract_final_candidate,
    grade_answer
)
from ch04 import(
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache,
    top_p_filter
)
from reasoning_from_scratch.qwen3 import KVCache

WHICH_MODEL = "base"

device = get_device()
model, tokenizer = load_model_and_tokenizer(
    which_model=WHICH_MODEL,
    device=device,
    use_compile=False
)

"""
(6.2)
The below is meant to ensure that we can load the model and run it without issues
Output: 47 47 (which is incorrect. Correct answer is \boxed{83})
"""
raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
prompt = render_prompt(raw_prompt)
torch.manual_seed(0)

""" response = generate_text_stream_concat_flex(
    model, tokenizer, prompt, device,
    max_new_tokens=2048, verbose=True, 
    generate_func=generate_text_top_p_stream_cache,
    temperature=0.9, top_p=0.9
)
print(response) """

"""  
Our training data set will be the Mathematics Aptitude Test of Heurstics (MATH) minus the the MATH-500 test set we've been
so far. The next function will load the entire MATH dataset
(6.3)
Output from run using print/pprint is:
Dataset size: 12000
{'answer': '6',
 'level': 'Level 3',
 'problem': 'Sam is hired for a 20-day period. On days that he works, he earns '
            '$\\$$60. For each day that he does not work, $\\$$30 is '
            'subtracted from his earnings. At the end of the 20-day period, he '
            'received $\\$$660. How many days did he not work?',
 'solution': 'Call $x$ the number of days Sam works and $y$ the number of days '
             'he does not. We can set up the following system of equations to '
             'represent the given information: \\begin{align*}\n'
             'x+y &= 20 \\\\\n'
             '60x - 30y &= 660 \\\\\n'
             '\\end{align*} The first equation represents the total number of '
             'days Sam works, and the second equation represents his total '
             'profit. Solving for $x$ in the first equation yields $x = 20 - '
             'y$. Substituting into the second equation gives $60(20-y) - 30y '
             '= 660$. Canceling a factor of $10$ and multiplying out gives '
             '$120 - 6y - 3y = 66$. This simplifies to $-9y = -54$, or $y = '
             '6$. Thus, Sam did not work for $\\boxed{6}$ days.',
 'type': 'Algebra',
 'unique_id': 4}

 We're are only going to use the 'answer' key from training purposes. Using the 'solution' would constrain the model
 and risk overfitting to a specific solution and style.
 Now we are going to implement the GRPO stages
"""
def load_math_train(local_path="math_train.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "math_full_minus_math500/refs/heads/main/"
        "math_full_minus_math500.json"
    )
    backup_url = (
        "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
        "MATH/math_full_minus_math500.json"
    )
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            print("Using backup URL")
            r = requests.get(backup_url, timeout=30)
            r.raise_for_status()
        data = r.json()
        if save_copy:
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    return data

math_train = load_math_train()
""" print("Dataset size:", len(math_train))
pprint(math_train[4]) """

"""  
The generate_text_stream_concat_flex we created in ch04 used the @inference_mode, disabling PyTorch features for efficiency.
However, in the current context we'll later perform a backward pass, where PyTorch computes the gradients from the loss so that
the optimizer can update the model weights. With @torch.no_grad decorator, we disable gradient tracking during the foward pass
without switching PyTorch into inference-only mode
(6.4)
This new iteration now returns the token ids of the prompt plus the answer tokens, the number of tokens and the answer text
"""
@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9
    ):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
    )
    """ Cache past keys and values for efficient generation as discussed in Chapter 2 """
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        """ Apply temperature scaling from chapter 4 """
        if temperature and temperature != 1.0:
            logits =  logits / temperature
        probas = torch.softmax(logits, dim=1)
        """ Apply top-p filter from chapter 4 """
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(
            probas.cpu(), num_samples=1
        ).to(device)
        token_id = next_token.item()
        generated.append(token_id)
        if (
            tokenizer.eos_token_id is not None 
            and token_id == tokenizer.eos_token_id
        ):
            break
        logits = model(next_token, cache=cache)[:, -1]
    full_token_ids = torch.cat(
        [input_ids,
        torch.tensor(generated, device=device, dtype=input_ids.dtype)]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


torch.manual_seed(5)
token_ids, prompt_len, answer_text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=512,
            temperature=0.9,
            top_p=0.9,
        )
""" 
print(answer_text)

"""

""" GRPO - stage 1 Rollouts """
rollouts = [
    r"\boxed{83}",
    r"The correct answer is \boxed{83}",
    r"The final answer is 83",
    r"We get \boxed{38}",
]

""" end 08-26-2026 pg199 """

""" start 08-26-2026 
We do not want to call sample_response multiple times to generate different rollouts, so we created the "rollouts" list above
Out focus below is in calculating rewards for each rollout 
We'll calculate the rewards using the math verifier from ch03
(6.6)
"""
def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)

""" (6.7) 
Output:
Answer: '\\boxed{83}'
Reward: 1.0

Answer: 'The correct answer is \\boxed{83}'
Reward: 1.0

Answer: 'The final answer is 83'
Reward: 0.0

Answer: 'We get \\boxed{38}'
Reward: 0.0
The DeepSeek-R-1 team tried to train with process rewards, this was unsuccessful -
better to train only on final answer correctness without intermediate rewards - see ch06_ex for Exercise 6.1
"""
rollout_rewards = []

for answer in rollouts:
    reward = reward_rlvr(answer_text=answer, ground_truth="83")
    #print(f"Answer: {answer!r}")
    #print(f"Reward: {reward}\n")
    rollout_rewards.append(reward)

""" Rewards tell us how well each rollout is. Advantage captures how a rollout performed relative to other rollouts 
GRPO stage three
advantage =  (r(i) - m) / (sigma + epsilon)
r(i) =  reward for the i-th rollout
m = mean reward for the rollouts
sigma = rollout rewards standard deviation
epsilon = small constant added for numerical stability - avoid division-by-zero
(6.8)
Output:
tensor([ 0.8659,  0.8659, -0.8659, -0.8659])
Captures which responses performed better
At a practical level, the advantage values directly scale the gradients during the policy update
If the advantage is positive the gradient increases the likelihood of the actions that produced
the rollout
"""
rewards = torch.tensor(rollout_rewards, device=device)
advantages =  (rewards - rewards.mean()) / (rewards.std() + 1e-4)
#print(advantages)

""" Scoring rollouts with sequence log probabilities 
The logprobs measure how likely the model considers each generated token to 
be given current model parameters - GRPO Stage four
The logprobs and the advantages form the core ingredients of the GRPO policy gradient loss
(6.9)
"""
def avg_logprob_answer(model, tokenizer, prompt, answer, device):
    prompt_ids = tokenizer.encode(prompt)
    #print(f"Prompt IDs: {prompt_ids!r}")
    answer_ids = tokenizer.encode(answer)
    #print(f"Answer IDs: {answer_ids!r}")
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    #print(f"Model Logits size: {logits.size()!r}")
    logprobs = torch.log_softmax(logits, dim=1)
    #print(f"Logprobs size: {logprobs.size()!r}")

    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=device)
    #print(f"T_IDX size: {t_idx.size()!r}")
    next_tokens = full_ids[start + 1 : end + 1]
    #print(f"Next_tokens size: {next_tokens.size()!r}")
    next_token_logps = logprobs[t_idx, next_tokens]
    return torch.mean(next_token_logps).item()


answer_text = "The correct answer is \boxed{83}"
""" 
avg_logprob_val = avg_logprob_answer(model, tokenizer, prompt, answer=answer_text, device=device)
print(avg_logprob_val)
 """

""" end 08-27-2026 pg205 """

""" start 08-27-2026  
In GRPO we do not use averaged token-level logprobs we use sequence-level logprobs
because GRPO assigns a single reward and advantage to each rollout which applies to the entire
generated response - captures likelihood of the model generating that sequence - this is computed
by summing the logprobs of all generated tokens. We can do this by multiplying avg_logprob_val by 
the number of answer tokens as long as the token count matches exactly.
"""

""" sequence_logprob_val = avg_logprob_val * (
    len(tokenizer.encode(answer_text))
)
print(sequence_logprob_val) """

""" 
(6.10)
"""
def sequence_logprob_draft(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=1)
    """ Positions whose next token probabilities we want to predict """
    start = prompt_len - 1
    end = token_ids.shape[0] - 1 #.shape returns a tuple of integers .shape[n], indexes into the nth-tuple to isolate one-dimension
    t_idx = torch.arange(start, end, device=token_ids.device)
    #print(f"T_IDX size: {t_idx.size()!r}")
    next_tokens = token_ids[start + 1 : end + 1]
    #print(f"Next_tokens size: {next_tokens.size()!r}")
    next_token_logps = logprobs[t_idx, next_tokens]
    return torch.sum(next_token_logps)
""" We're using token_ids and prompt_len from the result from sample_response in (6.4)  
in the call below.
Ouput: tensor(-15.0963, device='cuda:0', grad_fn=<SumBackward0>)

"""

""" print(sequence_logprob_draft(model, token_ids, prompt_len)) """

"""  
PyToch builds a computational graph that recors each differentiable operation applied to a tensor hence
grad_fn=<SumBackward0> in the output above...which means gradients can propagate back through the sequence-level
log probability to the model parameters
We can learn more about PyTorch computational graph by reading the following
https://sebastianraschka.com/teaching/pytorch-1h/#3-seeing-models-as-computation-graphs.
We can simplify the draft in 6.10 as follows
(6.11)
Output: 
tensor(-15.0963, device='cuda:0', grad_fn=<SumBackward0>)
The logprobs can be computed directly in the sample_response function, which would improve efficienct
since we wouldn't have to make two function calls
"""

def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(
        1, token_ids[1:].unsqueeze(-1)
    ).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])

""" 
print(sequence_logprob(model, token_ids, prompt_len))
 """


""" 
Using the rollouts defined in stage - 1 above we can compute sequence-level log probabilities
(6.12)
Output:
Answer:  \boxed{83}
Logprob: -8.0994

Answer:  The correct answer is \boxed{83}
Logprob: -19.8819

Answer:  The final answer is 83
Logprob: -16.3864

Answer:  We get \boxed{38}
Logprob: -23.1266

Shorter more concise answers receive higher sequence-level (less negative) logprobs...this observation is consistent
with their role in GRPO when rewards and advantages are applied at the sequence level
"""

""" 
rollout_logps = []
for text in rollouts:
    token_ids = tokenizer.encode(prompt + " " + text)
    logprob = sequence_logprob(
        model=model,
        token_ids=torch.tensor(token_ids, device=device),
        prompt_len=prompt_len,
    )
    #print(f"Answer:  {text}")
    #print(f"Logprob: {logprob.item():.4f}\n")
    rollout_logps.append(logprob)

"""

"""  
From Advantage to policy updates via the GRPO loss - Stage - 5
We combine the previously computes advantages and logprobs into a policy gradient loss
First we convert rollout_logps into PyTorch tensors, then we compute the gradient loss by
multiplying each rollout's sequence-level logprob by its corresponding advantage
(6.13)
"""
#logps = torch.stack(rollout_logps)
""" 
We need the .detach() below because we want to treat the advantages as fixed learning signals.
This way we ensure that we only backprop through the logprobs
We need the negative sign because PyTorch optimizers minimize by default
"""
#pg_loss =-(advantages.detach() * logps).mean()
#print(logps)
#print(pg_loss)

""" end 08-28-2026 pg212 """

""" start 09-16-2026  """

"""
(6.14) 

"""
def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])
    was_training = model.training
    model.eval()
    for _ in range(num_rollouts):
        """ Stage 1 generate the rollouts """
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        """ Stage 2.1 compute the rewards """
        reward = reward_rlvr(text, example["answer"])
        """ Stage 4.1 compute logprobs """
        logp = sequence_logprob(model, token_ids, prompt_len)
        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text":text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len
            }
        )
    if was_training:
        model.train()
    """ Stage 2.2 collect all rewards """
    rewards = torch.tensor(roll_rewards, device=device)
    """ Stage 3 compute advantages """
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    """ Stage 4.2 collect all logprobs """
    logps = torch.stack(roll_logps)
    """ Stage 5 compute policy gradient loss """
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss #In chapter 7 we'll add the KL term here
    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages":advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss
    }
""" 
For math problems GRPO performs better when the KL loss term is omitted 
(6.15)
"""

""" 
torch.manual_seed(123)
stats = compute_grpo_loss(
    model=model,
    tokenizer=tokenizer,
    example=math_train[4],
    device=device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9
)
print(stats)
"""

""" 
Output: (short)
'loss': -0.0, 'pg_loss': -0.0, 'rewards': [0.0, 0.0], 'advantages': [0.0, 0.0],
'gen_len': 4, 'gen_len': 256}

Model answers incorrectly. Correct answer must include \boxed{6}
So we have zero rewards and zero advantages...if we were training the model, the gradient would be zero
and the model parameters would not be updated.
We now have all the piees required to implement a full GRPO training loop using RLVR (Reinforcement Learning with Verifiable Rewards)
The training loop will consist of eight main stages
1 - initialize pretrained model and optimizer
2 - For each training step 
    3 - reset loss gradients from previous step
    4 - GRPO stages to calculate loss
    5 - Backward pass to calculate loss gradient
    6 -  Update model weights using loss gradients
    7 - Print reward, response length and losses
    8 -  save a model checkpoint
The structure above follows a standard deep learning training loop. 
Key difference, is instead of standard supervised objetive, the loss is obtained via GRPO stages
(6.16)
"""
def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,

):
    if steps is None:
        steps = len(math_data)
    """ Stage 1 Initialize optimizer (Model already initialized outside of the function) """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0
    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)

    try:
        """ Stage 2 iterate over the training loop """
        for step in range(steps):
            """ Stage 3 reset loss gradient (best practive to do at start of the loop) """
            optimizer.zero_grad()
            current_step = step + 1
            example = math_data[step % len(math_data)]
            """ Stage 4 Calculate GRPO loss """
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            """ Stage 5 backward pass to calculate the loss gradient """
            stats["loss_tensor"].backward()
            """ Clip large gradients to improve training stability """
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            """ Stage 6 Update model weights using loss gradients """
            optimizer.step()
            """ Stage 7 collect rewards, response lengths and losses """
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(
                sample["gen_len"] for sample in stats["samples"]
            )
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path, current_step, steps,
                stats["loss"], reward_avg, avg_response_len
            )
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )
            """ Stage 8 save model checkpoint """
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step
                )
                print(f"Saved checkpoint to {ckpt_path}")

    except KeyboardInterrupt:
        """  """
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt"
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model
    return model


def save_checkpoint(
    model, checkpoint_dir, step, suffix=""
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path

""" Utility function to save the results to a CSV file for record keeping and to visualize results in chapter 7 """
def append_csv_metrics(
    csv_log_path,
    step_idx,
    total_steps,
    loss,
    reward_avg,
    avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )

""" The functions above implement the full Reinforcement Learning with Verifiable Rewards (RLVR) loop
using GRPO. The structure mirrors a conventional PyTorch training loop where we: 
reset gradients, 
backpropagate a loss,
opeionally clip gradients,
update parameters via an optimizer step,
log metrics,
save checkpoints
For a general introduction to training neural networks in PyTorch see 
https://sebastianraschka.com/teaching/pytorch-1h/
"""

""" end 09-17-2026 pg218 """

""" start 09-17-2026  """

"""  
Some interesting facts to note:
The temperature and top_p are set within common range to encourage diversity
The number of steps, 50, is relatively small 
(lr)-learning rate, can be tweaked but is in a reasonable range
num_rollouts=4 & max_new_tokens=512 are relatively small to reduce resource requirements
(6.17)
"""

""" 
device = get_device()
model.to(device)

torch.manual_seed(0)

train_rlvr_grpo(
    model=model,
    tokenizer=tokenizer,
    math_data=math_train,
    device=device,
    steps=50,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=5,
    checkpoint_dir=".",
    csv_log_path="train_rlvr_grpo_metrics.csv",
)
 """

"""
Output:
[Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.2
[Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.8
[Step 3/50] loss=9.1689 reward_avg=0.500 avg_resp_len=56.0
[Step 4/50] loss=5.3334 reward_avg=0.500 avg_resp_len=106.2
[Step 5/50] loss=-0.8887 reward_avg=0.750 avg_resp_len=303.2
Saved checkpoint to qwen3-0.6B-rlvr-grpo-step00005.pth
** We ran out of memory with the T4 GPU, for actual work we'll need to upgrade to a larger machine
For full results we upgraded to a NVidia L4 machine
Analysis

Both the loss and reward values fluctuate which is normal in RL training
Ideally, we want to see the following trend over time
1 - The average reward should increase when averaged over many steps
    as the model learns to produce more accurate responses
2 - The reasoning accuracy should improve (We'll evaluate this next)

In the event we have access to multiple GPUs we could use the code in the folowing link
to inplement batched training
https://github.com/rasbt/reasoning-from-scratch/tree/main/ch06/02_rlvr_grpo_scripts_intro

GRPO is resource-intensive 

We were able to run all 50 steps with 4-rollouts
We want to load and evaluate the saved model checkpoints


The saved checkpoints can be loaded using the PyTorch state dict like:
model.load_state_dict(torch.load("qwen3-0.6B-rlvr-grpo-step00050.pth"))

To evalute any of the checkpoints we could use 
https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500.py
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base \
--checkpoint_path checkpoints/qwen3-0.6B-rlvr-grpo-step00050.pth


"""




""" end 09-18-2026 pg """

""" start 09-18-2026  """