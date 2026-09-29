""" start 09-18-2026  """
"""  
Our focus will be on: 
the metrics that are worth tracking, 
how to spot failure modes early
Why training can become unstable

GRPO can become unstable so we'll discuss practical GRPO extensions and 
fixes that are used in reasoning-model training
The examples discussed are based on actual experiments, but the results should be interpreted
with care. 
We're going to analyze the training results more closely and make several modifications on the
GRPO training code.

"""

from pathlib import Path
import requests
import csv
import matplotlib.pyplot as plt
import torch

from ch02 import get_device
from ch03 import(
    load_model_and_tokenizer, 
    render_prompt,
    #extract_final_candidate,
    #grade_answer
)

from ch06 import(
    sample_response,
    reward_rlvr,
    sequence_logprob
)

from reasoning_from_scratch.qwen3 import(
    download_qwen3_small,
    Qwen3Tokenizer,
    #Qwen3Model,
    #QWEN_CONFIG_06_B
)

""" 
Helper function to download relevant scripts from the supplementary material to avoid
repeating long code passages
(7.1)
"""

def download_from_github(rel_path, out=None):
    github_raw_base = (  # Base URL
        "https://raw.githubusercontent.com/rasbt/"
        "reasoning-from-scratch/refs/heads/main/"
    )

    rel_path = Path(rel_path)
    # Use URL file name as default output file name
    out = Path(out) if out is not None else Path(rel_path.name)

    # Skip download if file already exists locally
    if out.exists():
        size_kb = out.stat().st_size / 1e3
        print(f"{out}: {size_kb:.1f} KB (cached)")
        return out

    # Download file
    r = requests.get(github_raw_base + rel_path.as_posix())
    r.raise_for_status()

    out.write_bytes(r.content)
    size_kb = out.stat().st_size / 1e3
    print(f"{out}: {size_kb:.1f} KB")

#Log file for training run using 500 steps and 1024 max_new_tokens 
""" 
download_from_github(
    "ch07/02_logs/ch06_rlvr_grpo_original_no_kl_metrics.txt"
)
 """
# Same log file in CSV format to extract and plot data for further analysis
""" 
download_from_github(
    "ch07/02_logs/ch06_rlvr_grpo_original_no_kl_metrics.csv"
)
 """

""" 
Plotting function to inspect the training run
repeating long code passages
(7.2)
"""
def moving_average(values, window_fraction=0.25):
    window_size = max(1, int(window_fraction * len(values)))
    smoothed =[]

    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window_mean = sum(values[start_idx : i + 1]) / (i - start_idx + 1)
        smoothed.append(window_mean)
    return smoothed

def plot_grpo_metrics(csv_path, columns, save_as=None):
    data = {name: {"steps": [], "values": []} for name in columns}

    # Open and read CSV log file
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("step"):
                continue

            # Use the training step as the shared x-axis across all metrics
            step = int(row["step"])

            for name in columns:
                value_str = row.get(name)
                if value_str:
                    data[name]["steps"].append(step)
                    data[name]["values"].append(float(value_str))
    # Create a fixed grid so loss, rewards, response length, etc. can be shown side by side
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(6, 4))
    axes = axes.ravel()

    for i, name in enumerate(columns):
        steps = data[name]["steps"]
        values = data[name]["values"]

        # Skip metrics that are not present
        if not values:
            fig.delaxes(axes[i])
            continue

        # Evaluation accuracy as barplot because we don't have data for each step
        if name == "eval_acc":
            axes[i].bar(steps, values, width=20)
        else:
            axes[i].plot(steps, values, alpha=0.4)
            axes[i].plot(steps, moving_average(values))

        axes[i].set_ylabel(name)
    for j in (2, 3):
        if axes[j] in fig.axes:
            axes[j].set_xlabel("Step")

    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()

# Plot the GRPO training run

""" 
plot_grpo_metrics(
    "ch06_rlvr_grpo_original_no_kl_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"],
    save_as="4.png"
)
"""

"""  
Observations from the plot generated from the training runs
1 - Average response length should initially increase along with an improvement in accuracy, then level off
2 - Loss value is not very informative, just a sanity check. Large spikes observed halfway through the run are
    concerning
3 - Accuracy, measured by performance on external target task, initially improves but then begins to decline.

The evaluation accuracy was measured on the MATH-500 benchmark can be computed periodically
by adding the seeting --eval_on_checkpoint 500 when running the training script...generally not recommended
since it could significantly slow down training. We can calculate it separately using the evaluate_math500.py 
from chapter 3: As follows:
download_from_github(
    "ch03/02_math500-verifier-scripts/evaluate_math500.py"
)

uv run evaluate_math500.py \
  --dataset_size 500 \
  --checkpoint_path \
    "checkpoints/rlvr_grpo_original_no_kl/\
qwen3-0.6B-rlvr-grpo-step00050.pth"

If uv not available replace uv run with python

We are going to look at two more metrics whcih are 
 - rollout advantages computed in compute_grpo_loss in chapter 6
 - entropy of the generated sequence
 (7.3) Computing Advantage statistics
"""
def compute_advantage_stats(reward_list):
    rewards = torch.tensor(reward_list)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # These are the new statistics we add:
    adv_avg = advantages.mean().item()
    adv_std = advantages.std().item()

    return advantages, adv_avg, adv_std

""" 
adv, adv_avg, adv_std = compute_advantage_stats([1., 1., 0., 0.])
print(f"Advantages: {adv}")
print(f"Advantage mean = {adv_avg:.4f}, std = {adv_std:.4f}")
 """

"""  
Output:
Advantages: tensor([ 0.8659,  0.8659, -0.8659, -0.8659])
Advantage mean = 0.0000, std = 0.9998

Because of how we compute advantages, their mean is alwaus zero, it is mostly a sanity check
The standard deviation is more useful, values close to 1 (one) indicate a well-scaled gradient signal,
and are usually associated with stable updates.

Entropy measures how uncertain the model is when generating the next tokem. 
    - High entropy = probabilities are spread across many possible tokens-encourages exploration
    - Low entropy = most of the probability is concentrated on a single token, model is deterministic, or training collapse

LogProbs are calculated
torch.log(torch.softmax(logits))
torch.log_softmax(logits) -  more direct
Example
"""

# Toy logits
logits = torch.tensor([
    0.6667, -2.0000,  1.3333, -0.0000, -0.6667,  2.0000, -1.3333
])

""" 
logprobs = torch.log_softmax(logits, dim=-1)
print("All token logprobs:", logprobs)
# Index of the selected token
selected_token = torch.argmax(logprobs)

selected = logprobs[selected_token]
print("Selected token ID:", selected_token)
print("Selected token logprob:", selected)
 """

""" end 09-18-2026 pg237 """

""" start 09-21-2026  """

"""  
Output: the above Toy Logits will return:
All token logprobs: tensor([-2.0442, -4.7109, -1.3776, -2.7109, -3.3776, -0.7109, -4.0442])
Selected token ID: tensor(5)
Selected token logprob: tensor(-0.7109)
Note:

torch.log_softmax() - computes all logprobs,
torch.argmax() - returns the index (token Id) of the largest logprob (5 in this case)
logprobs[selected_token] - returns the value of that token ID (-0.719)

Entropy is closelt related to logprobs - it is calculated by multiplying each probability by its
logprob and summing the products - Entropy is widely used and an easy-to-intepret measure of uncertainty
in a probability distribution
- low entropy (~0-0.5) means one token dominates the distribution-model is almost deterministic
- Moderate entropy (~ 1-2) means probability is spread across a reasonably small set of tokens
- High entropy (approaching log*vocabulary size > 2) probabilities are spread across many tokens, model is behaving random
(7.4) - Calculating Entropy
"""

""" 
# Probabilities from logits
probs = torch.softmax(logits, dim=-1)
logprobs = torch.log_softmax(logits, dim=-1)

# Entropy from probabilities
entropy = torch.sum(-(probs * logprobs))

print("All token probabilities:", probs)
print("Entropy:", entropy)

 """
"""  
Output:
the above returns
All token probabilities: tensor([0.1295, 0.0090, 0.2522, 0.0665, 0.0341, 0.4912, 0.0175])
Entropy: tensor(1.3700)
Entropy indicates moderate level of uncertainty, where the the token with probability 0.49 clearly dominates

Note if we only have the logprobs we can convert to probabilities with
torch.exp(logprobs)

With the above, we are going to extend the sequence_logprob from chapter 6 to 
return both logprobs and entropy
(7.5)
"""
def sequence_logprob_and_entropy(model, token_ids, prompt_len):
    # Old: Code is identical to chapter 5
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    targets = token_ids[1:]
    selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)

    # Log-prob of the generated answer tokens (sum over answer steps)
    selected_answer_logprobs = selected[prompt_len - 1:]
    logp_all_steps = torch.sum(selected_answer_logprobs)

    """ (New Code) Calculate entropy """
    all_answer_logprobs = logprobs[:-1][prompt_len -1:]
    if all_answer_logprobs.numel() == 0:
        entropy_all_steps = logp_all_steps.new_tensor(0.0)
    else:
        """ Convert logprob to prob """
        all_answer_probs = torch.exp(all_answer_logprobs)
        """ elementwise p * log p """
        plogp = all_answer_probs * all_answer_logprobs
        """ sum over vocab -> entropy per step """
        step_entropy = torch.sum(plogp, dim=1)
        entropy_all_steps = torch.mean(step_entropy)
    return logp_all_steps, entropy_all_steps

"""Note:
sequence_logprob_and_entropy returns the average entropy across all answer tokens
the average entropy is the mean across all answer tokens.
We can now use entropy as a diagnostic for the model's generation behavior during training
by tracking the average entropy of the generated answer tokens
We expect entropy to gradually decrease as the model becomes more confident.
A sudden collapse to a very low entropy is a sign of unstable training
To analyze the quantities we discussed above we could
1 - update compute_grpo_loss to use sequence_logprob_and_entropy
2 - then print and log the entropy in train_rlvr_grpo, which calls compute_grpo_loss internally

The full modification can be downloaded using:
download_from_github(
    "ch07/03_rlvr_grpo_scripts_advanced/7_3_plus_tracking.py"
)
Training takes a long time so we can download the resulting log file using:
"""
""" 
download_from_github(
    "ch07/02_logs/7_3_plus_tracking_metrics.csv"
)
 """

""" 
plot_grpo_metrics(
    "7_3_plus_tracking_metrics.csv",
    columns=["reward_avg", "adv_avg", "adv_std", "entropy_avg"],
    save_as="9.png"
)

 """


""" 
Analyzing 9.png

- Average advantage stays at zero throughout training, as should be expected with GRPO style normalization
    This is a sanity check. If the advantage averages were to drift away from zero, it would indicate a bug
    or normalization issue
- After initially high std, the standard deviation gradually decreases and stabilizes - rollouts become more similar in quality
- As long as the advantage std remains nonzero and reasonably stable there is still a usable learning signal
- Early in the training entropy is low and fairly flat, indicating the model is behaving in a deterministic way. However, after
    roughly step 200, entropy increases quite noticeably

Computing clipped policy ratios
Measures how the current policy, that is the LLM being trained, has changed relative to an earlier version of itself
It compares sequence logprobs computed before an update step with those computed after the update.
-   this corresponds to comparing the logprobs from step 4 which are computed using the old weight parameters with
    logprobs produced by the updated model
(7.6) computing the policy gradient
"""

""" 
rewards = torch.tensor([1., 1., 0.,0.]) # compute rewards
logprobs = torch.tensor([-7.9243, -20.1546, -16.6130, -23.3677]) #compute sequence logprobs
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
pg_loss = -(advantages.detach() * logprobs).mean()
print("Policy gradient loss:", pg_loss)
 """

""" 
Original paper with mathematical foundation for clipped policy ratio can be found here:
https://arxiv.org/pdf/1707.06347
(7.7)
"""

logprobs = torch.tensor([-7.9243, -20.1546, -16.6130, -23.3677])
new_logps = logprobs
old_logps = torch.tensor([
    -10.9243,   # -7.9243
    -20.3546,   # -20.1546
    -14.6130,   # -16.6130
    -23.3677,   # -23.3677
])
log_ratio = new_logps - old_logps
ratio = torch.exp(log_ratio)
clip_eps = 10.0
clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

#print("Ratio:        ", ratio)
#print("Clipped ratio:", clipped_ratio)


"""
Output from the above:
Ratio:         tensor([20.0855,  1.2214,  0.1353,  1.0000])
Clipped ratio: tensor([11.0000,  1.2214,  0.1353,  1.0000]) 
We use the clipped_ratio limits how far the new policy can move away from the old one in a single update.
If the new model suddenly assigns a much higher or much lower probability to a rollout than the old model the
raw ratio will become very large or very small. Withoud clipping, this would scale the advantage term substantially
DeepSeek uses a clip_eps = 10, other RLHF using the PPO algorithm use 0.1, which is aggressive clipping - resulting in 
much smaller per-step policy changes. eps=epsilon
(7.8)
"""
""" 
rewards = torch.tensor([1., 1., 0., 0.])
logprobs = torch.tensor([-7.9243, -20.1546, -16.6130, -23.3677])
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
adv = advantages.detach() # Treat advantages as fixed learning signals (no backprop through rewards
unclipped = ratio * adv
clipped = clipped_ratio * adv
obj = torch.minimum(unclipped, clipped)
clipped_pg_loss = -torch.mean(obj)
policy_ratio = torch.mean(ratio)
print("Clipped policy gradient loss:", clipped_pg_loss)
print("Policy ratio:", policy_ratio)

 """

"""
Output
Ratio:         tensor([20.0855,  1.2214,  0.1353,  1.0000])
Clipped ratio: tensor([11.0000,  1.2214,  0.1353,  1.0000])  
The modified code applying the clipped ratio and policy loss computation from 7.7 and 7.8
has been reflected in a modified code that can be down loaded using:
download_from_github(
    "ch07/03_rlvr_grpo_scripts_advanced/7_4_plus_clip_ratio.py"
)
DeepSeek-R1 uses the following process:
- Make a copy of the current model as the reference policy
- sample a big rollout(8,192) pool using the reference model
- split the rollouts into minibatches
- for each minibatch
    a - compute old_logps under the reference model
    b - compute new_logps under the current model (changes after each minibatch update)
    c - update the current model
- update the reference model

For our case we apply the following step
1 Make a copy of the current model as the reference policy
2 Sample 8 rollouts using the reference policy
3 Compute
    old_logps under the reference model
    new_logps under the current model
    update the current model
4 Repeat 2 & 3 with different rollout - no minibatches
5 update the reference model
Log file for this approach is downloaded
"""
""" 
download_from_github(
    "ch07/02_logs/7_4_plus_clip_ratio_metrics.csv",
)
 """

"""  
plot_grpo_metrics(
    "7_4_plus_clip_ratio_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"],
    save_as="13_1.png"
)
 """
""" end 09-21-2026 pg249 """

""" start 09-22-2026  """

"""  
Training runs with clipped policy ratios is more stable.
What we've implemented this far is essentially REINFORCE with 
group-normalized advantaged. For additional information on the mathermatics underpinning 
REINFORCE see:
 - (notational intro) https://lilianweng.github.io/posts/2018-02-19-rl-overview/#key-concepts
 - (theory) https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
 KL - Kullback-Leibler Divergence - measures how much the current policy deviates from a reference polity, 
 typically, the original model at the start of training

 Clipped Policy Ratios limit how large individual update steps could be. The KL term is often used to control
 change over the training trajectory
 KL Loss term is computed by comparing logprobs and reference logprobs - sum the difference between the logprobs
- A small KL Loss is added to the polic gradient loss to compute the total loss - weight updates that
    greatly increase divergence from the reference model are penalized during training
We update the compute_grpo_loss from chapter 6 and rename it compute_grpo_loss_plus_ky
we download the complete script with:
download_from_github(
    "ch07/03_rlvr_grpo_scripts_advanced/7_5_plus_kl.py"
)
(7.9) Adding KL term to the GRPO loss computation (Explained)

1 - The script imported uses argparse.ArgumentParser (core built-in Python class - serves as a container for argument 
specifications and validation)
    on line 558 we define parser.add(
        --kl_coeff,
        type=float,
        default=0.001,
        help="KL penalty coefficient"
    )
    Then on line 596 (in 7.9 before def compute_grpo_loss_with_kl) after import copy
    if kl_coeff: (Make copy of original model and disable gradients so it doesn't update)
        ref_model = copy.deepcopy(model).to(device) 
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = false
    else:
        ref_model = None
def compute_grpo_loss_plus_kl(
    model,
    old_model,
    ref_model, #new: pass the reference model 
    tokenizer,
    example,
    device,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    clip_eps=10.0,
    kl_coeff=0.001,#new: specify KL strength
    skip_zero_adv=False,
):
#added throw error if ref_model and kl_coedd are None
#added additional defined list:  roll_old_logps, roll_ref_logps, roll_entropies, roll_token_ids, roll_prompt_lens

The remainder of the code is not exact since some of the changes in the 7_5_plus_kl.py
have not yet been discussed. We'll proceed for now and fill in the remaining changes as they are discussed.

We can download the results from training using the KL loss term using:
download_from_github(
    "ch07/02_logs/"
    "7_5_plus_kl_metrics.csv"
)

Plot the training results using:
plot_grpo_metrics(
    "7_5_plus_kl_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"],
    save_as="16.png"
)
Output:
See 16.png
    - Accuracy improves from 15% to 40% in first 50 steps but then model fails to make any correct answers.
    - At the same time the loss explodes and reward collapses to zero
    - Model fails to restrict response to allowed tokens
    Note: Why the above failures
    kl_loss = kl_coeff * mean(new_logps - ref_logps)
    - ref_logps are computed under the reference policy, which is the original model at the beginning of training
    The subtlety is that the responses are generated by the old rollout policy not the current policy after the update
    A more principled off-policy version would use an importance-sampling correction.
    Because the rollouts are sampled from the old policy this simple sampled surrogate is not an exact KL estimate
    for the current policy unless we either sample from the current policy or apply an implartance-sampling correction

    Also once the rewards collapse to zero the advantages are all zero so the policy gradient loss no longer contributes any gradient
    The KL term starts to dominate.

    KL loss term is standard in GRPO algorithm, however, several important works report models can train better without it. 

So far we've used an answer-correctness reward, which is sufficient in practice. It is also common to use one or more auxiliary rewards
such as format reward, that checks whether the generated answer follows certain formatting guidelines.

Specifically we'll focus on a format reward that encourages the model to use: <think> </think> - this encourages the model to 
separate its intermediate reasoning from the final answer 
(7.10) Illustrate <think> </think>
"""
device = get_device()

model, tokenizer_base = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)

#print(tokenizer_base.encode("<think>"))

""" 
Output:
[13708, 766, 29] - the tokenizer is breaking up "<think>" into several subword tokens - <think> is not part of the vocabulary

If we run the code:
for i in [13708, 766, 29]:
    print(tokenizer_base.decode([i]))
Output shows how the tokenizer is breaking up <think>
Output: 
<th
ink
>

Most LLM developers leave unused placeholder token IDs that can be used for specific purposes when fine-tuning a model
with:
tokenizer_base._tok.add_special_tokens(
    ["<tool_response>", "</tool_response>", "<think>", "</think>"]
)

 """

""" 
tokenizer_base._tok.add_special_tokens(
    ["<tool_response>", "</tool_response>", "<think>", "</think>"]
)
print(tokenizer_base.encode("<think>"))
print(tokenizer_base.encode("</think>"))
 """

""" 
Output from the above print:
[151667]
[151668]
We successfully added new tokens to the base model

(7.11) Loading the Qwen3 reasoning model instead of using the 'base' model as we did above
"""

""" 
download_qwen3_small(
    kind="reasoning", tokenizer_only=True, out_dir="qwen3"
)
tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
print(tokenizer.encode("<think>"))
print(tokenizer.encode("</think>"))
 """

"""  
Output:
[151667]
[151668]
This model supports <think></think>
(7.12) Implement format reward
"""

def reward_format(
    token_ids,
    prompt_len,
    start_think_id=151667,
    end_think_id=151668,
):
    try:
        gen = token_ids[prompt_len:].tolist()
        return float(
            gen.index(start_think_id) < gen.index(end_think_id)
        )
    except ValueError:
        return 0.0

tokenizer_path = Path("qwen3") / "tokenizer-reasoning.json"
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
prompt = "Calculate ..."
rollout = "Let's ... <think> ... </think> ..."
token_ids = tokenizer.encode(prompt + rollout)

""" 
result = reward_format(
    token_ids=torch.tensor(token_ids),
    prompt_len=len(tokenizer.encode(prompt))
)

print(result)
 """

"""
Output
1.0
(7.13) Update GRPO loss function with a format reward  
(after we added the ch06 imports)
"""

def compute_grpo_loss_plus_format_reward(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    format_reward_weight=1.0,
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
        rlvr_reward = reward_rlvr(text, example["answer"]) # renamed from reward to rlvr_reward
        """ Stage 4.1 compute logprobs """
        logp = sequence_logprob(model, token_ids, prompt_len)
        """ Format rewards modifies previous reward to consist of correctness and a format reward """
        format_reward = reward_format(token_ids, prompt_len)
        reward = rlvr_reward + format_reward_weight * format_reward #updated reward from renamed reward above
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


""" end 09-22-2026 pg261 """

""" start 09-28-2026  """

"""  
We are now going to modify render_prompt to direct the model to emit <think></think> tokens
(7.14)
"""
def render_prompt_with_think_tokens(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "When solving the problem, first write your reasoning inside <think> and </think> tags.\n"
        "Then write the final result on a new line in the exact format:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template

"""
In practice (7.11) through (7.14) is sufficient to train a reasoning model with an additional format reward.

Modified script can be downloaded using: 
download_from_github(
    "ch07/03_rlvr_grpo_scripts_advanced/7_6_plus_format_reward.py"
)
The script is configured to use the reasoning model instead of the base model.
The base model will perform poorly because it is unfamiliar with the <think></think> tags
We can run the above model using:
uv run 7_6_plus_format_reward.py \
    --steps 500  \
    --max_new_tokens 1024
Runing the experiment is resource intensive and expensive,
We'll save it for later when we rebuild a RLVR for our Sein
LLM. For now we'll download the results and proceed with the analysis
using:
download_from_github(
    "ch07/02_logs/7_6_plus_format_reward_metrics.csv"
)

We then plot the training results using:
plot_grpo_metrics(
    "7_6_plus_format_reward_metrics.csv",
    columns=["loss", "reward_avg", "avg_response_len", "eval_acc"],
    save_as="18.png"
)

In 18.png we notice that the accuracy drops sharply and then attempts to recover for 
the remainder of the training. This is probably due to the model receiving too much reward 
for the <think> tag. We can address this by reducing format reward from 1.0 to 0.1, or we can make it
conditional, so that it only applies for correct answers

Note that in the training script, 7_6_plus_format_reward.py, --kl_coeff is 0.0, which disables the KL
penalty and the kl_loss remains at 0.

To enable KL regularization with the same settings we should run the script as
uv run 7_6_plus_format_reward.py \
    --steps 500 \
    --max_new_tokens 1024 \
    --kl_coeff 0.001 \
    --inner_epochs 2

The more useful metrics to plot for the training run are
reward_avg, format_reward_avg, adv_std and entropy_avg
Using:
plot_grpo_metrics(
    "7_6_plus_format_reward_metrics.csv",
    columns=["reward_avg", "format_reward_avg", "adv_std", "entropy_avg"],
    save_as="19.png"
)
We notice:
- Overall reward is relatively constant
- Format average reward grows
- Entropy increases after 200 steps
GRPO improvements and changes to improve stability and performance:
- Zero gradient signal filtering (https://arxiv.org/abs/2503.14476)
    Skip prompts where all samples responses receive the same reward -  produce no useful advantage signal
- Active sampling (DAPO)
    Oversample or prioritize prompts that producea mix of correct & incorrect responses
    These provide stronger learning signal
- Token-level loss (DAPO)
    Compute loss over individual tokens rather than whole responses, which can reduce length-related bias
- No KL loss (DAPO and Dr. GRPO by Liu et al., 2025 - https://arxiv.org/abs/2503.20783)
    Omit KL penalty when it hurts stability or performance, especially in math reasoning settings
- Use higher Clip range (DAPO)
    Relaxe slipping threshold so the policy can make larger updates when the reward signal is reliable
- Truncated importance-sampling (https://fengyao.notion.site/off-policy-rl)
    Cap very large weights so a few off-policy samples cannot dominate the update
- No standard deviation normalization (Dr. GRPO)
    Normalize rewards relative to thr group mean but avoid dividing by the reward standard deviation
- KL tuning with domain-specific KL strengths; zero for math (https://arxiv.org/abs/2512.02556)
    Use different KL coefficients for different data types, including zero KL for math tasks.
-
-
For additional information and results for the improvements check out the supplementary materials at:
https://github.com/rasbt/reasoning-from-scratch/tree/b2e0841ee8be5fa3c17ad79c4056c55be26e74a3/ch07/03_rlvr_grpo_scripts_advanced

"""

























""" end 09-28-2026 pg2 """

""" start 09-28-2026  """

