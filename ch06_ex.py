
import torch
import platform
from reasoning_from_scratch.ch03 import (
    extract_final_candidate, grade_answer
)

""" Exercise 6.1 """
def reward_rlvr(answer_text, ground_truth):
    boxed = extract_final_candidate(answer_text, fallback="none")
    if boxed:
        correct = grade_answer(boxed, ground_truth)
        return 1.0 if correct else 0.0
    """ If no boxed answer is found, look for number """
    unboxed = extract_final_candidate(answer_text, fallback="number_then_full")
    if unboxed:
        correct = grade_answer(unboxed, ground_truth)
        return 0.5 if correct else 0.0
    return 0.0

""" Our goal above is to extend the function in Ch06.py to assign partial credit based on output format
Specifically, we want to assign 0.5 if the answer is correct but not boxed """

""" Exercise 6.2 """
rollout_rewards = [0., 0., 0., 0.]
rewards = torch.tensor(rollout_rewards)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
#print(advantages)

"""
Recall advantage = (r(i) - m) / (sigma + epsilon)
All rollouts are assigned the same reward
When all rewards in a group are the same, then the numerator for all rollouts is zero
If all rollouts are the same, there is no relative signal to tell the model which behavior
to reinforce or suppress
"""
print(platform.python_version())