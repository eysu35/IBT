import torch
import random
import pandas as pd
import numpy as np
from config.constants import (
    MAIN_TEXT,
    BERNOULLI_TEXT,
    LINCON_TEXT,
    QUESTION,
    SEED,
    CHOICES,
)
from utils import get_probs, compute_regret, gen_features

torch.cuda.manual_seed_all(SEED)


def probabilistic_MAB(
    model, tokenizer, hint, bandit, temp=1.0, ntrials=20, hint_timing=0
):
    base_text = BERNOULLI_TEXT if bandit.type == "bernoulli" else MAIN_TEXT
    if hint_timing == 0 or hint == "No hint.":
        instruction = base_text + "\n\nHint: " + hint + "\n"
    else:
        instruction = base_text + "\n"
    print(instruction)

    choice = []
    arms = []
    probs = []
    regret = []
    history_instruction = ""

    for step in range(ntrials):
        # inject hint at the specified trial
        if hint_timing > 0 and step == hint_timing and hint != "No hint.":
            history_instruction += f"Hint: {hint}\n"

        # query model
        input_text = instruction + history_instruction + QUESTION
        curr_probs = (
            get_probs(model, tokenizer, input_text, CHOICES, temp)[0].cpu().numpy()
        )
        n_probs = list(np.array(curr_probs) / sum(curr_probs))
        probs.append(n_probs)
        chosen_idx = np.random.choice(np.arange(bandit.narms), p=n_probs)
        curr_choice = chosen_idx + 1

        # save trial
        choice.append(curr_choice)
        arms.append(bandit.means.copy())
        reward = bandit.get_reward(chosen_idx)
        regret.append(compute_regret(choice, arms))
        trial_text = f"Trial {step}: choice: <<{curr_choice}>>, points: {reward:.2f}.\n"
        history_instruction += trial_text
        print(trial_text)

    return choice, arms, probs, regret


def contextual_MAB(model, tokenizer, instr, hint, bandit, temp=1.0, ntrials=40):
    instruction = MAIN_TEXT + instr + "\n\nHint: " + hint + "\n"
    print(instruction)

    choice = []
    contexts = []
    arms = []
    probs = []
    regret = []
    history_instruction = ""

    context_map = {0: "square", 1: "circle", 2: "triangle"}

    c = 0
    change_rate = 4
    for step in range(ntrials):
        if bandit.type == "two_context":
            if random.random() < 1 / change_rate:
                c = 1 - c
            input_text = (
                instruction
                + "You see a "
                + context_map[c]
                + ".\n"
                + history_instruction
                + QUESTION
            )

            # query model
            curr_probs = (
                get_probs(model, tokenizer, input_text, CHOICES, temp)[0].cpu().numpy()
            )
            n_probs = list(np.array(curr_probs) / sum(curr_probs))
            probs.append(n_probs)
            chosen_idx = np.random.choice(np.arange(bandit.narms), p=n_probs)
            curr_choice = chosen_idx + 1

            # save trial
            contexts.append(c)
            choice.append(curr_choice)
            if c == 0:
                arms.append(bandit.means.copy())
            elif c == 1:
                arms.append(bandit.rev_means.copy())

        elif bandit.type == "three_context":
            if random.random() < 1 / change_rate:
                c = random.choice(list(context_map.keys()))
            input_text = (
                instruction
                + "You see a "
                + context_map[c]
                + ".\n"
                + history_instruction
                + QUESTION
            )
            # query model
            curr_probs = (
                get_probs(model, tokenizer, input_text, CHOICES, temp)[0].cpu().numpy()
            )
            n_probs = list(np.array(curr_probs) / sum(curr_probs))
            probs.append(n_probs)
            chosen_idx = np.random.choice(np.arange(bandit.narms), p=n_probs)
            curr_choice = chosen_idx + 1

            # save trial
            contexts.append(c)
            choice.append(curr_choice)
            if c == 0:
                arms.append(bandit.s.copy())
            elif c == 1:
                arms.append(bandit.r.copy())
            elif c == 2:
                arms.append(bandit.rand.copy())

        reward = bandit.get_reward(chosen_idx, c)
        regret.append(compute_regret(choice, arms))
        trial_text = f"Trial {step}: shape: {context_map[c]}, choice: <<{curr_choice}>>, points: {reward:.2f}.\n"
        history_instruction += trial_text
        print(trial_text)

    return choice, arms, probs, contexts, regret


def lincon_MAB(model, tokenizer, hint, bandit, temp=1.0, ntrials=20):
    instruction = LINCON_TEXT + "\n\nHint: " + hint + "\n"
    print(instruction)

    choice = []
    armFts = []
    rewards = []
    probs = []
    regret = []
    history_instruction = ""

    for step in range(ntrials):
        feats, str_feats = gen_features(bandit)
        rwds = []
        for ft in feats:
            rwds.append(bandit.get_reward(ft))
        input_text = instruction + history_instruction + str_feats + QUESTION

        # query model
        curr_probs = (
            get_probs(model, tokenizer, input_text, CHOICES, temp)[0].cpu().numpy()
        )
        n_probs = list(np.array(curr_probs) / sum(curr_probs))
        probs.append(n_probs)
        chosen_idx = np.random.choice(np.arange(bandit.narms), p=n_probs)
        curr_choice = chosen_idx + 1

        # save trial
        choice.append(curr_choice)
        rewards.append(rwds)
        armFts.append(feats)
        reward = rwds[chosen_idx]
        regret.append(compute_regret(choice, rewards))

        trial_text = f"Trial {step}: features: {feats[chosen_idx]}, choice: <<{curr_choice}>>, points: {reward:.2f}.\n"
        history_instruction += trial_text
        print(trial_text)

    return choice, armFts, rewards, probs, regret


#### DEPRECATED - NOT USED ANYMORE ####


#     return hist, means_hist, probs_hist, regret_hist
