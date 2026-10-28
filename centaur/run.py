import json
import time
import argparse
from unsloth import FastLanguageModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from rotator import BanditArmsRotator
from bandits import (
    stationary_MAB,
    drifting_MAB,
    time_delayed_MAB,
    bernoulli_MAB,
    patch_foraging_MAB,
    two_context_MAB,
    three_context_MAB,
    linear_contextual_MAB,
)

from mab import probabilistic_MAB, contextual_MAB, lincon_MAB
from config.constants import N_GAMES


def run_experiment(model, tokenizer, pipe, config):
    results = []

    #### RUN LINEAR CONTEXTUAL BANDITS ####
    if config["bandit"] == "linear_contextual":
        for th in config["thetas"]:
            bandit = eval(config["bandit"] + "_MAB()")
            bandit.set_narms(config["narms"])
            bandit.set_theta(th)
            ## remove rotation for now
            for h in config["hints"]:
                for t in range(N_GAMES * 5):
                    choice, armFts, rewards, probs, regret = lincon_MAB(
                        model,
                        tokenizer,
                        h,
                        bandit,
                        config["temperature"],
                        config["ntrials"],
                    )
                    results.append(
                        {
                            "type": config["bandit"],
                            "thetas": bandit.theta,
                            "hint": h,
                            "ogHint": h,
                            "choice": choice,
                            "armFts": armFts,
                            "rewards": rewards,
                            "probs": probs,
                            "regret": regret,
                            "nTrials": len(choice),
                        }
                    )
        return pd.DataFrame(results)

    #### RUN OTHER BANDITS ####
    for a in config["arm_means"]:
        rotator = BanditArmsRotator(a, config["hints"])
        while rotator.current_index < config["narms"]:
            means, rhints = rotator.next()
            for ogh, h in zip(config["hints"], rhints):
                bandit = eval(config["bandit"] + "_MAB()")
                bandit.set_narms(config["narms"])
                bandit.set_stds(config["arm_stds"])
                bandit.set_means(means)
                for t in range(N_GAMES):
                    if bandit.type == "two_context" or bandit.type == "three_context":
                        choice, arms, probs, contexts, regret = contextual_MAB(
                            model,
                            tokenizer,
                            config["additional_instruction"],
                            h,
                            bandit,
                            config["temperature"],
                            config["ntrials"],
                        )
                        results.append(
                            {
                                "type": config["bandit"],
                                "ogArms": a,
                                "ogHints": ogh,
                                "arms": arms,
                                "hint": h,
                                "regret": regret,
                                "choice": choice,
                                "contexts": contexts,
                                "probs": probs,
                                "nTrials": len(choice),
                            }
                        )
                    else:
                        choice, arms, probs, regret = probabilistic_MAB(
                            model,
                            tokenizer,
                            h,
                            bandit,
                            config["temperature"],
                            config["ntrials"],
                            hint_timing=config.get("hint_timing", 0),
                        )
                        results.append(
                            {
                                "type": config["bandit"],
                                "ogArms": a,
                                "ogHints": ogh,
                                "arms": arms,
                                "hint": h,
                                "hint_timing": config.get("hint_timing", 0),
                                "regret": regret,
                                "choice": choice,
                                "probs": probs,
                                "nTrials": len(choice),
                            }
                        )
    df = pd.DataFrame(results)
    df["id"] = df.index  # index by game
    return df


def load_model(config):
    model_name = config.get("model_name", "eys8549/centaur_IBT")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        force_download=True,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=0,
        do_sample=True,
        temperature=config["temperature"],
        max_new_tokens=1,
    )
    return model, tokenizer, pipe


def run_exp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="config/stationary.json")
    parser.add_argument("--results-dir", type=str, default="results/")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    model, tokenizer, pipe = load_model(config)

    results_df = run_experiment(model, tokenizer, pipe, config)
    mode = config["mode"]
    results_path = (
        args.results_dir
        + config["model_nickname"]
        + "_"
        + config["bandit"]
        + f"_{time.strftime('%Y%m%d-%H%M%S')}.parquet"
    )
    results_df.to_parquet(
        results_path,
        engine="pyarrow",
        index=False,
    )
    print(results_df)


if __name__ == "__main__":
    run_exp()
