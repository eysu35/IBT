import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_bayesian import BANDITS, run_game, load_fitted_params, SIM_DIR, MODEL_NAME
from constants import HINT_LABELS
from hint_functions import HINT_TEMPLATE_TO_FUNCTION
from distributions import RandomNumberGenerator

OUT = SIM_DIR / "bayesian_timing_trials.parquet"
TIMINGS = [0, 4, 8, 12]
TYPES = ["stationary", "patch_foraging"]
N_TRIALS, N_ENVS = 20, 100  # 100 envs/condition to match the Centaur simulations


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="fixed hint strength (Exp 3); if omitted, load gamma from the fit",
    )
    p.add_argument("--model-comp", type=Path, default=None)
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument(
        "--use-temp",
        action="store_true",
        help="apply the fitted softmax temperature (Exp 1)",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or OUT
    if out.exists():
        print(f"Cached: {out}")
        return

    if args.gamma is not None:
        temp, hs = None, args.gamma
    else:
        temp, hs = load_fitted_params(args.model_comp, args.model_name, args.use_temp)
    print(f"Params: hs={hs:.2f}, temp={temp}")

    rng = RandomNumberGenerator(42)
    rows = []
    for btype in TYPES:
        hints = HINT_TEMPLATE_TO_FUNCTION.get(btype, {})
        es_list = [rng.randint(0, 2**32) for _ in range(N_ENVS)]
        as_list = [rng.randint(0, 2**32) for _ in range(N_ENVS)]
        for ht in TIMINGS:
            conditions = [("No hint.", None)] + list(hints.items())
            for text, fn in conditions:
                cat = ",".join(HINT_LABELS.get(text, [])) if fn is not None else "none"
                desc = f"{btype} ht={ht}/{getattr(fn, '__name__', 'none')}"
                for i, (es, aseed) in enumerate(
                    tqdm(list(zip(es_list, as_list)), desc=desc, leave=False)
                ):
                    bandit = BANDITS[btype](es % 1000)
                    for td in run_game(
                        bandit,
                        btype,
                        fn,
                        hs,
                        N_TRIALS,
                        aseed,
                        es,
                        hint_start=ht,
                        temp=temp,
                    ):
                        td.update(
                            type=btype,
                            ogHint=text,
                            hint_cat=cat,
                            hint_timing=ht,
                            sim_id=i,
                        )
                        rows.append(td)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
