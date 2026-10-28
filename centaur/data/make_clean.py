import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"

MODEL_PREFIXES = {
    "centaur_IBT": "centaur_lora",  # raw files still named centaur_lora_*
    "centaur": "centaur",
    "llama": "llama",
}
ALL_MODELS = list(MODEL_PREFIXES)

_HINT_FIXES = {
    "choose . When": "choose 1. When",
    "The points you get on the first pick of a button is the most representative of the button value.": "The points you get on the first pick of a button are the most representative of the button value.",
    "Don't stick with any one button for too long.": "Do not stick with any one button for too long.",
}


def _fix_hints(df):
    hint_col = (
        "ogHints"
        if "ogHints" in df.columns
        else ("ogHint" if "ogHint" in df.columns else None)
    )
    if hint_col is None:
        return df
    df = df.copy()
    for old, new in _HINT_FIXES.items():
        df[hint_col] = df[hint_col].astype(str).str.replace(old, new)
    return df


def explode_to_trials(df):
    game_type = df["type"].iloc[0]
    is_lincon = game_type == "linear_contextual"

    if "ogHint" in df.columns and "ogHints" not in df.columns:
        df = df.rename(columns={"ogHint": "ogHints"})

    df = df.copy()
    df["id"] = range(len(df))

    explode_cols = ["choice", "regret", "probs"]
    if is_lincon:
        explode_cols += ["armFts", "rewards"]
    else:
        explode_cols += ["arms"]
    if "contexts" in df.columns:
        explode_cols.append("contexts")

    explode_cols = [c for c in explode_cols if c in df.columns]
    scalar_cols = [c for c in df.columns if c not in explode_cols]

    rows = []
    for _, game in df.iterrows():
        n = int(game["nTrials"])
        base = {c: game[c] for c in scalar_cols}
        for t in range(n):
            row = dict(base)
            for col in explode_cols:
                val = game[col]
                if hasattr(val, "__len__") and not isinstance(val, str):
                    row[col] = val[t]
                else:
                    row[col] = val
            rows.append(row)

    exp = pd.DataFrame(rows)
    exp["trial_num"] = exp.groupby("id").cumcount()
    exp["choice"] = exp["choice"].astype(int) - 1

    if is_lincon:
        exp["arms"] = exp["rewards"].apply(lambda x: list(np.asarray(x, dtype=float)))
        exp["optimal"] = exp["arms"].apply(lambda x: int(np.argmax(x)))
    else:
        exp["optimal"] = exp["arms"].apply(lambda x: int(np.argmax(np.asarray(x))))
    exp["correct"] = (exp["choice"] == exp["optimal"]).astype(int)
    exp["regret"] = exp.groupby("id")["regret"].diff().fillna(exp["regret"])
    return exp


def _find_files(raw_dir, model):
    prefix = MODEL_PREFIXES[model]
    files = sorted(
        f for f in raw_dir.glob(f"{prefix}_*.parquet") if "_clean.parquet" not in f.name
    )
    # centaur_ prefix also matches centaur_lora_; exclude if model is centaur
    if model == "centaur":
        files = [f for f in files if not f.name.startswith("centaur_lora_")]
    return files


def process_model(raw_dir, model, output_dir):
    files = _find_files(raw_dir, model)
    if not files:
        raise FileNotFoundError(f"No {model}_*.parquet files in {raw_dir}")

    all_processed = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        df = _fix_hints(df)
        game_type = df["type"].iloc[0]
        n_games = len(df)
        exp = explode_to_trials(df)
        exp["_file"] = i
        all_processed.append(exp)
        print(f"  {game_type}: {n_games} games -> {len(exp)} trials ({f.name})")

    processed = pd.concat(all_processed, ignore_index=True)
    processed["id"] = processed.groupby(["_file", "id"]).ngroup()
    processed.drop(columns="_file", inplace=True)
    # drop columns that only exist for some types (causes mixed-type parquet errors)
    for col in ("armFts", "rewards", "thetas", "ogArms"):
        if col in processed.columns and processed[col].isna().any():
            processed.drop(columns=col, inplace=True)
    # normalize remaining list columns (ndarray → list, NaN → None)
    for col in processed.select_dtypes("object").columns:
        processed[col] = processed[col].apply(
            lambda x: list(x) if isinstance(x, np.ndarray) else x
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model}_clean.parquet"
    processed.to_parquet(out_path)
    print(f"[{model}] Saved {len(processed)} trial rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]
    for model in models:
        try:
            process_model(args.raw_dir, model, args.output_dir)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")


if __name__ == "__main__":
    main()
