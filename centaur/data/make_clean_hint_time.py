from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
HT_DIR = SCRIPT_DIR / "hint_time"

_HINT_FIXES = {
    "choose . When": "choose 1. When",
    "The points you get on the first pick of a button is the most representative of the button value.": "The points you get on the first pick of a button are the most representative of the button value.",
}


def _fix_hints(df):
    col = (
        "ogHints"
        if "ogHints" in df.columns
        else ("ogHint" if "ogHint" in df.columns else None)
    )
    if col is None:
        return df
    df = df.copy()
    for old, new in _HINT_FIXES.items():
        df[col] = df[col].astype(str).str.replace(old, new)
    return df


def explode_to_trials(df):
    df = df.copy()
    explode_cols = ["choice", "regret", "probs", "arms"]
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
    exp["optimal"] = exp["arms"].apply(lambda x: int(np.argmax(np.asarray(x))))
    exp["correct"] = (exp["choice"] == exp["optimal"]).astype(int)
    exp["regret"] = exp.groupby("id")["regret"].diff().fillna(exp["regret"])
    return exp


def main():
    files = sorted(HT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {HT_DIR}")

    all_processed = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        df = _fix_hints(df)
        df["id"] = range(len(df))
        game_type = df["type"].iloc[0]
        ht = df["hint_timing"].iloc[0]
        exp = explode_to_trials(df)
        exp["_file"] = i
        all_processed.append(exp)
        print(f"  {game_type} ht={ht}: {len(df)} games -> {len(exp)} trials ({f.name})")

    processed = pd.concat(all_processed, ignore_index=True)
    processed["id"] = processed.groupby(["_file", "id"]).ngroup()
    processed.drop(columns="_file", inplace=True)

    for col in ("ogArms",):
        if col in processed.columns and processed[col].isna().any():
            processed.drop(columns=col, inplace=True)
    for col in processed.select_dtypes("object").columns:
        processed[col] = processed[col].apply(
            lambda x: list(x) if isinstance(x, np.ndarray) else x
        )

    out_path = SCRIPT_DIR / "centaur_hint_time_clean.parquet"
    processed.to_parquet(out_path)
    print(f"\nSaved {len(processed)} trial rows to {out_path}")
    print(f"  Types: {sorted(processed['type'].unique())}")
    print(f"  Hint timings: {sorted(processed['hint_timing'].unique())}")
    print(f"  Games: {processed['id'].nunique()}")


if __name__ == "__main__":
    main()
