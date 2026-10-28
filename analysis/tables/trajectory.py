import numpy as np, pandas as pd
import common as C

R3 = C.R3
LOADERS = {
    "people": R3.load_human,
    "bayesian": R3.load_bayesian,
    "centaur": lambda: R3.load_llm(R3.CENTAUR_PARQUET),
    "centaur_ibt": lambda: R3.load_llm(R3.CENTAUR_IBT_PARQUET),
    "llama": lambda: R3.load_llm(R3.LLAMA_PARQUET),
}


def main():
    hid = C.hint_id_map()
    rows = []
    for a, f in LOADERS.items():
        g = f()
        for (t, h), s in g.groupby(["type", "ogHint"]):
            if (t, h) not in hid:
                continue
            r = dict(
                agent=a, task_type=t, hint_id=hid[(t, h)], hint_text=h, n_games=len(s)
            )
            for m in R3.METRICS:
                r[m] = float(s[m].mean())
                r[m + "_sem"] = float(s[m].std(ddof=1) / np.sqrt(len(s)))
            rows.append(r)
    df = (
        pd.DataFrame(rows)
        .sort_values(["agent", "task_type", "hint_id"])
        .reset_index(drop=True)
    )
    for a in LOADERS:
        assert (df.agent == a).sum() == 48, (a, (df.agent == a).sum())
    assert df.isna().sum().sum() == 0
    C.write_table(
        df,
        "trajectory.csv",
        dict(metrics=R3.METRICS, source="analysis/result3_finegrained.py"),
    )


if __name__ == "__main__":
    main()
