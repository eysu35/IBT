import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import common as C


def main():
    he = pd.read_csv(C.TABLES / "hint_effects.csv")
    he = he[he.hint_text != C.NO_HINT]
    wide = he.pivot(
        index=["task_type", "hint_id"], columns="agent", values="rr_own_base"
    )
    rows = []
    for y in ("bayesian", "centaur"):
        xv, yv = wide["people"].to_numpy(), wide[y].to_numpy()
        n = len(xv)
        rng = C.rng_for("hint_corr", y, "people")
        idx = rng.integers(0, n, size=(C.N_BOOT, n))
        bx, by = xv[idx], yv[idx]
        bx = bx - bx.mean(1, keepdims=True)
        by = by - by.mean(1, keepdims=True)
        br = (bx * by).sum(1) / np.sqrt((bx**2).sum(1) * (by**2).sum(1))
        rows.append(
            dict(
                y=y,
                x="people",
                pair=f"{C.AGENT_LABEL[y]} vs People",
                metric="hint_effect",
                pearson_r=float(pearsonr(xv, yv)[0]),
                spearman_rho=float(spearmanr(xv, yv)[0]),
                r_lo=C.ci(br)[0],
                r_hi=C.ci(br)[1],
                n=n,
            )
        )
    df = pd.DataFrame(rows)
    C.write_table(df, "hint_corr.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
