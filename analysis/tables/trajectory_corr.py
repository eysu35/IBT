import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import common as C

PAIRS = [("bayesian", "people"), ("centaur", "people"), ("centaur", "bayesian")]


def main():
    tr = pd.read_csv(C.TABLES / "trajectory.csv")
    wide = tr.pivot(
        index=["task_type", "hint_id"], columns="agent", values=C.R3.METRICS
    )
    rows = []
    for y, x in PAIRS:
        for m in C.R3.METRICS:
            xv, yv = wide[(m, x)].to_numpy(), wide[(m, y)].to_numpy()
            n = len(xv)
            rng = C.rng_for("corr", y, x, m)
            idx = rng.integers(0, n, size=(C.N_BOOT, n))
            bx, by = xv[idx], yv[idx]
            bx = bx - bx.mean(1, keepdims=True)
            by = by - by.mean(1, keepdims=True)
            br = (bx * by).sum(1) / np.sqrt((bx**2).sum(1) * (by**2).sum(1))
            rows.append(
                dict(
                    y=y,
                    x=x,
                    pair=f"{C.AGENT_LABEL[y]} vs {C.AGENT_LABEL[x]}",
                    metric=m,
                    pearson_r=float(pearsonr(xv, yv)[0]),
                    spearman_rho=float(spearmanr(xv, yv)[0]),
                    r_lo=C.ci(br)[0],
                    r_hi=C.ci(br)[1],
                    n=n,
                )
            )
    df = pd.DataFrame(rows)
    C.write_table(df, "trajectory_corr.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
