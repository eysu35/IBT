import numpy as np, pandas as pd
import common as C


def main():
    hid = C.hint_id_map()
    hl = C.hint_labels().set_index("hint_id")
    hpost = C.human_exp1_posterior()
    hcsv = pd.read_csv(C.HUMAN_EXP1_CSV)
    lmm = {
        "bayesian": pd.read_csv(C.R2.BAYES_EFFECTS_CSV),
        "centaur": pd.read_csv(C.R2.CENTAUR_EFFECTS_CSV),
    }
    games = {a: f() for a, f in C.GAMES_EXP1.items()}
    rows = []
    for a in C.AGENTS:
        g = games[a]
        for t in C.TYPE_ORDER:
            gt = g[g.type == t]
            nh = gt[gt.ogHint == C.NO_HINT]["avg_regret"].to_numpy()
            human_nh_regret = -hpost["nohint"][t]  # draws of human no-hint mean regret
            for h in C.hint_order(t):
                x = gt[gt.ogHint == h]["avg_regret"].to_numpy()
                assert len(x) > 0, (a, t, h)
                r = dict(
                    agent=a,
                    task_type=t,
                    hint_id=hid[(t, h)],
                    hint_text=h,
                    short_label=hl.loc[hid[(t, h)], "short_label"],
                    hint_type=hl.loc[hid[(t, h)], "hint_type"],
                    n_games=len(x),
                    mean_regret=float(x.mean()),
                    mean_regret_nohint=float(nh.mean()),
                )
                if a == "people":
                    if h == C.NO_HINT:
                        own = np.zeros(8000)
                    else:
                        own = hpost["delta"][(t, h)]
                    hum = own
                    row = hcsv[(hcsv.type == t) & (hcsv.ogHint == h)].iloc[0]
                    r.update(
                        rr_own_lmm=row.delta,
                        rr_own_lmm_lo=row.delta_lo,
                        rr_own_lmm_hi=row.delta_hi,
                        rr_own_se=(row.se if h != C.NO_HINT else 0.0),
                        rr_own_boot=0.0,
                        rr_own_boot_lo=0.0,
                        rr_own_boot_hi=0.0,
                    )
                else:
                    rng = C.rng_for(a, t, h)
                    bx = C.boot_mean(x, rng)
                    bnh = C.boot_mean(nh, rng)
                    own = bnh - bx if h != C.NO_HINT else np.zeros_like(bx)
                    hn, _ = C.pair_draws(human_nh_regret, bx, rng)
                    hum = hn - bx
                    row = lmm[a][(lmm[a].type == t) & (lmm[a].ogHint == h)]
                    assert len(row), (a, t, h)
                    row = row.iloc[0]
                    r.update(
                        rr_own_lmm=row.delta,
                        rr_own_lmm_lo=row.delta_lo,
                        rr_own_lmm_hi=row.delta_hi,
                        rr_own_se=row.se,
                    )
                if (
                    a == "people"
                ):  # exact Fig. 4 LMM summary (posterior mean and 2.5/97.5 %)
                    pt, lo, hi = (
                        (0.0, 0.0, 0.0)
                        if h == C.NO_HINT
                        else (
                            float(row.delta),
                            float(row.delta_lo),
                            float(row.delta_hi),
                        )
                    )
                    r.update(
                        rr_own_base=pt,
                        rr_own_lo=lo,
                        rr_own_hi=hi,
                        rr_human_base=pt,
                        rr_human_lo=lo,
                        rr_human_hi=hi,
                    )
                else:
                    r.update(
                        rr_own_base=float(row.delta),
                        rr_own_lo=float(row.delta_lo),
                        rr_own_hi=float(row.delta_hi),
                        rr_own_boot=float(np.mean(own)),
                        rr_own_boot_lo=C.ci(own)[0],
                        rr_own_boot_hi=C.ci(own)[1],
                        rr_human_base=float(np.mean(hum)),
                        rr_human_lo=C.ci(hum)[0],
                        rr_human_hi=C.ci(hum)[1],
                    )
                r.update(
                    human_nohint_regret=float(np.mean(human_nh_regret)),
                    human_nohint_lo=C.ci(human_nh_regret)[0],
                    human_nohint_hi=C.ci(human_nh_regret)[1],
                )
                rows.append(r)
    df = pd.DataFrame(rows)
    df["rr_own_se"] = df["rr_own_se"].fillna(0.0)  # no-hint rows
    assert len(df) == 3 * 48 and df.isna().sum().sum() == 0, df.isna().sum()
    C.write_table(df, "hint_effects.csv", dict(human_posterior=hpost["source"]))
    # exit check: people rows vs current Fig. 4 CSV
    pp = df[df.agent == "people"]
    d = (pp.rr_own_base - pp.rr_own_lmm).abs().max()
    print(f"people rr_own_base vs human CSV delta: max |diff| = {d:.4f}")
    mm = df[df.agent != "people"]
    print(
        f"models: max |OLS/HC3 - bootstrap| point estimate = {(mm.rr_own_base - mm.rr_own_boot).abs().max():.3f}"
    )


if __name__ == "__main__":
    main()
