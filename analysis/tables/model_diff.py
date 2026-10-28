import pandas as pd
import common as C


def main():
    hid = C.hint_id_map()
    hl = C.hint_labels().set_index("hint_id")
    hpost = C.human_exp1_posterior()
    hcsv = pd.read_csv(C.HUMAN_EXP1_CSV).set_index(["type", "ogHint"])
    games = {a: f() for a, f in C.GAMES_EXP1.items()}
    lmm = {
        a: pd.read_csv(csvp).set_index(["type", "ogHint"])
        for a, csvp in dict(
            bayesian=C.R2.BAYES_EFFECTS_CSV, centaur=C.R2.CENTAUR_EFFECTS_CSV
        ).items()
    }
    rows = []
    for t in C.TYPE_ORDER:
        human_nh_regret = -hpost["nohint"][t]
        draws = {}
        for a in ("bayesian", "centaur"):
            g = games[a][games[a].type == t]
            nh = g[g.ogHint == C.NO_HINT]["avg_regret"].to_numpy()
            for h in C.hint_order(t)[1:]:
                rng = C.rng_for(a, t, h)  # identical draws to hint_effects.py
                bx = C.boot_mean(g[g.ogHint == h]["avg_regret"].to_numpy(), rng)
                bnh = C.boot_mean(nh, rng)
                hn, _ = C.pair_draws(human_nh_regret, bx, rng)
                draws[(a, h)] = dict(own=bnh - bx, hum=hn - bx)
        for h in C.hint_order(t)[1:]:
            prow = hcsv.loc[(t, h)]
            b, c = draws[("bayesian", h)], draws[("centaur", h)]
            diff_boot = c["own"] - b["own"]
            diff_hb = c["hum"] - b["hum"]
            blo, bhi = C.ci(diff_boot)
            lo2, hi2 = C.ci(diff_hb)
            br, cr = lmm["bayesian"].loc[(t, h)], lmm["centaur"].loc[(t, h)]
            pm, bm, cm = float(prow.delta), float(br.delta), float(cr.delta)
            dm, lo, hi, dse = C.normal_diff(cm, float(cr.se), bm, float(br.se))
            rows.append(
                dict(
                    task_type=t,
                    hint_id=hid[(t, h)],
                    hint_type=hl.loc[hid[(t, h)], "hint_type"],
                    short_label=hl.loc[hid[(t, h)], "short_label"],
                    hint_text=h,
                    people=pm,
                    people_lo=float(prow.delta_lo),
                    people_hi=float(prow.delta_hi),
                    people_se=float(prow.se),
                    bayesian=bm,
                    bayesian_lo=float(br.delta_lo),
                    bayesian_hi=float(br.delta_hi),
                    bayesian_se=float(br.se),
                    centaur=cm,
                    centaur_lo=float(cr.delta_lo),
                    centaur_hi=float(cr.delta_hi),
                    centaur_se=float(cr.se),
                    diff=dm,
                    diff_lo=lo,
                    diff_hi=hi,
                    diff_se=dse,
                    differ=bool(lo > 0 or hi < 0),
                    diff_boot=float(diff_boot.mean()),
                    diff_boot_lo=blo,
                    diff_boot_hi=bhi,
                    differ_boot=bool(blo > 0 or bhi < 0),
                    closer="bayesian" if abs(bm - pm) <= abs(cm - pm) else "centaur",
                    abs_diff=abs(cm - bm),
                    bayesian_hb=float(b["hum"].mean()),
                    centaur_hb=float(c["hum"].mean()),
                    diff_hb=float(diff_hb.mean()),
                    diff_hb_lo=lo2,
                    diff_hb_hi=hi2,
                    differ_hb=bool(lo2 > 0 or hi2 < 0),
                )
            )
    df = pd.DataFrame(rows)
    assert len(df) == 40 and df.isna().sum().sum() == 0
    C.write_table(df, "model_diff.csv", dict(human_posterior=hpost["source"]))
    print(
        f"differ (own baseline): {int(df.differ.sum())}/40; closer: {df.closer.value_counts().to_dict()}; "
        f"differ (human baseline): {int(df.differ_hb.sum())}/40"
    )


if __name__ == "__main__":
    main()
