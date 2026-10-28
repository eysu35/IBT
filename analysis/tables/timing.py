import numpy as np, pandas as pd
import common as C

ONSETS = [0, 4, 8, 12]


def main():
    hid = C.hint_id_map()
    hl = C.hint_labels().set_index("hint_id")
    hpost = C.human_exp2_posterior()
    hcsv = pd.read_csv(C.HUMAN_EXP2_CSV)
    hcsv["hint_timing"] = hcsv["hint_timing"].astype(int)
    hcsv = hcsv.set_index(["type", "ogHint", "hint_timing"])
    rows = []
    for a, f in C.TRIALS_EXP2.items():
        tr = f()
        for t in C.R4.TYPE_ORDER:
            trt = tr[tr.type == t]
            human_nh_regret = -hpost["nohint"][t]
            hints = [h for h in trt.ogHint.unique() if h != C.NO_HINT]
            for on in ONSETS:
                g = C.games_post_onset(trt, on)
                nh = g[g.ogHint == C.NO_HINT]
                nh_x = nh["avg_regret"].to_numpy()
                for h in hints:
                    key = (t, C.R4._norm(h))
                    hid_ = hid.get(key) or hid.get((t, h))
                    assert hid_, (a, t, h)
                    gx = g[(g.ogHint == h) & (g.hint_timing == on)]
                    x = gx["avg_regret"].to_numpy()
                    if len(x) == 0:
                        continue
                    rng = C.rng_for("timing", a, t, h, on)
                    bx = C.boot_mean(x, rng)
                    bnh = C.boot_mean(nh_x, rng)
                    boot = bnh - bx
                    hn, _ = C.pair_draws(human_nh_regret, bx, rng)
                    hum = hn - bx
                    if a == "people":  # exact per-cell mixed-model summary
                        prow = hcsv.loc[(t, h, on)]
                        own_pt, own_lo, own_hi, own_se = (
                            float(prow.delta),
                            float(prow.delta_lo),
                            float(prow.delta_hi),
                            float(prow.se),
                        )
                        hum_pt, hum_lo, hum_hi = own_pt, own_lo, own_hi
                    else:
                        e = C.exp2_contrast(gx, nh, on, label=f"{a} {t} {h} {on}")
                        own_pt, own_lo, own_hi, own_se = (
                            float(e.delta),
                            float(e.delta_lo),
                            float(e.delta_hi),
                            float(e.se),
                        )
                        hum_pt, (hum_lo, hum_hi) = float(np.mean(hum)), C.ci(hum)
                    rows.append(
                        dict(
                            agent=a,
                            task_type=t,
                            hint_id=hid_,
                            hint_text_model=h,
                            hint_text=C.R4._norm(h),
                            short_label=hl.loc[hid_, "short_label"],
                            hint_type=hl.loc[hid_, "hint_type"],
                            onset=on,
                            n_games=len(x),
                            n_games_nohint=len(nh_x),
                            n_trials=20,
                            n_post_onset_trials=20 - on,
                            mean_regret=float(x.mean()),
                            mean_regret_nohint=float(nh_x.mean()),
                            rr_own_base=own_pt,
                            rr_own_lo=own_lo,
                            rr_own_hi=own_hi,
                            rr_own_se=own_se,
                            rr_own_boot=float(np.mean(boot)),
                            rr_own_boot_lo=C.ci(boot)[0],
                            rr_own_boot_hi=C.ci(boot)[1],
                            rr_human_base=hum_pt,
                            rr_human_lo=hum_lo,
                            rr_human_hi=hum_hi,
                            human_nohint_regret=float(np.mean(human_nh_regret)),
                            human_nohint_lo=C.ci(human_nh_regret)[0],
                            human_nohint_hi=C.ci(human_nh_regret)[1],
                        )
                    )
    df = (
        pd.DataFrame(rows)
        .sort_values(["agent", "task_type", "hint_id", "onset"])
        .reset_index(drop=True)
    )
    assert len(df) == 3 * 2 * 5 * 4, len(df)
    assert df.isna().sum().sum() == 0
    C.write_table(
        df,
        "timing.csv",
        dict(
            human_posterior=hpost["source"],
            window="post-onset",
            covariate="pre_regret (onset > 0)",
        ),
    )
    # exit check: people vs the Exp 2 CSV
    hcsv = pd.read_csv(C.HUMAN_EXP2_CSV)
    pp = df[(df.agent == "people")].merge(
        hcsv.rename(
            columns={"type": "task_type", "ogHint": "hint_text", "hint_timing": "onset"}
        ),
        on=["task_type", "hint_text", "onset"],
        how="inner",
    )
    print(
        f"people vs Exp 2 CSV ({len(pp)} rows): max |delta diff| = {(pp.rr_own_base - pp.delta).abs().max():.4f}"
    )
    mm = df[df.agent != "people"]
    print(
        f"models: max |OLS/HC3 (adjusted) - unadjusted bootstrap| point estimate = {(mm.rr_own_base - mm.rr_own_boot).abs().max():.3f}"
    )


if __name__ == "__main__":
    main()
