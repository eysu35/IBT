from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
sys.path.insert(0, str(_REPO_ROOT / "behavioral/hint_time/analysis"))
import freq_lmm as F
import hint_effects as S2  # behavioral/hint_time/analysis/hint_effects.py (Study 2 cells)
from constants import TYPE_ORDER

# ------------------------------------------------------------------ config
SMOKE = os.environ.get("SENS_SMOKE") == "1"
N_SIMS = 16 if SMOKE else 1000  # MC SE on power ~ .013 near .80 at 1000
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SENS_JOBS", 8)))
TARGET_POWER = 0.80
ALPHA = 0.05
Z_TWO_SIDED = float(stats.norm.ppf(1 - ALPHA / 2))  # 1.959964
# delta grid in units of total outcome SD (same grid as the earlier one-sided version)
DELTA_MULTIPLIERS = [
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.65,
    0.80,
    1.00,
    1.25,
]
if SMOKE:
    DELTA_MULTIPLIERS = [0.0, 0.3, 1.0]

NO_HINT = "No hint."

STUDY1_PARQUET = (
    _REPO_ROOT / "behavioral/instructed_bandits/data/behavioral_clean.parquet"
)
S1_HINT_CSV = _REPO_ROOT / "behavioral/instructed_bandits/analysis/hint_effects.csv"
S1_TASK_CSV = _REPO_ROOT / "analysis/results/task_effects.csv"
S2_HINT_CSV = _REPO_ROOT / "behavioral/hint_time/analysis/hint_effects.csv"
OUT_DIR = _REPO_ROOT / "analysis/results"
SUFFIX = "_smoke" if SMOKE else ""


def load_study1():
    df = F.normalize_regret(pd.read_parquet(STUDY1_PARQUET))
    g = (
        df.groupby(["user_id", "game_id", "type", "ogHint"])
        .agg(avg_regret=("regret", "mean"))
        .reset_index()
    )
    g["neg_regret"] = -g["avg_regret"]
    g["any_hint"] = (g["ogHint"] != NO_HINT).astype(int)
    return g


def load_study2_cells():
    return {k: v.reset_index(drop=True) for k, v in S2.load_cells().items()}


# ------------------------------------------------------------------ realised FDR thresholds
def realised_fdr_z(csv, p_col="p", padj_col="p_adj"):
    d = pd.read_csv(csv)
    rej = d[(d[padj_col] < ALPHA) & d[p_col].notna()]
    if rej.empty:
        return float("inf"), float("nan")
    p_max = float(rej[p_col].max())
    return float(stats.norm.ppf(1 - p_max / 2)), p_max


# ------------------------------------------------------------------ simulation workers
_W = {}  # per-process state: design arrays and model kind


def _init_worker(kind, hint_labels, group_codes, hinted, levels, cov=None, beta=0.0):
    _W.update(
        kind=kind,
        hint=np.asarray(hint_labels, dtype=object),
        groups=np.asarray(group_codes),
        hinted=np.asarray(hinted, float),
        levels=list(levels),
        cov=None if cov is None else np.asarray(cov, float),
        beta=float(beta),
    )
    _W["n_groups"] = int(np.max(group_codes)) + 1
    F.log.setLevel(logging.WARNING)


def _one_sim(args):
    delta, tau, sigma, seed = args
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, tau, _W["n_groups"])[_W["groups"]]
    e = rng.normal(0.0, sigma, len(_W["hinted"]))
    y = delta * _W["hinted"] + u + e
    if _W["cov"] is not None:
        y = y + _W["beta"] * _W["cov"]
    df = pd.DataFrame(
        {
            "neg_regret": y,
            "user_id": _W["groups"],
            "ogHint": _W["hint"],
            "any_hint": _W["hinted"],
        }
    )
    covs = None
    if _W["cov"] is not None:
        df["pre"] = _W["cov"]
        covs = ["pre"]
    try:
        if _W["kind"] == "binary":
            r = F.fit_binary_contrast(df, "any_hint", subject_col="user_id")
            return [r["z"]]
        eff = F.fit_hint_contrasts(
            df, hint_col="ogHint", subject_col="user_id", ref=NO_HINT, covariates=covs
        )
        z = eff.set_index("ogHint")["z"]
        return [float(z.get(l, np.nan)) for l in _W["levels"]]
    except Exception:
        return [np.nan] * (1 if _W["kind"] == "binary" else len(_W["levels"]))


def _seed_base(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def run_frame(name, g, kind, z_fdr, covariate=None):
    g = g.reset_index(drop=True)
    if covariate is not None:
        g = g.dropna(subset=[covariate]).reset_index(drop=True)
    codes, _ = pd.factorize(g["user_id"])
    hinted = (g["ogHint"] != NO_HINT).astype(float).to_numpy()
    levels = (
        [h for h in g["ogHint"].unique() if h != NO_HINT]
        if kind == "factor"
        else ["any_hint"]
    )
    if hinted.sum() == 0 or (1 - hinted).sum() == 0:
        print(f"  [skip] {name}: degenerate frame", flush=True)
        return [], pd.DataFrame()

    # variance components from the paper's fit of the real data
    if kind == "binary":
        real = F.fit_binary_contrast(g, "any_hint", subject_col="user_id", label=name)
        tau, sigma, method = real["sd_subject"], real["sd_resid"], real["method"]
    else:
        eff = F.fit_hint_contrasts(
            g,
            hint_col="ogHint",
            subject_col="user_id",
            ref=NO_HINT,
            label=name,
            covariates=[covariate] if covariate else None,
        )
        b = eff.attrs["baseline"]
        tau, sigma, method = b["sd_subject"], b["sd_resid"], b["method"]
    cov_c, beta = None, 0.0
    if covariate is not None:
        v = g[covariate].astype(float).to_numpy()
        cov_c, beta = v - v.mean(), eff.attrs["covariates"][covariate]
    if not np.isfinite(tau):
        print(
            f"  [warn] {name}: participant SD unavailable ({method}); simulating with tau=0",
            flush=True,
        )
        tau = 0.0
    sd_total = float(np.sqrt(tau**2 + sigma**2))
    deltas = [m * sd_total for m in DELTA_MULTIPLIERS]
    seed0 = _seed_base(name)

    t0 = time.time()
    curve_rows = []
    with Pool(
        N_JOBS,
        initializer=_init_worker,
        initargs=(kind, g["ogHint"].to_numpy(), codes, hinted, levels, cov_c, beta),
    ) as pool:
        for m, delta in zip(DELTA_MULTIPLIERS, deltas):
            args = [
                (delta, tau, sigma, seed0 + i) for i in range(N_SIMS)
            ]  # common random numbers
            Z = np.array(
                pool.map(_one_sim, args, chunksize=max(1, N_SIMS // (4 * N_JOBS))),
                float,
            )
            for j, lev in enumerate(levels):
                zj = Z[:, j]
                ok = np.isfinite(zj)
                curve_rows.append(
                    dict(
                        frame=name,
                        level=lev,
                        delta=delta,
                        delta_sd_units=m,
                        power_05=float(np.mean(np.abs(zj[ok]) > Z_TWO_SIDED))
                        if ok.any()
                        else np.nan,
                        power_fdr=float(np.mean(np.abs(zj[ok]) > z_fdr))
                        if ok.any()
                        else np.nan,
                        n_fail=int((~ok).sum()),
                        n_sims=N_SIMS,
                    )
                )
    curve = pd.DataFrame(curve_rows)

    n_subj = int(g["user_id"].nunique())
    summaries = []
    for lev in levels:
        c = curve[curve["level"] == lev]
        sub = g[g["ogHint"] == lev] if kind == "factor" else g[g["ogHint"] != NO_HINT]
        mde = mde_at_target(c["delta"], c["power_05"])
        mde_fdr = mde_at_target(c["delta"], c["power_fdr"])
        summaries.append(
            dict(
                frame=name,
                level=lev,
                n_subjects=n_subj,
                n_games=int(len(g)),
                n_hint_games=int(len(sub)),
                n_nohint_games=int((g["ogHint"] == NO_HINT).sum()),
                n_hint_subjects=int(sub["user_id"].nunique()),
                tau=tau,
                sigma=sigma,
                sd_total=sd_total,
                method=method,
                covariate=covariate or "",
                beta_cov=beta,
                mde_80=mde,
                mde_80_fdr=mde_fdr,
                z_fdr=z_fdr,
                power_at_null=float(
                    c.loc[c["delta_sd_units"] == 0.0, "power_05"].iloc[0]
                ),
                max_power=float(c["power_05"].max()),
                n_fail_total=int(c["n_fail"].sum()),
            )
        )
        print(
            f"  {name} | {lev[:45]:<45s} tau={tau:.2f} sigma={sigma:.2f} "
            f"MDE@80%={'%.2f' % mde if mde is not None else 'not reached'} "
            f"(FDR: {'%.2f' % mde_fdr if mde_fdr is not None else 'not reached'})",
            flush=True,
        )
    print(
        f"  [{name}] {len(levels)} level(s) x {len(deltas)} deltas x {N_SIMS} sims in {time.time() - t0:.0f}s",
        flush=True,
    )
    return summaries, curve


def mde_at_target(deltas, powers, target=TARGET_POWER):
    d = np.asarray(deltas, float)
    p = np.asarray(powers, float)
    o = np.argsort(d)
    d, p = d[o], p[o]
    for i in range(len(d)):
        if np.isnan(p[i]):
            continue
        if p[i] >= target:
            if i == 0:
                return float(d[0])
            d0, d1, p0, p1 = d[i - 1], d[i], p[i - 1], p[i]
            if not np.isfinite(p0) or p1 == p0:
                return float(d1)
            return float(d0 + (target - p0) / (p1 - p0) * (d1 - d0))
    return None


# ------------------------------------------------------------------ the three levels
def study1_per_task(g1, z_fdr):
    print(
        "\n=== (A) Study 1 task-level hint effect: neg_regret ~ any_hint + (1|user_id), per task ===",
        flush=True,
    )
    S, C = [], []
    for btype in TYPE_ORDER[:2] if SMOKE else TYPE_ORDER:
        s, c = run_frame(btype, g1[g1["type"] == btype], "binary", z_fdr)
        for r in s:
            r.update(type=btype)
        S += s
        C.append(c)
    return pd.DataFrame(S), pd.concat(C, ignore_index=True)


def study1_per_hint(g1, z_fdr):
    print(
        "\n=== (B) Study 1 per-hint effects: neg_regret ~ C(ogHint) + (1|user_id), per task ===",
        flush=True,
    )
    S, C = [], []
    for btype in TYPE_ORDER[:2] if SMOKE else TYPE_ORDER:
        s, c = run_frame(btype, g1[g1["type"] == btype], "factor", z_fdr)
        for r in s:
            r.update(type=btype, ogHint=r["level"])
        S += s
        C.append(c)
    return pd.DataFrame(S), pd.concat(C, ignore_index=True)


def study2_per_hint_timing(z_fdr):
    print(
        "\n=== (C) Study 2 per-hint x onset effects: neg_regret ~ C(ogHint) [+ pre_regret] + (1|user_id), "
        "per (type, onset), post-onset window ===",
        flush=True,
    )
    S, C = [], []
    cells = load_study2_cells()
    keys = list(cells)[:2] if SMOKE else list(cells)
    for btype, onset in keys:
        name = f"{btype} | onset {onset}"
        s, c = run_frame(
            name,
            cells[(btype, onset)],
            "factor",
            z_fdr,
            covariate=S2.PRE if int(onset) > 0 else None,
        )
        for r in s:
            r.update(type=btype, hint_timing=onset, ogHint=r["level"])
        S += s
        C.append(c)
    return pd.DataFrame(S), pd.concat(C, ignore_index=True)


# ------------------------------------------------------------------ observed effects + report
def attach_observed(summary, csv, keys, eff_col="delta"):
    obs = pd.read_csv(csv)
    if "agent" in obs.columns:  # task_effects.csv also holds the model rows
        obs = obs[obs["agent"] == "Human"]
    cols = keys + [eff_col, "se", "p", "p_adj", "stars"]
    obs = obs[[c for c in cols if c in obs.columns]].rename(
        columns={eff_col: "observed"}
    )
    for k in keys:
        if k in summary.columns:
            summary[k] = summary[k].astype(str)
            obs[k] = obs[k].astype(str)
    out = summary.merge(obs, on=keys, how="left")
    out["abs_observed"] = out["observed"].abs()
    out["ratio_obs_mde"] = out["abs_observed"] / out["mde_80"].astype(float)
    out["stars"] = out["stars"].fillna("")
    out["detectable"] = out["abs_observed"] >= out["mde_80"].astype(float)
    return out


def _fmt(x, nd=2):
    return (
        "not reached"
        if x is None or (isinstance(x, float) and not np.isfinite(x))
        else f"{float(x):.{nd}f}"
    )


def _mde_range(df, label):
    v = pd.to_numeric(df["mde_80"], errors="coerce").dropna()
    reached = f"{len(v)}/{len(df)} cells reached 80% within the grid"
    if len(v):
        return f"- **{label}**: MDE 80% power (p < .05) from {v.min():.2f} to {v.max():.2f} % of arm spread; {reached}."
    return f"- **{label}**: no cell reached 80% power within the grid."


def write_report(s1t, s1h, s2, fdr_info, runtime):
    L = [
        "# Sensitivity / power analysis (frequentist, two-sided)",
        "",
        f"Simulations per delta: {N_SIMS}; grid (SD units): {DELTA_MULTIPLIERS}; detection |z| > {Z_TWO_SIDED:.3f} "
        f"(two-sided p < .05). Runtime {runtime / 60:.0f} min on {N_JOBS} workers.",
        "",
        "Each cell is simulated with the model the paper fits (see module docstring); tau / sigma are the "
        "participant and residual SDs of that fit on the real data. Outcome: regret as % of the game's arm spread. "
        "MDE = smallest true effect (% of spread) "
        "detected with 80% power. `mde_80_fdr` uses the per-family Benjamini–Hochberg threshold the paper's "
        "FDR step realised in the observed data:",
        "",
    ]
    for fam, (z, pmax) in fdr_info.items():
        L.append(f"- {fam}: largest rejected raw p = {pmax:.4f} (|z| > {z:.2f})")
    L += [
        "",
        "## MDE range per inferential level",
        "",
        _mde_range(s1t, "Study 1 task-level (8 cells)"),
        _mde_range(s1h, "Study 1 per-hint (40 cells)"),
        _mde_range(
            s2,
            "Study 2 per-hint x onset, post-onset window with pre-onset covariate (40 cells)",
        ),
    ]

    def table(df, idcols, title):
        L.extend(
            [
                "",
                f"## {title}",
                "",
                "| "
                + " | ".join(
                    idcols
                    + [
                        "observed",
                        "SE",
                        "p_adj",
                        "stars",
                        "MDE_80",
                        "MDE_80_fdr",
                        "|obs|/MDE",
                        "tau",
                        "sigma",
                        "n_subj",
                        "n_hint/n_nohint",
                    ]
                )
                + " |",
                "|" + "---|" * (len(idcols) + 11),
            ]
        )
        for r in df.itertuples():
            ids = [str(getattr(r, c))[:60] for c in idcols]
            L.append(
                "| "
                + " | ".join(
                    ids
                    + [
                        f"{r.observed:.2f}" if np.isfinite(r.observed) else "",
                        f"{r.se:.2f}" if np.isfinite(r.se) else "",
                        f"{r.p_adj:.3f}" if np.isfinite(r.p_adj) else "",
                        r.stars,
                        _fmt(r.mde_80),
                        _fmt(r.mde_80_fdr),
                        f"{r.ratio_obs_mde:.2f}"
                        if np.isfinite(r.ratio_obs_mde)
                        else "",
                        f"{r.tau:.2f}",
                        f"{r.sigma:.2f}",
                        str(r.n_subjects),
                        f"{r.n_hint_games}/{r.n_nohint_games}",
                    ]
                )
                + " |"
            )

    sig_t = s1t[s1t["p_adj"] < ALPHA]
    table(
        sig_t,
        ["type"],
        f"Study 1 task-level effects significant after FDR ({len(sig_t)}): observed |effect| vs MDE",
    )
    sig_h = s1h[s1h["p_adj"] < ALPHA]
    table(
        sig_h,
        ["type", "ogHint"],
        f"Study 1 per-hint effects significant after FDR ({len(sig_h)}): observed |effect| vs MDE",
    )
    sw = s2[s2["p_adj"] < ALPHA]
    table(
        sw,
        ["type", "hint_timing", "ogHint"],
        f"Study 2 per-hint x onset effects significant after FDR ({len(sw)}): observed |effect| vs MDE",
    )

    L += ["", "## Significant effects smaller than their MDE (|observed| < MDE_80)", ""]
    small = pd.concat(
        [
            sig_t.assign(level_name="S1 task", ogHint="any hint", hint_timing=""),
            sig_h.assign(level_name="S1 hint", hint_timing=""),
            s2[(s2["p_adj"] < ALPHA)].assign(level_name="S2"),
        ],
        ignore_index=True,
    )
    small = small[~small["detectable"].fillna(False)]
    L += ["none" if small.empty else ""] + [
        f"- {r.level_name}: {r.type} / {str(r.ogHint)[:60]}{(' / onset ' + str(r.hint_timing)) if r.hint_timing else ''}: "
        f"|observed| {r.abs_observed:.2f} vs MDE {_fmt(r.mde_80)}"
        for r in small.itertuples()
    ]

    L += [
        "",
        "## Cells that did not reach 80% power within the grid (p < .05 criterion)",
        "",
    ]
    nr = pd.concat(
        [
            s1t.assign(level_name="S1 task", ogHint="any hint", hint_timing=""),
            s1h.assign(level_name="S1 hint", hint_timing=""),
            s2.assign(level_name="S2"),
        ],
        ignore_index=True,
    )
    nr = nr[pd.to_numeric(nr["mde_80"], errors="coerce").isna()]
    L += ["none" if nr.empty else ""] + [
        f"- {r.level_name}: {r.type} / {str(r.ogHint)[:60]}{(' / onset ' + str(r.hint_timing)) if r.hint_timing else ''}: "
        f"max power {r.max_power:.2f} at delta = {DELTA_MULTIPLIERS[-1]} SD ({r.sd_total * DELTA_MULTIPLIERS[-1]:.2f} % of spread)"
        for r in nr.itertuples()
    ]

    L += [
        "",
        "## tau (participant SD) and sigma (residual SD) per simulated frame",
        "",
        "| frame | tau | sigma | sd_total | n_subjects | n_games | fit |",
        "|---|---|---|---|---|---|---|",
    ]
    frames = pd.concat(
        [
            s1t.assign(frame=lambda d: "S1 task-level | " + d["type"]),
            s1h.assign(frame=lambda d: "S1 per-hint | " + d["type"]),
            s2.assign(frame=lambda d: "S2 | " + d["frame"]),
        ],
        ignore_index=True,
    )
    for r in frames.drop_duplicates("frame").itertuples():
        L.append(
            f"| {r.frame} | {r.tau:.2f} | {r.sigma:.2f} | {r.sd_total:.2f} | {r.n_subjects} | {r.n_games} | {r.method} |"
        )

    L += [
        "",
        "## Type-I error check: detection rate at delta = 0 (should be ~0.05)",
        "",
        f"- Study 1 task-level: {s1t['power_at_null'].min():.3f}–{s1t['power_at_null'].max():.3f}",
        f"- Study 1 per-hint: {s1h['power_at_null'].min():.3f}–{s1h['power_at_null'].max():.3f}",
        f"- Study 2: {s2['power_at_null'].min():.3f}–{s2['power_at_null'].max():.3f}",
        f"- simulated fits that failed: {int(s1t['n_fail_total'].sum() + s1h['n_fail_total'].sum() + s2['n_fail_total'].sum())}",
    ]
    (OUT_DIR / f"sensitivity_report{SUFFIX}.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(
        f"N_SIMS={N_SIMS}, N_JOBS={N_JOBS}, grid={DELTA_MULTIPLIERS}, two-sided z={Z_TWO_SIDED:.3f}"
        + (" [SMOKE]" if SMOKE else ""),
        flush=True,
    )

    fdr = {
        "Study 1 task-level (8)": realised_fdr_z(S1_TASK_CSV),
        "Study 1 per-hint (40)": realised_fdr_z(S1_HINT_CSV),
        "Study 2 per-hint x onset (40)": realised_fdr_z(S2_HINT_CSV),
    }
    for k, (z, p) in fdr.items():
        print(f"realised FDR threshold, {k}: p_max={p:.4f} -> |z|>{z:.3f}", flush=True)

    g1 = load_study1()
    print(
        f"Study 1: {len(g1)} games, {g1['user_id'].nunique()} participants", flush=True
    )

    s1t, c1t = study1_per_task(g1, fdr["Study 1 task-level (8)"][0])
    s1t = attach_observed(s1t, S1_TASK_CSV, ["type"], eff_col="advantage")
    s1t.to_csv(OUT_DIR / f"sensitivity_study1_per_task{SUFFIX}.csv", index=False)
    c1t.to_csv(OUT_DIR / f"sensitivity_study1_per_task_curves{SUFFIX}.csv", index=False)

    s1h, c1h = study1_per_hint(g1, fdr["Study 1 per-hint (40)"][0])
    s1h = attach_observed(s1h, S1_HINT_CSV, ["type", "ogHint"])
    s1h.to_csv(OUT_DIR / f"sensitivity_study1_per_hint{SUFFIX}.csv", index=False)
    c1h.to_csv(OUT_DIR / f"sensitivity_study1_per_hint_curves{SUFFIX}.csv", index=False)

    s2, c2 = study2_per_hint_timing(fdr["Study 2 per-hint x onset (40)"][0])
    s2 = attach_observed(s2, S2_HINT_CSV, ["type", "hint_timing", "ogHint"])
    s2.to_csv(OUT_DIR / f"sensitivity_study2_per_hint_timing{SUFFIX}.csv", index=False)
    c2.to_csv(
        OUT_DIR / f"sensitivity_study2_per_hint_timing_curves{SUFFIX}.csv", index=False
    )

    write_report(s1t, s1h, s2, fdr, time.time() - t0)
    print(
        f"\nwrote sensitivity_*{SUFFIX}.csv and sensitivity_report{SUFFIX}.md in {OUT_DIR} "
        f"(total {time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
