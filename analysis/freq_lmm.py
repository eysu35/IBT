from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("freq_lmm")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("  [freq_lmm] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

Z95 = 1.96
_HINT = "_hint_"  # temporary column holding the factor (hint text is not a valid identifier)
_LEVEL_RE = re.compile(r"\[T\.(.*)\]$")
MIN_REL_VAR = 1e-6  # participant variance / residual variance below this counts as "~0"

# --------------------------------------------------------------------------- outcome scale
SPREAD_DEFAULT = 70.0  # best - worst of the (10, 20, 40, 60, 80) arm set: every simulation, and every
# human task except stationary (and the 60-spread stationary batch)
SPREAD_BY_TYPE = {
    "linear_contextual": 7.0
}  # reward range -2..5 (regret = 5 - reward), no arm means
PERCENT = 100.0


def _arm_spread(a):
    try:
        v = np.asarray(a, float).ravel()
    except (TypeError, ValueError):
        return np.nan
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size >= 2 else np.nan


def regret_spread(df, type_col="type", arms_col="ogArms"):
    sp = (
        df[type_col].map(SPREAD_BY_TYPE).astype(float)
        if type_col in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    sp = sp.fillna(SPREAD_DEFAULT)
    if arms_col in df.columns:
        from_arms = df[arms_col].map(_arm_spread)
        use = from_arms.notna() & (
            from_arms > 1.0
        )  # > 1 guards against probability-scaled arms
        if type_col in df.columns:
            use &= ~df[type_col].isin(SPREAD_BY_TYPE)
        sp = sp.where(~use, from_arms)
    return sp


def normalize_regret(df, type_col="type", arms_col="ogArms", regret_col="regret"):
    out = df.copy()
    if "regret_points" in out.columns:  # already normalised
        return out
    out["spread"] = regret_spread(out, type_col=type_col, arms_col=arms_col)
    out["regret_points"] = out[regret_col].astype(float)
    out[regret_col] = PERCENT * out["regret_points"] / out["spread"]
    return out


# --------------------------------------------------------------------------- core statistics
def _normal_p(z):
    return 2.0 * stats.norm.sf(np.abs(z))


def linear_contrast(params, cov, L):
    L = np.asarray(L, float)
    est = float(L @ params)
    se = float(np.sqrt(L @ cov @ L))
    z = est / se if se > 0 else np.nan
    return dict(
        est=est, se=se, z=z, p=float(_normal_p(z)), lo=est - Z95 * se, hi=est + Z95 * se
    )


def fixed_effects(res):
    params = np.asarray(res.params, float)
    names = list(res.params.index)
    cov = np.asarray(res.cov_params(), float)
    k = len(names)
    if cov.shape[0] > k:  # MixedLM: cov_params also covers the variance components
        cov = cov[:k, :k]
    elif hasattr(
        res, "fe_params"
    ):  # MixedLM params may already be the fixed effects only
        k = len(res.fe_params)
        params, names, cov = (
            np.asarray(res.fe_params, float),
            list(res.fe_params.index),
            cov[:k, :k],
        )
    return params, names, cov


def _level_of(name):
    m = _LEVEL_RE.search(name)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- fitting
def _fit_mixed(formula, df, subject_col):
    groups = df[subject_col].to_numpy()
    for method in (None, "lbfgs"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = smf.mixedlm(formula, df, groups=groups)
                res = (
                    model.fit(reml=True, method=method)
                    if method
                    else model.fit(reml=True)
                )
            except Exception as e:  # singular design, numerical failure
                log.info(
                    f"MixedLM{' (' + method + ')' if method else ''} raised {type(e).__name__}: {e}"
                )
                continue
        var_re = float(np.asarray(res.cov_re)[0, 0])
        rel = var_re / float(res.scale) if res.scale > 0 else 0.0
        ok = bool(getattr(res, "converged", True)) and rel > MIN_REL_VAR
        if ok:
            return res, "mixedlm" + (f"_{method}" if method else ""), ""
        why = (
            "did not converge"
            if not getattr(res, "converged", True)
            else f"participant variance ~0 (ratio {rel:.2e})"
        )
        log.info(
            f"MixedLM{' (' + method + ')' if method else ''}: {why}; "
            + (
                "retrying with lbfgs"
                if method is None
                else "falling back to cluster-robust OLS"
            )
        )
        last = why
    return None, "ols_cluster", last


def fit_hint_contrasts(
    df,
    hint_col="ogHint",
    subject_col=None,
    ref="No hint.",
    y="neg_regret",
    label="",
    covariates=None,
):
    covariates = list(covariates or [])
    d = (
        df[[y, hint_col] + covariates + ([subject_col] if subject_col else [])]
        .dropna()
        .copy()
    )
    d[_HINT] = d[hint_col].astype(str)
    levels = [l for l in d[_HINT].unique() if l != ref]
    if ref not in set(d[_HINT]):
        raise ValueError(f"{label}: reference level {ref!r} absent")
    cov_terms = []
    for i, c in enumerate(covariates):
        name = f"_cov{i}_"
        d[name] = d[c].astype(float) - d[c].astype(float).mean()
        cov_terms.append(name)
    formula = f'{y} ~ C({_HINT}, Treatment("{ref}"))' + "".join(
        f" + {t}" for t in cov_terms
    )

    method, note = "ols_hc3", ""
    sd_subject, sd_resid = np.nan, np.nan
    if subject_col:
        res, method, note = _fit_mixed(formula, d, subject_col)
        if res is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.ols(formula, d).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": pd.factorize(d[subject_col])[0]},
                )
            sd_resid = float(np.sqrt(res.scale))
        else:
            sd_subject = float(np.sqrt(np.asarray(res.cov_re)[0, 0]))
            sd_resid = float(np.sqrt(res.scale))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.ols(formula, d).fit(cov_type="HC3")
        sd_resid = float(np.sqrt(res.scale))

    params, names, cov = fixed_effects(res)
    idx = {_level_of(n): i for i, n in enumerate(names) if _level_of(n) is not None}
    i0 = names.index("Intercept")
    e0 = np.zeros(len(names))
    e0[i0] = 1.0
    base = linear_contrast(params, cov, e0)
    n_subj = int(d[subject_col].nunique()) if subject_col else np.nan
    rows = []
    for lev in levels:
        if lev not in idx:
            log.info(
                f"{label}: level {lev!r} has no coefficient (dropped by patsy); skipped"
            )
            continue
        L = np.zeros(len(names))
        L[idx[lev]] = 1.0
        c = linear_contrast(params, cov, L)
        sub = d[d[_HINT] == lev]
        rows.append(
            dict(
                **{hint_col: lev},
                delta=c["est"],
                delta_lo=c["lo"],
                delta_hi=c["hi"],
                se=c["se"],
                z=c["z"],
                p=c["p"],
                n_games=int(len(sub)),
                n_subjects=int(sub[subject_col].nunique()) if subject_col else np.nan,
                no_hint_mean=base["est"],
                no_hint_se=base["se"],
                no_hint_lo=base["lo"],
                no_hint_hi=base["hi"],
                n_games_ref=int((d[_HINT] == ref).sum()),
                n_subjects_total=n_subj,
                sd_subject=sd_subject,
                sd_resid=sd_resid,
                method=method,
                fit_note=note,
            )
        )
    out = pd.DataFrame(rows)
    out.attrs["result"] = res
    out.attrs["baseline"] = dict(
        mean=base["est"],
        se=base["se"],
        lo=base["lo"],
        hi=base["hi"],
        n_games=int((d[_HINT] == ref).sum()),
        n_subjects=n_subj,
        sd_subject=sd_subject,
        sd_resid=sd_resid,
        method=method,
        fit_note=note,
    )
    out.attrs["covariates"] = {
        c: float(params[names.index(t)]) for c, t in zip(covariates, cov_terms)
    }
    return out


def fit_binary_contrast(df, x_col, subject_col=None, y="neg_regret", label=""):
    d = df[[y, x_col] + ([subject_col] if subject_col else [])].dropna().copy()
    d["_x_"] = d[x_col].astype(float)
    formula = f"{y} ~ _x_"
    method, note, sd_subject = "ols_hc3", "", np.nan
    if subject_col:
        res, method, note = _fit_mixed(formula, d, subject_col)
        if res is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.ols(formula, d).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": pd.factorize(d[subject_col])[0]},
                )
        else:
            sd_subject = float(np.sqrt(np.asarray(res.cov_re)[0, 0]))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.ols(formula, d).fit(cov_type="HC3")
    params, names, cov = fixed_effects(res)
    L = np.zeros(len(names))
    L[names.index("_x_")] = 1.0
    c = linear_contrast(params, cov, L)
    return dict(
        delta=c["est"],
        delta_lo=c["lo"],
        delta_hi=c["hi"],
        se=c["se"],
        z=c["z"],
        p=c["p"],
        n_games=int(len(d)),
        n_subjects=int(d[subject_col].nunique()) if subject_col else np.nan,
        n_games_1=int((d["_x_"] == 1).sum()),
        n_games_0=int((d["_x_"] == 0).sum()),
        sd_subject=sd_subject,
        sd_resid=float(np.sqrt(res.scale)),
        method=method,
        fit_note=note,
    )


# --------------------------------------------------------------------------- multiplicity
def add_adjusted_p(df, group_cols=None, p_col="p", out_col="p_adj"):
    df = df.copy()
    df[out_col] = np.nan
    groups = (
        [(None, df)]
        if not group_cols
        else df.groupby(group_cols, sort=False, dropna=False)
    )
    for _, g in groups:
        ok = g[p_col].notna()
        if ok.sum():
            df.loc[g.index[ok], out_col] = multipletests(
                g.loc[ok, p_col].to_numpy(), method="fdr_bh"
            )[1]
    return df


def stars(p_adj):
    if p_adj is None or not np.isfinite(p_adj):
        return ""
    return (
        "***"
        if p_adj < 0.001
        else "**"
        if p_adj < 0.01
        else "*"
        if p_adj < 0.05
        else ""
    )
