"""
Screening Delegation Targets under Bargaining-Power Dependence
Synthetic-data proof-of-concept — core module.

Design principle: generation-validation split.
  - g() builds a LATENT true bargaining power beta* (never seen by the procedure).
  - observed proxies are noisy logit-space emanations of beta*.
  - the screening procedure sees ONLY proxies; beta* is used solely as an answer key.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.special import expit as sigmoid, logit
from sklearn.metrics import roc_auc_score, silhouette_score

DOMAINS = ["E", "S", "G"]


# ----------------------------- utilities -----------------------------
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _z(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else x - x.mean()


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


# ------------------------- data generation ---------------------------
def generate_sizes(N, alpha, rng):
    """Power-law (Pareto) sizes; smaller alpha -> heavier tail -> more concentrated."""
    return (rng.pareto(alpha, size=N) + 1.0)


def generate_scenario(cfg, scenario_name):
    sc = cfg["scenarios"][scenario_name]
    rng_struct = np.random.default_rng(cfg["seeds"]["structure"])
    rng_beta = np.random.default_rng(cfg["seeds"]["beta"])
    rng_asset = np.random.default_rng(cfg["seeds"]["asset"])
    g = cfg["g_function"]

    rows = []
    for dom in DOMAINS:
        N, alpha = sc["N"][dom], sc["alpha"][dom]
        size = generate_sizes(N, alpha, rng_struct)
        share = size / size.sum()
        conc = gini(size)  # domain-level concentration, broadcast to members

        a_irr, b_irr = g["irr_beta_params"][dom]
        irr = rng_beta.beta(a_irr, b_irr, size=N)

        # latent true bargaining power beta*  (the answer key)
        # absolute log-size (centered by a global constant, NOT within-domain z):
        # concentrated structures -> a few very large sizes -> higher beta* spread
        log_size_c = np.log(size) - np.log(2.0)
        logit_beta = (
            g["theta_size"] * log_size_c
            + g["theta_irr"] * (irr - 0.5) * 2.0
            + g["theta_conc"] * ((conc - 0.4) * 2.0)
            + rng_beta.normal(0, g["sigma_beta"], size=N)
        )
        beta_star = sigmoid(logit_beta)

        # asset specificity s — INDEPENDENT seed stream (discriminant validity)
        s = rng_asset.beta(2, 5, size=N)

        for i in range(N):
            rows.append(dict(domain=dom, size=size[i], share=share[i],
                             conc=conc, irr=irr[i], beta_star=beta_star[i], s=s[i]))
    df = pd.DataFrame(rows).reset_index(drop=True)

    # throughput = size * lognormal noise  (capacity axis)
    rng_tp = np.random.default_rng(cfg["seeds"]["structure"] + 7)
    df["throughput"] = df["size"] * rng_tp.lognormal(0, 0.3, size=len(df))
    return df


def _reliability_to_sigma(rho, var_logit_beta):
    return np.sqrt(var_logit_beta * (1 - rho) / rho)


def add_noisy_proxies(df, cfg):
    """Observed proxies = sigmoid(logit(beta*) + common + idiosyncratic noise)."""
    rng = np.random.default_rng(cfg["seeds"]["noise"])
    rel = cfg["reliability"]
    w_c = cfg["noise_common_ratio"]

    lb = logit(np.clip(df["beta_star"].values, 0.02, 0.98))
    var_lb = lb.var()
    common = rng.normal(0, 1, size=len(df))  # shared "this record is generally noisy"

    proxy_map = {"reach": "reach", "share": "share",
                 "switch_cost": "switch", "concentration": "conc_proxy"}
    for rkey, out in proxy_map.items():
        sigma = _reliability_to_sigma(rel[rkey], var_lb)
        idio = rng.normal(0, 1, size=len(df))
        eps = np.sqrt(w_c) * (sigma * common) + np.sqrt(1 - w_c) * (sigma * idio)
        df[out] = sigmoid(lb + eps)
    # reach normalized to [0,1] within data
    df["reach"] = (df["reach"] - df["reach"].min()) / (df["reach"].max() - df["reach"].min() + 1e-9)
    return df


def insert_extreme_case(df, cfg):
    """Plant a 'super-supplier': high beta* target in the non-substitutable domain."""
    ec = cfg["extreme_case"]
    dom_mask = df["domain"] == ec["domain"]
    idx = df.loc[dom_mask].sort_values("beta_star").index[-1]
    df.loc[idx, "beta_star"] = max(df.loc[idx, "beta_star"], ec["beta_floor"])
    df.loc[idx, "is_extreme"] = True
    df["is_extreme"] = df["is_extreme"].fillna(False)
    return df, idx


# ------------------------- index & risk ------------------------------
def compute_b(df, cfg):
    w = cfg["b_weights"]
    b = (w["reach"] * df["reach"] + w["share"] * df["share_proxy_norm"]
         + w["switch_cost"] * df["switch"] + w["concentration"] * df["conc_proxy"])
    return b


def compute_risk(b, s, c, cfg):
    """U-shaped in b:  R = (1+s+c)*R_d(b) + R_e(b)."""
    rd_p = cfg["risk"]["rd_power"]
    re_p = cfg["risk"]["re_power"]
    R_d = b ** rd_p                 # dependence-reversal: increasing, convex
    R_e = (1 - b) ** re_p           # exploitation/exit: decreasing, convex
    # (1+s+c) scales the dependence-reversal arm (high specificity/concentration
    # makes lock-in costlier). Divide R_e by the same factor so the U-trough
    # stays interior rather than being pushed to b->0.
    w = 1 + s + c
    return w * R_d + R_e / w


# ---------------------------- screening ------------------------------
def screen(df, cfg):
    tq = cfg["screening"]["throughput_quantile"]
    rf = cfg["screening"]["risk_lower_fraction"]
    tp_ok = df["throughput"] >= df["throughput"].quantile(tq)
    risk_cut = df["risk"].quantile(rf)
    risk_ok = df["risk"] <= risk_cut
    df["selected"] = tp_ok & risk_ok
    return df


# --------------------------- validation ------------------------------
def validate(df, cfg):
    out = {}
    # convergent: b vs latent beta*  (no circularity — beta* unseen)
    out["corr_b_beta"] = np.corrcoef(df["b"], df["beta_star"])[0, 1]
    # reference (downgraded): b vs its own input share
    out["corr_b_share"] = np.corrcoef(df["b"], df["share"])[0, 1]
    # discriminant: b vs asset specificity s (should be ~0)
    out["corr_b_s"] = np.corrcoef(df["b"], df["s"])[0, 1]
    # criterion: b predicts behavioral outcome
    out["corr_b_behavior"] = np.corrcoef(df["b"], df["behavior"])[0, 1]

    # discrimination via beta*: the manageable optimum is the INDIVIDUAL
    # U-trough b*(s,c) = 1/(w^2+1), w = 1+s+c. A target is "manageable" when its
    # TRUE beta* sits near its own optimum. AUC: does low risk rank them to top?
    w = 1 + df["s"] + df["conc_proxy"]
    b_opt = 1.0 / (w**2 + 1.0)
    dist = (df["beta_star"] - b_opt).abs()
    manageable = (dist <= dist.median()).astype(int)  # closer half to own optimum
    if manageable.nunique() > 1:
        out["auc"] = roc_auc_score(manageable, -df["risk"])
    else:
        out["auc"] = np.nan
    # separation: distance-to-optimum, selected vs isolated (Cliff's delta)
    d_sel = dist[df["selected"]].values
    d_iso = dist[~df["selected"]].values
    if len(d_sel) > 1 and len(d_iso) > 1:
        out["cliffs_delta"] = float(np.sign(d_iso[:, None] - d_sel[None, :]).mean())
    else:
        out["cliffs_delta"] = np.nan
    # silhouette of risk-based 2-grouping in proxy space
    feats = df[["reach", "share_proxy_norm", "switch", "conc_proxy"]].values
    labels = (df["risk"] > df["risk"].median()).astype(int).values
    out["silhouette"] = silhouette_score(feats, labels) if len(set(labels)) > 1 else np.nan
    # structural: variance of beta*
    out["var_beta_star"] = df["beta_star"].var()
    # legacy variance ratio (between/within on risk groups)
    g0, g1 = df.loc[labels == 0, "risk"], df.loc[labels == 1, "risk"]
    within = (g0.var() * len(g0) + g1.var() * len(g1)) / len(df)
    between = ((g0.mean() - df["risk"].mean())**2 * len(g0)
               + (g1.mean() - df["risk"].mean())**2 * len(g1)) / len(df)
    out["variance_ratio"] = between / within if within > 1e-9 else np.nan
    return out


def add_behavior(df, cfg):
    """Behavioral outcome = h(beta*) + noise; NOT part of b (criterion validity)."""
    rng = np.random.default_rng(cfg["seeds"]["behavior"])
    df["behavior"] = np.clip(df["beta_star"] + rng.normal(0, 0.15, len(df)), 0, 1)
    return df


# --------------------------- full pipeline ---------------------------
def run_scenario(cfg, scenario_name):
    df = generate_scenario(cfg, scenario_name)
    df = add_noisy_proxies(df, cfg)
    df, ext_idx = insert_extreme_case(df, cfg)
    # after planting, refresh proxies for the extreme row to reflect new beta*
    df = add_noisy_proxies(df, cfg)
    df, ext_idx = insert_extreme_case(df, cfg)
    df = add_behavior(df, cfg)

    df["share_proxy_norm"] = (df["share"] - df["share"].min()) / (df["share"].max() - df["share"].min() + 1e-9)
    df["b"] = compute_b(df, cfg)
    df["risk"] = compute_risk(df["b"].values, df["s"].values, df["conc_proxy"].values, cfg)
    df = screen(df, cfg)

    metrics = validate(df, cfg)
    metrics["scenario"] = scenario_name
    metrics["extreme_isolated"] = bool(~df.loc[ext_idx, "selected"])
    metrics["extreme_b"] = float(df.loc[ext_idx, "b"])
    metrics["extreme_risk"] = float(df.loc[ext_idx, "risk"])
    metrics["n_targets"] = len(df)
    return df, metrics, ext_idx
