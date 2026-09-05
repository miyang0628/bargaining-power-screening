"""Revision R3: sensitivity to the risk-function functional form.
Answers reviewer concern 4: is the U-shape / interior optimum an artifact of the
specific algebra R = w*b^p + (1-b)^p / w ?

Varies (a) the convexity exponent p in {1.5, 2, 3} and
(b) the coupling of w: 'multiplicative' (baseline, w scales R_d and 1/w scales R_e)
    vs 'additive' (R = (1+s+c) added to R_d, R_e unscaled).
Reports AUC, extreme isolation, and the interior-optimum share (fraction of
targets whose risk-minimizing b lies strictly inside (0,1)).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import screening as sc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"


def compute_risk_variant(b, s, c, p, coupling):
    R_d = b ** p
    R_e = (1 - b) ** p
    w = 1 + s + c
    if coupling == "multiplicative":
        return w * R_d + R_e / w
    if coupling == "additive":
        return w * R_d + R_e          # w scales only R_d; interior trough still possible
    raise ValueError(coupling)


def interior_optimum_share(s, c, p, coupling, grid=None):
    """Fraction of targets whose risk-minimizing b over a grid is interior."""
    if grid is None:
        grid = np.linspace(0.001, 0.999, 199)
    interior = 0
    for si, ci in zip(s, c):
        R = compute_risk_variant(grid, si, ci, p, coupling)
        argmin = grid[np.argmin(R)]
        if 0.02 < argmin < 0.98:
            interior += 1
    return interior / len(s)


def run(cfg, scenario, p, coupling, n_rep=60):
    from sklearn.metrics import roc_auc_score
    aucs, isos, ish = [], [], []
    for r in range(n_rep):
        c = copy.deepcopy(cfg)
        for k in ["structure", "beta", "noise", "asset", "behavior"]:
            c["seeds"][k] = cfg["seeds"][k] + r
        df = sc.generate_scenario(c, scenario)
        df = sc.add_noisy_proxies(df, c)
        df, ext_idx = sc.insert_extreme_case(df, c)
        df = sc.add_noisy_proxies(df, c)
        df, ext_idx = sc.insert_extreme_case(df, c)
        df = sc.add_behavior(df, c)
        df["share_proxy_norm"] = (df["share"] - df["share"].min()) / (df["share"].max() - df["share"].min() + 1e-9)
        df["b"] = sc.compute_b(df, c)
        df["risk"] = compute_risk_variant(df["b"].values, df["s"].values, df["conc_proxy"].values, p, coupling)
        df = sc.screen(df, c)
        # AUC vs individual optimum proximity, recomputed for this variant
        w = 1 + df["s"] + df["conc_proxy"]
        # interior optimum for multiplicative quadratic is 1/(w^2+1); for general
        # forms, locate numerically per target
        grid = np.linspace(0.001, 0.999, 199)
        bopt = np.array([grid[np.argmin(compute_risk_variant(grid, si, ci, p, coupling))]
                         for si, ci in zip(df["s"].values, df["conc_proxy"].values)])
        dist = np.abs(df["beta_star"].values - bopt)
        manageable = (dist <= np.median(dist)).astype(int)
        if len(set(manageable)) > 1:
            aucs.append(roc_auc_score(manageable, -df["risk"].values))
        isos.append(bool(~df.loc[ext_idx, "selected"]))
        ish.append(interior_optimum_share(df["s"].values, df["conc_proxy"].values, p, coupling))
    return np.mean(aucs), np.mean(isos), np.mean(ish)


def main():
    cfg = sc.load_config(ROOT / "config.yaml")
    rows = []
    for coupling in ["multiplicative", "additive"]:
        for p in [1.5, 2.0, 3.0]:
            # aggregate across the three scenarios for a compact robustness table
            a_all, i_all, s_all = [], [], []
            for scen in ["oligopolistic", "intermediate", "fragmented"]:
                a, i, sh = run(cfg, scen, p, coupling)
                a_all.append(a); i_all.append(i); s_all.append(sh)
            rows.append(dict(coupling=coupling, exponent=p,
                             auc=np.mean(a_all), iso=np.mean(i_all),
                             interior_share=np.mean(s_all)))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_risk_form.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
