"""Replication driver: run N seeds per scenario, aggregate, persist tables."""
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import screening as sc

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
DATA = ROOT / "data"
TAB.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

METRICS = ["corr_b_beta", "corr_b_share", "corr_b_s", "corr_b_behavior",
           "auc", "cliffs_delta", "silhouette", "variance_ratio",
           "var_beta_star", "extreme_isolated"]


def replicate(cfg, scenario, n_rep):
    base = cfg["seeds"]
    recs = []
    for r in range(n_rep):
        c = copy.deepcopy(cfg)
        for k in ["structure", "beta", "noise", "asset", "behavior"]:
            c["seeds"][k] = base[k] + r  # offset all streams jointly per replication
        _, m, _ = sc.run_scenario(c, scenario)
        recs.append({k: m[k] for k in METRICS})
    return pd.DataFrame(recs)


def summarize(df):
    out = {}
    for c in df.columns:
        v = df[c].astype(float)
        out[c] = (v.mean(), v.quantile(0.025), v.quantile(0.975))
    return out


def main():
    cfg = sc.load_config(ROOT / "config.yaml")
    n_rep = cfg["seeds"]["n_replications"]
    rows = []
    example_frames = {}
    scenario_names = list(cfg["scenarios"].keys())
    for scen in scenario_names:
        rep = replicate(cfg, scen, n_rep)
        rep.to_csv(DATA / f"replications_{scen}.csv", index=False)
        s = summarize(rep)
        for metric, (mean, lo, hi) in s.items():
            rows.append(dict(scenario=scen, metric=metric,
                             mean=mean, ci_lo=lo, ci_hi=hi))
        # save one representative frame (seed offset 0) for positioning map
        df0, _, idx0 = sc.run_scenario(cfg, scen)
        df0["extreme_row"] = df0.index == idx0
        df0.to_csv(DATA / f"targets_{scen}.csv", index=False)
        example_frames[scen] = df0

    # --- rho reliability-profile sensitivity sweep ---
    rho_rows = []
    base_rel = cfg["reliability"]
    shifts = np.round(np.arange(-0.20, 0.201, 0.05), 2)
    for scen in list(cfg["scenarios"].keys()):
        for d in shifts:
            cb, au, iso = [], [], []
            for r in range(20):
                cc = copy.deepcopy(cfg)
                cc["reliability"] = {k: float(np.clip(v + d, 0.05, 0.95))
                                     for k, v in base_rel.items()}
                for kk in ["structure", "beta", "noise", "asset", "behavior"]:
                    cc["seeds"][kk] = cfg["seeds"][kk] + r
                _, m, _ = sc.run_scenario(cc, scen)
                cb.append(m["corr_b_beta"]); au.append(m["auc"])
                iso.append(m["extreme_isolated"])
            rho_rows.append(dict(scenario=scen, rho_shift=float(d),
                                 corr_b_beta=np.mean(cb), auc=np.mean(au),
                                 iso=np.mean(iso)))
    pd.DataFrame(rho_rows).to_csv(DATA / "rho_sensitivity.csv", index=False)

    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "table1_metrics.csv", index=False)
    # pretty pivot
    piv = tab.pivot(index="metric", columns="scenario", values="mean").round(3)
    piv.to_csv(TAB / "table1_metrics_pivot.csv")
    print(piv)
    print("\nSaved tables to", TAB)
    return tab, example_frames


if __name__ == "__main__":
    main()
