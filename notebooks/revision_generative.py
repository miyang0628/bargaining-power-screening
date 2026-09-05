"""Revision R2: robustness of the headline finding across generative families.
Answers reviewer concern 2: is the concentration ordering an artifact of Pareto?

Re-runs the three scenarios under pareto / lognormal / truncated size laws and
reports Var(beta*), AUC, and extreme-case isolation for each.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import screening as sc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"


def run(cfg, scenario, dist, n_rep=100):
    vb, auc, iso = [], [], []
    for r in range(n_rep):
        c = copy.deepcopy(cfg)
        c["size_distribution"] = dist
        for k in ["structure", "beta", "noise", "asset", "behavior"]:
            c["seeds"][k] = cfg["seeds"][k] + r
        _, m, _ = sc.run_scenario(c, scenario)
        vb.append(m["var_beta_star"]); auc.append(m["auc"]); iso.append(m["extreme_isolated"])
    return np.mean(vb), np.mean(auc), np.mean(iso)


def main():
    cfg = sc.load_config(ROOT / "config.yaml")
    n_rep = cfg["seeds"]["n_replications"]
    rows = []
    for dist in ["pareto", "lognormal", "truncated"]:
        for scen in ["oligopolistic", "intermediate", "fragmented"]:
            vb, auc, iso = run(cfg, scen, dist, n_rep)
            rows.append(dict(distribution=dist, scenario=scen,
                             var_beta=vb, auc=auc, iso=iso))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_generative_robustness.csv", index=False)
    # check monotone ordering of var_beta by concentration within each dist
    print(out.round(3).to_string(index=False))
    print("\nMonotone Var(beta*) ordering (oligo >= inter >= frag) per distribution:")
    for dist in ["pareto", "lognormal", "truncated"]:
        d = out[out.distribution == dist].set_index("scenario")["var_beta"]
        mono = d["oligopolistic"] >= d["intermediate"] >= d["fragmented"]
        print(f"  {dist:10s}: {'YES' if mono else 'NO'} "
              f"({d['oligopolistic']:.3f} / {d['intermediate']:.3f} / {d['fragmented']:.3f})")
    return out


if __name__ == "__main__":
    main()
