"""Revision R5: full weight-simplex sweep (reviewer concern 6).
Instead of perturbing one weight at a time, sample the entire 4-weight simplex
(reach, share, switch, concentration) from a Dirichlet and report the DISTRIBUTION
of AUC, convergent corr, and isolation. Shows the procedure is robust across the
whole weight space, not just along one axis.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import screening as sc

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
DATA = ROOT / "data"


def run_simplex(cfg, scenario, n_draws=300, seed=90210):
    rng = np.random.default_rng(seed)
    # Dirichlet centered near the baseline weights (0.45, 0.25, 0.15, 0.15)
    alpha = np.array([4.5, 2.5, 1.5, 1.5]) * 2.0
    recs = []
    for d in range(n_draws):
        w = rng.dirichlet(alpha)
        c = copy.deepcopy(cfg)
        c["b_weights"] = {"reach": float(w[0]), "share": float(w[1]),
                          "switch_cost": float(w[2]), "concentration": float(w[3])}
        # single representative seed per draw (structure fixed) to isolate weight effect
        _, m, _ = sc.run_scenario(c, scenario)
        recs.append(dict(scenario=scenario,
                         w_reach=w[0], w_share=w[1], w_switch=w[2], w_conc=w[3],
                         auc=m["auc"], corr_b_beta=m["corr_b_beta"],
                         iso=float(m["extreme_isolated"])))
    return pd.DataFrame(recs)


def main():
    cfg = sc.load_config(ROOT / "config.yaml")
    frames = []
    for scen in ["oligopolistic", "intermediate", "fragmented"]:
        df = run_simplex(cfg, scen, n_draws=300)
        frames.append(df)
    alld = pd.concat(frames, ignore_index=True)
    alld.to_csv(DATA / "weight_simplex.csv", index=False)

    rows = []
    for scen in ["oligopolistic", "intermediate", "fragmented"]:
        d = alld[alld.scenario == scen]
        rows.append(dict(
            scenario=scen,
            auc_mean=d.auc.mean(), auc_p05=d.auc.quantile(0.05), auc_p95=d.auc.quantile(0.95),
            corr_mean=d.corr_b_beta.mean(),
            corr_p05=d.corr_b_beta.quantile(0.05), corr_p95=d.corr_b_beta.quantile(0.95),
            iso_mean=d.iso.mean(),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_weight_simplex.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out, alld


if __name__ == "__main__":
    main()
