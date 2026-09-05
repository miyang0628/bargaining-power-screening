"""Revision R1: baseline comparison — competence-only vs two-axis screening.
Answers reviewer concern 5: quantify what the second (bargaining-power) axis buys.

Competence-only rule: select top-throughput targets (above median), ignore risk.
Two-axis rule: throughput above median AND total risk in lowest tier.
Key decision-relevant metric: how often each rule RETAINS the planted super-supplier
(a retention is a screening failure), plus mean latent power of the selected set.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import screening as sc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"


def competence_only(df, cfg):
    tq = cfg["screening"]["throughput_quantile"]
    return df["throughput"] >= df["throughput"].quantile(tq)


def run_baseline(cfg, scenario, n_rep=100):
    recs = []
    for r in range(n_rep):
        c = copy.deepcopy(cfg)
        for k in ["structure", "beta", "noise", "asset", "behavior"]:
            c["seeds"][k] = cfg["seeds"][k] + r
        df, m, ext_idx = sc.run_scenario(c, scenario)
        comp_sel = competence_only(df, c)
        two_sel = df["selected"]
        # super-supplier retention (a failure): selected == retained
        rec = dict(
            scenario=scenario,
            comp_retains_super=bool(comp_sel.loc[ext_idx]),
            two_retains_super=bool(two_sel.loc[ext_idx]),
            comp_mean_beta=float(df.loc[comp_sel, "beta_star"].mean()),
            two_mean_beta=float(df.loc[two_sel, "beta_star"].mean()),
            comp_max_beta=float(df.loc[comp_sel, "beta_star"].max()),
            two_max_beta=float(df.loc[two_sel, "beta_star"].max()),
            comp_n=int(comp_sel.sum()),
            two_n=int(two_sel.sum()),
        )
        recs.append(rec)
    return pd.DataFrame(recs)


def main():
    cfg = sc.load_config(ROOT / "config.yaml")
    n_rep = cfg["seeds"]["n_replications"]
    rows = []
    for scen in ["oligopolistic", "intermediate", "fragmented"]:
        b = run_baseline(cfg, scen, n_rep)
        b.to_csv(DATA / f"baseline_{scen}.csv", index=False)
        rows.append(dict(
            scenario=scen,
            comp_super_retention=b["comp_retains_super"].mean(),
            two_super_retention=b["two_retains_super"].mean(),
            comp_mean_beta=b["comp_mean_beta"].mean(),
            two_mean_beta=b["two_mean_beta"].mean(),
            comp_max_beta=b["comp_max_beta"].mean(),
            two_max_beta=b["two_max_beta"].mean(),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "table_baseline.csv", index=False)
    pd.set_option("display.width", 160)
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
