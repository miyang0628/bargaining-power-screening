"""Figure generation — grayscale, dpi 600, PNG+PDF, no captions/titles."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import copy
import screening as sc

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
DATA = ROOT / "data"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
# canonical display order: most -> least concentrated
DISPLAY_ORDER = ["oligopolistic", "intermediate", "fragmented"]


def _ordered(scenarios):
    return [s for s in DISPLAY_ORDER if s in scenarios] + \
           [s for s in scenarios if s not in DISPLAY_ORDER]
plt.rcParams.update({
    "figure.dpi": 600, "savefig.dpi": 600,
    "font.family": "serif", "axes.edgecolor": "0.2",
    "axes.linewidth": 0.8, "grid.color": "0.85", "grid.linewidth": 0.5,
    "image.cmap": "gray",
})
GRAYS = ["0.15", "0.45", "0.70"]


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print("saved", name)


def fig1_positioning():
    """Capacity (throughput) x bargaining power (b), risk as marker, selected/isolated/extreme."""
    cfg = sc.load_config(ROOT / "config.yaml")
    scenarios = _ordered(list(cfg["scenarios"].keys()))
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.3), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, scen in zip(axes, scenarios):
        df, m, idx = sc.run_scenario(cfg, scen)
        tp = np.log10(df["throughput"])
        sel = df["selected"].values
        ext = np.asarray(df.index == idx)
        # isolated (not selected, not extreme)
        base = (~sel) & (~ext)
        ax.scatter(df["b"][base], tp[base], s=18, c="0.70",
                   marker="o", edgecolors="0.4", linewidths=0.3, label="Isolated")
        ax.scatter(df["b"][sel & ~ext], tp[sel & ~ext], s=26, c="0.15",
                   marker="s", edgecolors="k", linewidths=0.3, label="Selected")
        ax.scatter(df["b"][ext], tp[ext], s=95, c="1.0",
                   marker="*", edgecolors="k", linewidths=1.1, label="Extreme (super-supplier)")
        ax.set_xlabel(r"Latent bargaining-power index $b$")
        ax.text(0.03, 0.94, scen.capitalize(), transform=ax.transAxes,
                fontsize=9, va="top", fontweight="bold")
    axes[0].set_ylabel(r"Throughput ($\log_{10}$)")
    axes[-1].legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "fig1_positioning_map")


def fig2_sensitivity():
    """AUC heatmap over b-weight perturbation x noise-common-ratio, per scenario."""
    cfg = sc.load_config(ROOT / "config.yaml")
    scenarios = _ordered(list(cfg["scenarios"].keys()))
    reach_w = np.round(np.linspace(0.30, 0.60, 6), 3)
    wc_grid = np.round(np.linspace(0.10, 0.50, 6), 3)
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, scen in zip(axes, scenarios):
        M = np.zeros((len(reach_w), len(wc_grid)))
        for i, rw in enumerate(reach_w):
            for j, wc in enumerate(wc_grid):
                c = copy.deepcopy(cfg)
                # redistribute weight: reach takes rw, remainder keeps proportions
                rem = 1 - rw
                base = {"share": 0.25, "switch_cost": 0.15, "concentration": 0.15}
                tot = sum(base.values())
                c["b_weights"] = {"reach": rw,
                                  "share": rem * base["share"] / tot,
                                  "switch_cost": rem * base["switch_cost"] / tot,
                                  "concentration": rem * base["concentration"] / tot}
                c["noise_common_ratio"] = float(wc)
                _, mm, _ = sc.run_scenario(c, scen)
                M[i, j] = mm["auc"]
        sns.heatmap(M, ax=ax, cmap="gray_r", vmin=0.6, vmax=0.95,
                    xticklabels=wc_grid, yticklabels=reach_w,
                    cbar_kws={"label": "AUC"}, linewidths=0.4, linecolor="0.9",
                    annot=True, fmt=".2f", annot_kws={"size": 6})
        ax.set_xlabel(r"Noise common ratio $w_c$")
        ax.set_ylabel(r"reach weight in $b$")
        ax.text(0.5, 1.04, scen.capitalize(), transform=ax.transAxes,
                ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig2_sensitivity_heatmap")


def fig3_isolation_and_discrimination():
    """(a) extreme-case isolation recall over reps; (b) beta* variance & discrimination bars."""
    cfg = sc.load_config(ROOT / "config.yaml")
    scenarios = _ordered(list(cfg["scenarios"].keys()))
    reps = {s: pd.read_csv(DATA / f"replications_{s}.csv") for s in scenarios}
    shades = [str(round(0.20 + 0.55 * i / max(1, len(scenarios) - 1), 2))
              for i in range(len(scenarios))]
    labels_scen = [s.capitalize() for s in scenarios]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    # (a) isolation recall
    recall = [reps[s]["extreme_isolated"].mean() for s in scenarios]
    axes[0].bar(labels_scen, recall, color=shades, edgecolor="k",
                linewidth=0.6, width=0.6)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Extreme-case isolation recall")
    axes[0].tick_params(axis="x", labelsize=7)
    for i, v in enumerate(recall):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # (b) grouped bars: var_beta_star, silhouette, auc
    metrics = ["var_beta_star", "silhouette", "auc"]
    mlabels = [r"Var($\beta^*$)", "Silhouette", "AUC"]
    x = np.arange(len(metrics))
    w = 0.8 / len(scenarios)
    for k, s in enumerate(scenarios):
        vals = [reps[s][m].mean() for m in metrics]
        axes[1].bar(x + (k - (len(scenarios) - 1) / 2) * w, vals, w,
                    label=labels_scen[k], color=shades[k],
                    edgecolor="k", linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mlabels, fontsize=8)
    axes[1].set_ylabel("Value")
    axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    _save(fig, "fig3_isolation_discrimination")


def fig4_rho_sensitivity():
    """corr(b, beta*) and AUC vs joint reliability-profile shift, per scenario."""
    df = pd.read_csv(DATA / "rho_sensitivity.csv")
    scenarios = _ordered(df["scenario"].unique().tolist())
    shades = {s: str(round(0.15 + 0.55 * i / max(1, len(scenarios) - 1), 2))
              for i, s in enumerate(scenarios)}
    markers = {0: "o", 1: "s", 2: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    cfg = sc.load_config(ROOT / "config.yaml")
    lo = cfg["validation"]["target_corr_beta_low"]
    hi = cfg["validation"]["target_corr_beta_high"]

    for ax, col, ylab in zip(axes, ["corr_b_beta", "auc"],
                             [r"corr($b$, $\beta^*$)", "Discrimination AUC"]):
        for i, s in enumerate(scenarios):
            d = df[df["scenario"] == s].sort_values("rho_shift")
            ax.plot(d["rho_shift"], d[col], color=shades[s],
                    marker=markers.get(i, "o"), markersize=4, linewidth=1.3,
                    markeredgecolor="k", markeredgewidth=0.4, label=s.capitalize())
        ax.set_xlabel(r"Joint reliability shift $\Delta\rho$")
        ax.set_ylabel(ylab)
    axes[0].axhspan(lo, hi, color="0.85", alpha=0.6, zorder=0)
    axes[0].axhline(lo, color="0.5", lw=0.6, ls="--")
    axes[0].axhline(hi, color="0.5", lw=0.6, ls="--")
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig4_rho_sensitivity")


if __name__ == "__main__":
    fig1_positioning()
    fig2_sensitivity()
    fig3_isolation_and_discrimination()
    fig4_rho_sensitivity()
    print("all figures done ->", FIG)
