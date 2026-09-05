# Screening Delegation Targets under Bargaining-Power Dependence

**A Synthetic-Data Proof-of-Concept**

Reproducibility package for the manuscript of the same title, currently under
review (author and venue withheld for double-blind review).

---

## Overview

When low-knowledge, high-repetition work is delegated to an external party,
screening candidates on **task competence alone** ignores the risk that a
candidate later accrues **bargaining power** over the delegator. This repository
provides a two-dimensional screening procedure — a competence axis (throughput)
and a bargaining-power–dependence axis (latent power `b`) — and a synthetic-data
protocol that establishes its **construct validity, discriminative power, and
robustness** under realistic conditions where real transaction data are
inaccessible.

The study is a **proof-of-concept**: it does not test the unobservable
counterfactual of whether a given supplier would in fact behave opportunistically.
It tests whether the *procedure* is well-constructed, discriminating, and robust.

## Design principle: generation–validation split

The data-generating rules are frozen in [`config.yaml`](config.yaml)
**independently of** the screening procedure. A latent **true bargaining power**
`beta*` is generated as an answer key that the procedure **never observes**;
the procedure sees only noisy proxies. This blocks the "the data were tuned to
make the procedure look good" critique at the file level, and it is why a key
result (below) runs *against* the authors' prior intuition — evidence the split
genuinely held.

## Key result (two propositions)

The finding separates into two claims the literature tends to conflate:

- **P1 — robustness.** Procedure *accuracy* is invariant to market structure:
  discrimination AUC ≈ 0.82 across all three structures (oligopolistic,
  intermediate, fragmented), across the b-weight × noise-common-ratio grid, and
  across joint reliability shifts Δρ ∈ [−0.2, +0.2].
- **P2 — conditional value.** Procedure *discriminative value* — the latent
  bargaining-power heterogeneity there is to exploit — rises **monotonically with
  concentration**: Var(`beta*`) and silhouette order oligopolistic > intermediate
  > fragmented. Under the pre-registered power-law generator, heavy tails place a
  few very large targets in the upper power range, widening heterogeneity exactly
  where concentration is high.

The planted super-supplier is isolated in **100% of replications** in every
structure. Notably, the concentration ordering runs **against the authors' prior
expectation** (that fragmented markets would benefit most); reporting it unchanged
is evidence that the generation–validation split held.

## Repository structure

```
.
├── config.yaml                 # PRE-REGISTERED parameters (frozen before analysis)
│                               #   three structures: oligopolistic / intermediate / fragmented
├── manuscript_framing.md       # revised introduction & contribution plan
├── requirements.txt
├── LICENSE
├── data/                       # generated artefacts (replications, example targets)
├── notebooks/
│   ├── 01_reproduce_all.ipynb  # end-to-end reproduction notebook
│   ├── screening.py            # core: generation, indexing, risk, screening, validation
│   ├── run_replications.py     # 100-seed replication driver + tables
│   └── make_figures.py         # grayscale figures (dpi 600, PNG + PDF)
└── results/
    ├── figures/                # fig1–fig3 (.png and .pdf)
    └── tables/                 # metrics tables (.csv)
```

## Reproduction

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_reproduce_all.ipynb
```

Or run the pipeline directly:

```bash
cd notebooks
python run_replications.py     # writes results/tables/ and data/
python make_figures.py         # writes results/figures/
```

All randomness is controlled by five stratified, purpose-specific seed streams in
`config.yaml` (`structure`, `beta`, `noise`, `asset`, `behavior`), offset jointly
per replication. Results are therefore bit-for-bit reproducible for a fixed
environment.

## Method summary

**Latent true power** `beta*` is generated from log-size, per-domain
irreplaceability, and concentration (`g()` in `screening.py`), with
irreplaceability weighted most heavily (theory: substitutability dominates size).

**Observed proxies** (reach, share, switch cost, concentration) are noisy
logit-space emanations of `beta*`. Noise magnitude is **back-solved from a
pre-registered reliability profile** `rho_x` (how easily each proxy is observed),
not tuned to hit a target correlation. Noise decomposes into a shared component
(ratio `w_c`) and an idiosyncratic component.

**Bargaining-power index** `b` is a weighted sum of proxies. **Risk** is U-shaped
in `b`: `R = w·R_d(b) + R_e(b)/w`, with `w = 1 + s + c` (asset specificity `s`,
concentration `c`); `R_d` is dependence-reversal risk (rising) and `R_e` is
exploitation/exit risk (falling). The individual optimum sits at the U-trough
`b*(s,c) = 1/(w²+1)`.

**Screening rule**: keep targets with above-median throughput **and** total risk
in the lowest 40% (avoiding both extremes — the "manageable middle").

## Validation metrics

| Metric | Type | Circularity | Expectation |
|---|---|---|---|
| corr(`b`, `beta*`) | convergent | none (`beta*` unseen) | 0.70–0.85 |
| corr(`b`, behavior) | criterion | none | positive |
| corr(`b`, `s`) | discriminant | none | ≈ 0 |
| AUC (optimum proximity) | discrimination | none | high |
| Var(`beta*`), silhouette | structural | none | structure-dependent |
| corr(`b`, share) | internal ref. | present | reported for reference only |
| extreme-case isolation | sanity check | none | 1.00 |

The input-sharing correlation corr(`b`, share) is **downgraded to a reference
footnote**; the primary evidence is convergence to the unseen `beta*`, criterion
validity, discriminant separation from `s`, and optimum-proximity AUC.

## Figures

All figures are grayscale, dpi 600, saved as both `.png` and `.pdf`, without
captions or titles (captions live in the manuscript).

- **Fig 1** — capacity × bargaining-power positioning map (selected / isolated /
  extreme), per structure.
- **Fig 2** — AUC sensitivity heatmap over the `b`-weight × noise-common-ratio
  grid (robustness; no direction reversal).
- **Fig 3** — extreme-case isolation recall; structural and discrimination metrics
  by market structure (monotone in concentration).
- **Fig 4** — reliability-profile sensitivity: corr(`b`, `beta*`) and AUC vs joint
  ρ shift, with the pre-registered target band; shows graceful degradation and
  structure-invariance.

## Limitations

- Validation targets construct validity, discrimination, and robustness — **not**
  the actual causal performance of delegation (an unobservable counterfactual),
  which is left to longitudinal / real-data follow-up.
- Results are a **synthetic-data proof-of-concept**, not empirical confirmation.
- Proxy weights in `b` are assigned a priori and treated as a sensitivity-analysis
  object; standardization is future work.

## License

Released under the MIT License (see [LICENSE](LICENSE)). Author attribution is
withheld pending double-blind review.
