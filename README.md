# A Target-Separated Recoverability Audit for Frozen Foundation-Model Embeddings, Demonstrated on AMH in IVF

This repository audits where AMH and ovarian-reserve information is recoverable from frozen foundation-model embeddings across public IVF-related data, including DINOv2, DINOv3, CLIP, USF-MAE, and related encoders. The analyses use predictive V-information probing, conditional age probing, control tasks, MDL summaries, and model-class sufficiency checks with TOST. The contribution is the target-separated audit protocol and the reusable `reserve_audit` package, not a new model.

## Key Findings

1. Two single-cohort age-conditioning results separate reserve-relevant and reserve-adjacent settings. In Brigham/Leahy endocrine trajectories, AMH-tertile recoverability changed from marginal `R/H=0.100225` to age-conditioned `R/H=0.087189`; the 2000-shuffle conditional permutation result was `p=0.0004997501249375312`. In Wang embryo morphology, AMH-tertile recoverability changed from marginal `R/H=0.004202` to age-conditioned `R/H=-0.000044`, with `p=0.597015`.

2. The same frozen-encoder pipeline recovers an ovary-US phenotype positive control: DINOv2 `R/H=0.3401` and USF-MAE `R/H=0.2974` on FUID. This makes the embryo result a measured boundary rather than a failed pipeline check.

3. Clinical concordance points in the same direction. A 6-cohort AMH meta-analysis found a quantity-marker association of `r=0.4054` with 95% HKSJ CI `0.2833` to `0.5145` and `p=0.0005032429233777444`; the quality-marker pool was near zero at `r=0.0156` with 95% HKSJ CI `-0.0262` to `0.0575` and `p=0.35808119845986486`. Quantity effects were positive in 6/6 cohorts, while quality effects were near zero in 5/5 estimable cohorts.

These are modest aggregate effects. The age-conditioning results are single-cohort and adjust for age only. They should be read as a representation-analysis audit, not as an image-to-serum-AMH predictor.

## Repository Layout

```text
reserve_audit/              reusable audit package
scripts/                    benchmark, hardening, verification, and figure generators
results/diagnostics/        aggregate diagnostic JSON files used for the public claims
figures/                    public figure generated from aggregate diagnostics
requirements.txt            minimal Python dependencies
```

## Reproducing

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

The diagnostics JSON files in `results/diagnostics/` back the numeric claims above. The public figure can be regenerated from the aggregate benchmark JSON:

```bash
python scripts/make_fig_encoder_benchmark.py
```

The benchmark-value check can be run with:

```bash
python scripts/verify_benchmark.py
```

Raw cohort data and cached embeddings are not redistributed here because users should obtain those assets from their original sources and licenses. This release contains the protocol code, aggregate diagnostics, figure, and generators.

## Data

Public cohorts used by the audit include Brigham/Leahy 2021, Mendeley 5k IVF, Wang embryo-image data, FUID ovary-US, Clomiphene DOR, PGT-A euploidy DOR, Embryoscope hr-NGS KIDScore, and Heavy-metal DOR. Users should obtain each cohort from its original source.

## Scope

This is representation-analysis and probing methodology demonstrated on AMH and IVF; image-to-serum-AMH is future work gated on paired ovary-US plus continuous-AMH data.
