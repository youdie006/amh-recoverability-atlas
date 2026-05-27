# Generative Feature-Space Transport for AMH-Linked Information Across IVF Modalities

Code release for a study that maps **where information related to anti-Müllerian hormone (AMH) /
ovarian reserve is recoverable** in learned representations of public IVF data — across maternal
endocrine trajectories, ovarian ultrasound, and embryo morphology.

Rather than asking only whether a discriminative probe can read a label off frozen foundation-model
features, we ask the question **generatively**: does an AMH- (or reserve-) conditioned generative
model induce a statistically non-random shift of the *feature distribution*? We instantiate this with
flow matching (FM), optimal-transport conditional flow matching (OT-CFM), and denoising diffusion
bridge models (DDBM), all operating in feature space (no pixel generation), and read each transport
together with a permutation null.

## Key findings

| Modality (cohort) | AMH relation | Generative transport | Distribution-free | Verdict |
|---|---|---|---|---|
| Endocrine trajectory (Brigham) | direct (measured AMH) | AMH-conditioned FM, Cohen's d = 1.14, perm p < 0.001 | C2ST 0.66 | directly recoverable |
| Ovarian ultrasound (fuid / PCOSGen) | AMH-linked reserve phenotype | OT-CFM d = 1.90 after metadata residualization, perm p < 0.001 | C2ST 0.70 | recoverable, distributionally robust |
| Embryo morphology (Wang / Kromp) | measured AMH, image-paired | none | MMD / HSIC / energy n.s. | not practically recoverable (grade AUROC 0.83–0.96) |

A central methodological observation: under acquisition-confound removal (metadata residualization),
the **generative distribution-transport signal survives** while an ordinary **discriminative probe
collapses** toward its metadata baseline (replicated on two independent ovarian cohorts). Analyses are
within-cohort / population-level (distinct populations; no patient-matched causal claim).

## Repository layout

- `experiments/` — image / feature-space transport experiments (ovarian US + embryo), distribution-free
  tests, representational geometry, and figure generation.
- `endocrine/` — AMH-conditioned flow matching on endocrine trajectories.
- `results/` — aggregate result summaries (JSON) that back the tables/figures (no data, no PHI).
- `figures/` — generated figures.

## Reproducing

```bash
pip install -r requirements.txt
export PROJECT_ROOT=/path/to/working/dir   # holds data/external/... and results/diagnostics/
python experiments/ovary_otcfm.py
python experiments/ovary_residualize_localization_null.py
python experiments/pcosgen_dissociation.py
# ... etc; see each script's docstring
```

Each script reads `PROJECT_ROOT` (default: current directory) and expects the relevant public dataset
under `PROJECT_ROOT/data/external/`. Frozen encoders are downloaded from Hugging Face on first run.

## Data (public)

- Endocrine: Brigham/Leahy 2021 IVF cycle data (measured AMH + hormone trajectories).
- Ovarian ultrasound: fuid (Borna et al., Front. Physiol. 2025); PCOSGen (Zenodo, AUTO-PCOS).
- Embryo: Wang 2026 (Mendeley Data); Kromp 2023 (Scientific Data).

No raw data is redistributed here; please obtain datasets from their original sources.

## Methods referenced

Flow Matching (Lipman et al., ICLR 2023, arXiv:2210.02747); OT-CFM (Tong et al., TMLR 2024,
arXiv:2302.00482); DDBM (Zhou et al., ICLR 2024, arXiv:2309.16948); DINOv2 (Oquab et al., TMLR 2024,
arXiv:2304.07193); Classifier Two-Sample Test (Lopez-Paz & Oquab, ICLR 2017); MMD / HSIC (Gretton et
al.); Kraskov mutual information (2004).

## License

MIT (see `LICENSE`).
