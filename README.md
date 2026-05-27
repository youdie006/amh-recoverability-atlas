# Modality-dependent AMH Recoverability and Model-class Sufficiency across IVF Endocrine and Image Representations

Code release for a study that maps, across public IVF data, **(i) where AMH / ovarian-reserve information is
recoverable** in learned representations and **(ii) the simplest model class that suffices to recover it** —
across maternal endocrine trajectories, ovarian ultrasound, and embryo morphology.

We do **not** claim generative models are universally superior. We ask **when distributional generative
modeling is load-bearing and when simpler affine or discriminative models are sufficient**, defining a model
class as *sufficient* when a simpler class matches or exceeds a more complex one (within uncertainty) on
distributional fidelity, AMH-conditioned effect recovery, and downstream phenotype recovery. One shared
protocol (affine / discriminative / generative, with permutation nulls and acquisition-metadata controls) is
applied to every modality.

## Key findings — recoverability is modality-dependent

| Modality (cohort) | AMH-linked signal? | Simplest sufficient model | Neural generative load-bearing? |
|---|---|---|---|
| Endocrine trajectory (Brigham) | yes, direct (measured AMH) | **masked Flow Matching** (AMH-cond d ≈ 1.12; permuted → ≈ 0; no permutation exceeded the observed statistic) | **YES** — affine ill-matched to sparse irregular series |
| Ovarian ultrasound (fuid / PCOSGen) | yes, AMH-linked reserve/PCOM phenotype (within-cohort AUROC ≈ 0.85; acquisition-conditioned, does not transfer) | **affine** (closed-form Gaussian-OT ≥ FM / OT-CFM / DDBM on held-out fidelity) | **NO** |
| Embryo morphology (Wang / Kromp) | no recoverable signal (7 encoders, distribution-free n.s.); grade encoded 0.83–0.96; clinical check AMH→oocyte yield ρ≈0.46 vs →grade ρ≈0.02 | n/a (signal absent) | **NO** |

The contribution is this map and its lesson: **AMH recoverability and the required model complexity are
modality-dependent; model complexity should be justified by data geometry and observation structure, not by
generative branding.** A neural generative model is load-bearing only where the data geometry demands it
(sparse irregular trajectories); near-affine image representations are adequately served by a closed-form map,
and embryo morphology carries no recoverable AMH signal. Earlier "generative-beats-discriminative" and
generative-OOD claims were **retracted** after controls showed a non-linear discriminative probe recovers the
same residual signal and the apparent OOD win was cross-dataset acquisition detection (see
`results/*` notes). Analyses are representation-level and within-cohort (distinct public cohorts; no
patient-matched or causal claim).

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
