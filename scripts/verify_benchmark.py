"""Cross-check public benchmark values against aggregate diagnostics JSON.

The script reads README.md and results/diagnostics/unified_encoder_benchmark.json
and reports whether the rounded positive-control R/H values shown publicly are
backed by the JSON. It does not require raw cohort data or embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DIAGNOSTIC = json.loads((ROOT / "results/diagnostics/unified_encoder_benchmark.json").read_text(encoding="utf-8"))


def present(value: float) -> bool:
    return f"{value:.4f}" in README


def main() -> None:
    missing: list[tuple[str, str]] = []

    print("Embryo morphology to AMH-tertile R/H")
    emb = DIAGNOSTIC["embryo_amh"]["encoders"]
    null_like = True
    for name, metrics in emb.items():
        rh = metrics.get("R/H")
        if rh is None:
            print(f"  {name}: status={metrics.get('status')}")
            continue
        if rh > 0.001:
            null_like = False
        print(f"  {name:20s} R/H={rh:+.4f}")
    print(f"  null_like={null_like}")

    print("\nOvary-US phenotype positive control")
    for name, metrics in DIAGNOSTIC["ovary_fuid_phenotype"]["encoders"].items():
        rh = metrics.get("R/H")
        if rh is None:
            print(f"  {name}: status={metrics.get('status')}")
            continue
        ci = metrics.get("bootstrap_CI")
        in_readme = present(float(rh))
        if not in_readme:
            missing.append((name, f"{rh:.4f}"))
        print(f"  {name:10s} R/H={rh:.4f} CI=[{ci[0]:.4f},{ci[1]:.4f}] in_readme={in_readme}")

    if missing:
        print("\nMissing rounded values from README:")
        for name, value in missing:
            print(f"  {name}: {value}")
        raise SystemExit(1)

    print("\nAll rounded public benchmark values are present in README.md.")


if __name__ == "__main__":
    main()
