# reserve_audit

Reusable audit harness for target-separated AMH-IVF recoverability checks.

## Install

From `code/repos/kromp-blastocyst-dataset-audit`:

```bash
python -m pip install -e .
```

The library is dependency-light: `numpy`, `pandas`, `scipy`, and `scikit-learn`.

## CLI

```bash
python -m reserve_audit.cli \
  --data X.csv \
  --target amh_tertile \
  --baseline age,sex \
  --task classification \
  --groups patient_id
```

By default, features are all columns except the target, baseline columns, and
group column. Use `--features col1,col2,...` to choose an explicit feature set.
The CLI prints one JSON result containing recoverability and an empirical
cluster-aware permutation p-value. Add `--sufficiency` to also compare a simple
model class with a gradient-boosted model using TOST equivalence.

## Python API

```python
from reserve_audit import recoverability

result = recoverability(
    X,
    y,
    baseline=age[:, None],
    task="classification",
    groups=patient_id,
    n_splits=5,
    seed=20260530,
)
```

Classification returns held-out predictive V-information as log-loss reduction
in bits and as a fraction of target entropy. With `baseline=...`, the statistic
is conditional recoverability beyond the baseline feature block. Regression
returns incremental held-out R2.

## Registry Schema

The target registry records the target semantics for every audit node:

```python
from reserve_audit.registry import TargetNode, TargetRegistry

registry = TargetRegistry([
    TargetNode(
        node_id="embryo_to_wang_amh",
        modality="Embryo image",
        encoder="DINOv2 ViT-B/14 CLS cached",
        target_name="Wang AMH tertile",
        target_kind="AMH-tertile",
        target_entropy=1.584962,
        baseline_vars=["age"],
    ),
])
```

Before producing a ranking, call `registry.assert_comparable(node_ids)`. It
raises if the requested nodes mix different `target_kind` values, enforcing the
target-separated atlas design rather than silently comparing unlike targets.
