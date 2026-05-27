import os
"""FAIR cross-cohort transfer re-test (is the 0.44-0.55 a real limit or a label/domain artifact?).

track2A33 trained fuid PCO-vs-(DF+Normal) and tested on PCOSGen PCOS-vs-healthy -> 0.44-0.55. Two confounds:
  (1) LABEL MISMATCH: fuid 'rest' includes DF (a NORMAL mid-cycle dominant follicle), so the axis was
      "PCO vs (PCO-absent incl. normal-follicle)", not "PCO vs healthy". Re-test with ALIGNED labels:
      fuid PCO(1) vs Normal(0) [drop DF], PCOSGen PCOS(1) vs healthy(0).
  (2) DOMAIN SHIFT: only per-cohort z-score was applied. Add CORAL (2nd-order statistic alignment).

Reuses cached features (results/diagnostics/cache/{fuid,pcosgen}_Z.npy) -> fast, no re-extraction.
Reports transfer AUROC under {z-only, CORAL} x {raw labels, aligned labels} both directions. No new deps."""
import glob, json, time
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = Path(os.environ.get("PROJECT_ROOT", ".")); OUT = R / "results/diagnostics"; CACHE = OUT / "cache"
SEED = 20260526; np.random.seed(SEED)

fZ = np.load(CACHE / "fuid_Z.npy")
# rebuild fuid labels in the SAME path order track2A33 used (sorted glob per class Normal,DF,PCO)
fb = R / "data/external/ovarian_us/fuid/extracted"; fy = []
for cls, lab in [("Normal", 0), ("Dominant_Follicle", 1), ("PCO", 2)]:
    for p in sorted(glob.glob(f"{fb}/**/{cls}/*", recursive=True)):
        if p.lower().endswith((".jpg", ".png", ".jpeg")): fy.append(lab)
fy = np.array(fy); assert len(fy) == len(fZ), (len(fy), len(fZ))

pZ = np.load(CACHE / "pcosgen_Z.npy")
pb = R / "data/external/ovarian_us/pcosgen/PCOSGen-train/PCOSGen-train"
lab = pd.read_excel(pb / "class_label.xlsx"); imgdir = pb / "images"; py = []
for _, r in lab.iterrows():
    if (imgdir / str(r["imagePath"])).exists(): py.append(int(r["Healthy"]))
py = np.array(py); p_pcos = (py == 0).astype(int); assert len(p_pcos) == len(pZ)
print(f"fuid {len(fZ)} (N{int((fy==0).sum())}/DF{int((fy==1).sum())}/PCO{int((fy==2).sum())})  PCOSGen {len(pZ)} (PCOS{int(p_pcos.sum())})")

def coral(Xs, Xt):
    """Align source covariance to target (CORAL): whiten source, recolor to target."""
    Cs = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]); Ct = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1])
    Ws = linalg.fractional_matrix_power(Cs, -0.5).real
    Wt = linalg.fractional_matrix_power(Ct, 0.5).real
    return (Xs - Xs.mean(0)) @ Ws @ Wt + Xt.mean(0)

def transfer(Xtr, ytr, Xte, yte, mode):
    if mode == "z":
        sctr = StandardScaler().fit(Xtr); scte = StandardScaler().fit(Xte)
        Atr, Ate = sctr.transform(Xtr), scte.transform(Xte)
    else:  # coral: align train onto test, then standardize by test stats
        Xtr2 = coral(Xtr, Xte); sc = StandardScaler().fit(Xte)
        Atr, Ate = sc.transform(Xtr2), sc.transform(Xte)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(Atr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Ate)[:, 1]))

# label sets
raw_f = (fy == 2).astype(int); raw_idx_f = np.arange(len(fy))                 # PCO vs (DF+Normal)
al_idx_f = np.where(np.isin(fy, [0, 2]))[0]; al_f = (fy[al_idx_f] == 2).astype(int)  # PCO vs Normal (drop DF)

res = {"spec": "fair cross-cohort transfer (aligned labels + CORAL); cached features"}
for lname, idx, yf in [("raw_labels(PCO_vs_DF+Normal)", raw_idx_f, raw_f),
                       ("aligned_labels(PCO_vs_Normal,dropDF)", al_idx_f, al_f)]:
    for mode in ["z", "coral"]:
        a = transfer(fZ[idx], yf, pZ, p_pcos, mode)
        b = transfer(pZ, p_pcos, fZ[idx], yf, mode)
        res[f"{lname}|{mode}"] = {"fuid->pcosgen": round(a, 3), "pcosgen->fuid": round(b, 3)}
        print(f"  {lname:38s} {mode:6s}  f->p {a:.3f}   p->f {b:.3f}")

# best aligned result for the headline
al = [v for k, v in res.items() if isinstance(v, dict) and "aligned" in k]
best = max(max(d["fuid->pcosgen"], d["pcosgen->fuid"]) for d in al)
res["best_aligned_transfer"] = best
res["verdict"] = ("aligned labels + CORAL materially recover transfer" if best >= 0.65 else
                  "transfer remains weak even with aligned labels + CORAL -> genuine cross-site domain shift (honest limitation)")
res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
print(f"\nbest aligned transfer = {best:.3f}  => {res['verdict']}")
json.dump(res, open(OUT / "crosscohort_fair.json", "w"), indent=2)
print("saved", OUT / "crosscohort_fair.json")
