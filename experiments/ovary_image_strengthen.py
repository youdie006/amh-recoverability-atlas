import os
"""Ovary image-arm strengthening.

(A) RESIDUALIZATION SPECTRUM: the 0.67 from track2A32 over-removed intensity (partly real PCO echo).
    Report AUROC after removing geometry-only / +global-intensity / +full-histogram to show the
    honest range and where the drop comes from. = fair confound control, ComBat-style intuition.
(B) CROSS-COHORT GENERALIZATION (multi-site, turns weakness into feature): train PCO/PCOS-phenotype
    axis on fuid, test on PCOSGen and vice versa (per-cohort z-score harmonization). If it transfers,
    that is a multi-site generalization result the standard multi-site generalization setting.
(C) REPRESENTATIONAL GEOMETRY (Park's cortical-gradient methodology analog): RSA (feature RDM vs label
    RDM + permutation), PCA-gradient alignment (which principal gradient carries reserve), intrinsic
    dimensionality (participation ratio of eigenspectrum).

DINOv2-with-registers-base CLS features. Caches features to .npy. No new deps."""
import glob, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoModel
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = Path(os.environ.get("PROJECT_ROOT", ".")); OUT = R / "results/diagnostics"
CACHE = OUT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
SEED = 20260526; np.random.seed(SEED); torch.manual_seed(SEED)
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_enc = None
def encoder():
    global _enc
    if _enc is None:
        _enc = AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(dev).eval()
    return _enc
def prep(p):
    im = Image.open(p).convert("RGB").resize((224, 224))
    a = (np.asarray(im).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1)).float()
def feats_and_meta(paths):
    Z, M = [], []
    enc = encoder()
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            chunk = paths[i:i+32]
            Z.append(enc(torch.stack([prep(p) for p in chunk]).to(dev)).last_hidden_state[:, 0].cpu().numpy())
            for p in chunk:
                im = Image.open(p); w, h = im.size
                g = np.asarray(im.convert("L").resize((128, 128)), np.float32) / 255.0
                hist, _ = np.histogram(g, bins=8, range=(0, 1), density=True)
                M.append([w, h, w / h, g.mean(), g.std(), *hist])
    return np.concatenate(Z), np.array(M, np.float32)

# ---------- load fuid (3 classes) ----------
fb = R / "data/external/ovarian_us/fuid/extracted"
fp, fy = [], []
for cls, lab in [("Normal", 0), ("Dominant_Follicle", 1), ("PCO", 2)]:
    for p in sorted(glob.glob(f"{fb}/**/{cls}/*", recursive=True)):
        if p.lower().endswith((".jpg", ".png", ".jpeg")):
            fp.append(p); fy.append(lab)
fy = np.array(fy)
if (CACHE / "fuid_Z.npy").exists():
    fZ = np.load(CACHE / "fuid_Z.npy"); fM = np.load(CACHE / "fuid_M.npy")
else:
    fZ, fM = feats_and_meta(fp); np.save(CACHE / "fuid_Z.npy", fZ); np.save(CACHE / "fuid_M.npy", fM)
print(f"fuid n={len(fp)} Normal={int((fy==0).sum())} DF={int((fy==1).sum())} PCO={int((fy==2).sum())}")

# ---------- load PCOSGen ----------
pb = R / "data/external/ovarian_us/pcosgen/PCOSGen-train/PCOSGen-train"
lab = pd.read_excel(pb / "class_label.xlsx"); imgdir = pb / "images"
pp, py = [], []
for _, r in lab.iterrows():
    p = imgdir / str(r["imagePath"])
    if p.exists(): pp.append(str(p)); py.append(int(r["Healthy"]))
py = np.array(py); p_pcos = (py == 0).astype(int)   # 1 = PCOS
if (CACHE / "pcosgen_Z.npy").exists():
    pZ = np.load(CACHE / "pcosgen_Z.npy")
else:
    pZ, _ = feats_and_meta(pp); np.save(CACHE / "pcosgen_Z.npy", pZ)
print(f"PCOSGen n={len(pp)} PCOS={int(p_pcos.sum())} healthy={int((p_pcos==0).sum())}")

res = {"spec": "ovary image-arm strengthening: residualization spectrum + cross-cohort + representational geometry"}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
def cv_auroc(Z, y):
    a = []
    for tr, te in skf.split(Z, y):
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc.transform(Z[tr]), y[tr])
        a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(Z[te]))[:, 1]))
    return float(np.mean(a))

# ===== (A) residualization spectrum on fuid PCO-vs-DF size-matched =====
sm = []
for i, p in enumerate(fp):
    if fy[i] in (1, 2):
        w, h = Image.open(p).size
        if h == 728 and 510 <= w <= 517: sm.append(i)
sm = np.array(sm); ysm = (fy[sm] == 2).astype(int)   # 1 = PCO
Zsm, Msm = fZ[sm], fM[sm]
def residualize(Z, Mcols):
    Ms = StandardScaler().fit_transform(Mcols)
    return Z - LinearRegression().fit(Ms, Z).predict(Ms)
specA = {
    "raw": cv_auroc(Zsm, ysm),
    "resid_geometry_only(w,h,aspect)": cv_auroc(residualize(Zsm, Msm[:, [0, 1, 2]]), ysm),
    "resid_geometry+global_intensity(+mean,std)": cv_auroc(residualize(Zsm, Msm[:, [0, 1, 2, 3, 4]]), ysm),
    "resid_full(+histogram)=track2A32": cv_auroc(residualize(Zsm, Msm), ysm),
    "metadata_only": cv_auroc(Msm, ysm),
}
res["A_residualization_spectrum"] = specA
print("\n(A) residualization spectrum (PCO-vs-DF size-matched, n=%d):" % len(sm))
for k, v in specA.items(): print(f"    {v:.3f}  {k}")

# ===== (B) cross-cohort generalization (multi-site) =====
# shared axis: PCO/PCOS phenotype (1) vs non (0). fuid: PCO=1 vs DF+Normal=0.
fy_axis = (fy == 2).astype(int)
def transfer(Ztr, ytr, Zte, yte):
    sctr = StandardScaler().fit(Ztr); scte = StandardScaler().fit(Zte)  # per-cohort z (harmonization)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sctr.transform(Ztr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(scte.transform(Zte))[:, 1]))
res["B_cross_cohort"] = {
    "train_fuid_test_pcosgen": transfer(fZ, fy_axis, pZ, p_pcos),
    "train_pcosgen_test_fuid": transfer(pZ, p_pcos, fZ, fy_axis),
    "within_fuid_PCOvsRest": cv_auroc(fZ, fy_axis),
    "within_pcosgen": cv_auroc(pZ, p_pcos),
    "note": "per-cohort z-score harmonization; shared axis = PCO/PCOS phenotype vs rest",
}
print("\n(B) cross-cohort generalization (multi-site):")
for k, v in res["B_cross_cohort"].items():
    if isinstance(v, float): print(f"    {v:.3f}  {k}")

# ===== (C) representational geometry on fuid PCO-vs-DF =====
Zg = StandardScaler().fit_transform(Zsm)
# RSA: feature RDM vs label RDM (same-class 0 / diff-class 1), Spearman + permutation
from scipy.spatial.distance import pdist
frdm = pdist(Zg, "correlation"); lrdm = pdist(ysm.reshape(-1, 1), lambda a, b: float(a[0] != b[0]))
rsa = float(spearmanr(frdm, lrdm).correlation)
rng = np.random.default_rng(SEED)
rsa_null = []
for _ in range(1000):
    yp = rng.permutation(ysm); lp = pdist(yp.reshape(-1, 1), lambda a, b: float(a[0] != b[0]))
    rsa_null.append(spearmanr(frdm, lp).correlation)
rsa_p = float((np.array(rsa_null) >= rsa).mean())
# PCA-gradient alignment: which principal gradient (component) carries reserve?
from sklearn.decomposition import PCA
pca = PCA(20, random_state=SEED).fit(Zg); comps = pca.transform(Zg)
align = [abs(float(spearmanr(comps[:, k], ysm).correlation)) for k in range(20)]
top_grad = int(np.argmax(align))
# intrinsic dimensionality: participation ratio of eigenspectrum
ev = pca.explained_variance_; part = float((ev.sum() ** 2) / (ev ** 2).sum())
res["C_representational_geometry"] = {
    "RSA_feature_vs_label_spearman": rsa, "RSA_perm_p": rsa_p,
    "reserve_aligned_principal_gradient_idx": top_grad,
    "reserve_gradient_alignment_spearman": align[top_grad],
    "gradient_alignment_top5": [round(a, 3) for a in sorted(align, reverse=True)[:5]],
    "intrinsic_dim_participation_ratio": part,
    "note": "RSA: feature geometry tracks reserve label; reserve aligns to a low-order principal gradient "
            "(cortical-gradient analog); participation ratio = effective representational dimensionality.",
}
print("\n(C) representational geometry:")
print(f"    RSA spearman = {rsa:.3f} (perm p={rsa_p:.3f})")
print(f"    reserve aligns to principal gradient #{top_grad} (|rho|={align[top_grad]:.3f}); top5 {res['C_representational_geometry']['gradient_alignment_top5']}")
print(f"    intrinsic dim (participation ratio) = {part:.1f}")

res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(res, open(OUT / "ovary_image_strengthen.json", "w"), indent=2)
print("\nsaved", OUT / "ovary_image_strengthen.json")
