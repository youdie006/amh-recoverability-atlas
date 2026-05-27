import os
"""GPT-Pro Round-2 required robustness: metadata residualization + localization null.

Defends two lethal reviewer attacks before submission (mask-overlap #1 needs manual ROI -> deferred):
  (R2) ARTIFACT: (a) metadata-only baseline (resolution/aspect/intensity-hist) predicting PCO-vs-DF
       -> should be ~chance for the size-matched contrast; (b) residualize metadata out of DINOv2
       features (per-dim linear regression), re-run probe AUROC + OT-CFM transport -> signal must survive.
  (LOC-null) participation-ratio 0.727 needs a NULL: compare to label-shuffle, random-probe-direction,
       and center-prior baselines -> report percentile so "spatially concentrated" is defensible.

PCO vs Dominant_Follicle, both native ~512x728 (no size leakage). DINOv2-with-registers-base.
Reuses track2A30/2A31 protocol. No new deps."""
import glob, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = Path(os.environ.get("PROJECT_ROOT", ".")); OUT = R / "results/diagnostics"
SEED = 20260526; np.random.seed(SEED); torch.manual_seed(SEED)
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = R / "data/external/ovarian_us/fuid/extracted"

paths, y, meta = [], [], []
for cls, lab in [("PCO", 1), ("Dominant_Follicle", 0)]:
    for p in glob.glob(f"{base}/**/{cls}/*", recursive=True):
        if not p.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        try:
            im = Image.open(p); w, h = im.size
            if h == 728 and 510 <= w <= 517:
                g = np.asarray(im.convert("L").resize((128, 128)), np.float32) / 255.0
                hist, _ = np.histogram(g, bins=8, range=(0, 1), density=True)
                paths.append(p); y.append(lab)
                meta.append([w, h, w / h, g.mean(), g.std(), *hist])
        except Exception:
            pass
y = np.array(y); M = np.array(meta, np.float32)
print(f"PCO-vs-DF: n={len(paths)} PCO={int(y.sum())} DF={int((1-y).sum())}  meta_dim={M.shape[1]}  dev={dev}")

def prep(p):
    im = Image.open(p).convert("RGB").resize((224, 224))
    a = (np.asarray(im).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1)).float()

enc = AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(dev).eval()
CLS, PATCH = [], []
with torch.no_grad():
    for i in range(0, len(paths), 16):
        out = enc(torch.stack([prep(p) for p in paths[i:i+16]]).to(dev))
        h = out.last_hidden_state; seq = h.shape[1]; off = seq - 256
        CLS.append(h[:, 0].cpu().numpy()); PATCH.append(h[:, off:].cpu().numpy())
del enc
if dev.type == "cuda":
    torch.cuda.empty_cache()
CLS = np.concatenate(CLS); PATCH = np.concatenate(PATCH)

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
def cv_auroc(Z):
    a = []
    for tr, te in skf.split(Z, y):
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc.transform(Z[tr]), y[tr])
        a.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(Z[te]))[:, 1]))
    return float(np.mean(a)), float(np.std(a))

# --- (R2a) metadata-only baseline ---
meta_auc, meta_sd = cv_auroc(M)
# --- (R2b) residualize metadata out of CLS features, re-probe ---
Msc = StandardScaler().fit_transform(M)
resid = CLS - LinearRegression().fit(Msc, CLS).predict(Msc)
raw_auc, raw_sd = cv_auroc(CLS)
res_auc, res_sd = cv_auroc(resid)
print(f"[R2a] metadata-only AUROC = {meta_auc:.3f}+/-{meta_sd:.3f} (chance 0.5)")
print(f"[R2b] raw-feature AUROC = {raw_auc:.3f}; residualized AUROC = {res_auc:.3f} (signal survives if ~unchanged)")

# --- residualized OT-CFM transport (signal must survive metadata removal) ---
Zr = PCA(32, random_state=SEED).fit_transform(StandardScaler().fit_transform(resid)).astype(np.float32)
D = Zr.shape[1]; X0, X1 = Zr[y == 0], Zr[y == 1]
lda = LinearDiscriminantAnalysis().fit(Zr, y)
def reserve_d(gen, ref):
    a = lda.transform(gen).ravel(); b = lda.transform(ref).ravel()
    sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / (sp + 1e-8))
class V(nn.Module):
    def __init__(s):
        super().__init__(); s.net = nn.Sequential(nn.Linear(D+1,128), nn.SiLU(), nn.Linear(128,128), nn.SiLU(), nn.Linear(128,D))
    def forward(s, x, t): return s.net(torch.cat([x, t], 1))
def couple_ot(a, b, rng):
    n = min(len(a), len(b)); A = a[rng.choice(len(a), n, False)]; B = b[rng.choice(len(b), n, False)]
    C = ((A[:,None,:]-B[None,:,:])**2).sum(-1); ri, ci = linear_sum_assignment(C); return A[ri], B[ci]
def train_int(Xa, Xb, seed, ep=600):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); v = V().to(dev); opt = torch.optim.Adam(v.parameters(), 1e-3)
    for _ in range(ep):
        a, b = couple_ot(Xa, Xb, rng); a = torch.tensor(a, device=dev); b = torch.tensor(b, device=dev)
        t = torch.rand(len(a),1,device=dev); xt = (1-t)*a + t*b
        loss = ((v(xt,t)-(b-a))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = torch.tensor(X0, device=dev)
        for k in range(50): x = x + v(x, torch.full((len(x),1), k/50, device=dev))/50
    return x.cpu().numpy()
gen = train_int(X0, X1, SEED); d_res = reserve_d(gen, X0)
rng = np.random.default_rng(SEED); nulls = []
for k in range(30):
    yp = rng.permutation(y); g = train_int(Zr[yp==0], Zr[yp==1], SEED+k, ep=300); nulls.append(abs(reserve_d(g, Zr[yp==0])))
nulls = np.array(nulls); otcfm_res_p = float((nulls >= abs(d_res)).mean())
print(f"[R2b] residualized OT-CFM transport d = {d_res:.3f}  perm_p = {otcfm_res_p:.3f} (null mean {nulls.mean():.3f})")

# --- (LOC-null) participation-ratio null for patch localization ---
def part_ratio_from_dirvec(w):
    ps = (PATCH.reshape(-1, PATCH.shape[2]) @ w).reshape(len(PATCH), -1)
    pm = (ps - ps.mean(1, keepdims=True)) / (ps.std(1, keepdims=True) + 1e-8)
    diff = pm[y == 1].mean(0) - pm[y == 0].mean(0)
    a = np.abs(diff); a = a / (a.sum() + 1e-12)
    return float(1.0 / (np.sum(a**2) * len(a)))
sc = StandardScaler().fit(CLS)
w_real = (LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc.transform(CLS), y).coef_[0]) / sc.scale_
pr_real = part_ratio_from_dirvec(w_real)
rng = np.random.default_rng(SEED)
# null A: random probe directions
pr_rand = [part_ratio_from_dirvec(rng.standard_normal(CLS.shape[1])) for _ in range(200)]
# null B: label-shuffle probe directions
pr_shuf = []
for k in range(100):
    ys = rng.permutation(y)
    ws = (LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5).fit(sc.transform(CLS), ys).coef_[0]) / sc.scale_
    # recompute diff under shuffled labels too
    ps = (PATCH.reshape(-1, PATCH.shape[2]) @ ws).reshape(len(PATCH), -1)
    pm = (ps - ps.mean(1, keepdims=True)) / (ps.std(1, keepdims=True) + 1e-8)
    diff = pm[ys == 1].mean(0) - pm[ys == 0].mean(0); a = np.abs(diff); a = a/(a.sum()+1e-12)
    pr_shuf.append(float(1.0/(np.sum(a**2)*len(a))))
pr_rand = np.array(pr_rand); pr_shuf = np.array(pr_shuf)
pctl_rand = float((pr_rand <= pr_real).mean()); pctl_shuf = float((pr_shuf <= pr_real).mean())
print(f"[LOC-null] real participation ratio = {pr_real:.3f}")
print(f"  random-direction null mean = {pr_rand.mean():.3f}; real is at {pctl_rand*100:.1f} pctl (lower=more concentrated)")
print(f"  label-shuffle null mean = {pr_shuf.mean():.3f}; real at {pctl_shuf*100:.1f} pctl")

res = {
    "spec": "GPT-Pro R2 robustness: metadata residualization + localization participation-ratio null",
    "contrast": "PCO vs DF size-matched (~512x728)", "n": len(paths),
    "metadata_only_auroc": meta_auc, "metadata_only_sd": meta_sd,
    "raw_feature_auroc": raw_auc, "residualized_feature_auroc": res_auc,
    "signal_survives_residualization": bool(res_auc > 0.75),
    "residualized_otcfm_transport_d": d_res, "residualized_otcfm_perm_p": otcfm_res_p,
    "participation_ratio_real": pr_real,
    "participation_null_random_mean": float(pr_rand.mean()), "participation_pctl_vs_random": pctl_rand,
    "participation_null_shuffle_mean": float(pr_shuf.mean()), "participation_pctl_vs_shuffle": pctl_shuf,
    "localized_vs_null": bool(pctl_rand < 0.05 and pctl_shuf < 0.05),
    "note": "metadata-only ~chance + signal survives residualization => not a resolution/intensity artifact. "
            "participation ratio below random/shuffle nulls => patch-token spatial concentration is non-random. "
            "Mask-overlap (manual ROI) deferred to manual annotation step.",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(res, open(OUT / "ovary_residualize_locnull.json", "w"), indent=2)
print("saved", OUT / "ovary_residualize_locnull.json")
