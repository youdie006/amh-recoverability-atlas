import os
"""Replicate the generative-vs-discriminative dissociation on a SECOND ovarian cohort.

The thesis headline (Ch1/Ch3) rests on ONE arm (fuid): under metadata residualization the discriminative
probe collapses while the OT-CFM generative transport survives. To answer "is your central claim n=1?",
re-run the SAME residualization protocol on the independent PCOSGen cohort (PCOS vs healthy, 300x300 uniform):
  - metadata-only baseline AUROC
  - raw probe AUROC vs metadata-residualized probe AUROC  (expect drop)
  - OT-CFM transport d raw vs residualized + permutation null  (expect survives)
If the pattern repeats, the dissociation is a 2-cohort finding, not a single-dataset quirk.

Reuses cached PCOSGen features (results/diagnostics/cache/pcosgen_Z.npy). No GPU encoder needed. No new deps."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = Path(os.environ.get("PROJECT_ROOT", ".")); OUT = R / "results/diagnostics"; CACHE = OUT / "cache"
SEED = 20260526; np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z = np.load(CACHE / "pcosgen_Z.npy")
pb = R / "data/external/ovarian_us/pcosgen/PCOSGen-train/PCOSGen-train"
lab = pd.read_excel(pb / "class_label.xlsx"); imgdir = pb / "images"
paths, y = [], []
for _, r in lab.iterrows():
    p = imgdir / str(r["imagePath"])
    if p.exists(): paths.append(str(p)); y.append(int(r["Healthy"]))
y = np.array(y); pcos = (y == 0).astype(int)   # 1 = PCOS (positive)
assert len(paths) == len(Z), (len(paths), len(Z))

# metadata in the SAME order as cached features
M = []
for p in paths:
    im = Image.open(p); w, h = im.size
    g = np.asarray(im.convert("L").resize((128, 128)), np.float32) / 255.0
    hist, _ = np.histogram(g, bins=8, range=(0, 1), density=True)
    M.append([w, h, w / h, g.mean(), g.std(), *hist])
M = np.array(M, np.float32)
print(f"PCOSGen n={len(paths)} PCOS={int(pcos.sum())} healthy={int((pcos==0).sum())} meta_dim={M.shape[1]} dev={dev}")

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
def cv_auroc(X, yy):
    a = []
    for tr, te in skf.split(X, yy):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc.transform(X[tr]), yy[tr])
        a.append(roc_auc_score(yy[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(a))
def residualize(X, Mcols):
    Ms = StandardScaler().fit_transform(Mcols)
    return X - LinearRegression().fit(Ms, X).predict(Ms)

meta_auc = cv_auroc(M, pcos)
raw_auc = cv_auroc(Z, pcos)
resid_full = residualize(Z, M)
res_auc = cv_auroc(resid_full, pcos)
print(f"[probe] metadata-only {meta_auc:.3f} | raw {raw_auc:.3f} | residualized {res_auc:.3f}")

# ---- OT-CFM transport: raw vs residualized, with permutation null ----
def otcfm_transport_d(feat, labels, seed=SEED, perms=30):
    Zp = PCA(32, random_state=seed).fit_transform(StandardScaler().fit_transform(feat)).astype(np.float32)
    D = Zp.shape[1]; X0, X1 = Zp[labels == 0], Zp[labels == 1]
    lda = LinearDiscriminantAnalysis().fit(Zp, labels)
    def rd(gen, ref):
        a = lda.transform(gen).ravel(); b = lda.transform(ref).ravel()
        sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
        return float((a.mean()-b.mean())/(sp+1e-8))
    class V(nn.Module):
        def __init__(s):
            super().__init__(); s.n = nn.Sequential(nn.Linear(D+1,128), nn.SiLU(), nn.Linear(128,128), nn.SiLU(), nn.Linear(128,D))
        def forward(s,x,t): return s.n(torch.cat([x,t],1))
    def couple(a,b,rng,mb=200):
        # minibatch-OT (Tong 2023): Hungarian on a fixed-size minibatch, not the full set (O(n^3) safe)
        n=min(len(a),len(b),mb); A=a[rng.choice(len(a),n,False)]; B=b[rng.choice(len(b),n,False)]
        C=((A[:,None,:]-B[None,:,:])**2).sum(-1); ri,ci=linear_sum_assignment(C); return A[ri],B[ci]
    def train_int(Xa,Xb,sd,ep=500):
        torch.manual_seed(sd); rng=np.random.default_rng(sd); v=V().to(dev); opt=torch.optim.Adam(v.parameters(),1e-3)
        for _ in range(ep):
            a,b=couple(Xa,Xb,rng); a=torch.tensor(a,device=dev); b=torch.tensor(b,device=dev)
            t=torch.rand(len(a),1,device=dev); xt=(1-t)*a+t*b
            loss=((v(xt,t)-(b-a))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            x=torch.tensor(X0,device=dev)
            for k in range(50): x=x+v(x,torch.full((len(x),1),k/50,device=dev))/50
        return x.cpu().numpy()
    d=rd(train_int(X0,X1,seed),X0)
    rng=np.random.default_rng(seed); nulls=[]
    for k in range(perms):
        yp=rng.permutation(labels); g=train_int(Zp[yp==0],Zp[yp==1],seed+k,ep=250); nulls.append(abs(rd(g,Zp[yp==0])))
    nulls=np.array(nulls); return d, float((nulls>=abs(d)).mean()), float(nulls.mean())

d_raw, p_raw, n_raw = otcfm_transport_d(Z, pcos)
d_res, p_res, n_res = otcfm_transport_d(resid_full, pcos)
print(f"[OT-CFM] raw d={d_raw:.3f} (p={p_raw:.3f}) | residualized d={d_res:.3f} (p={p_res:.3f}, null {n_res:.3f})")

res = {
    "spec": "PCOSGen replication of generative-vs-discriminative dissociation (2nd ovarian cohort)",
    "cohort": "PCOSGen PCOS-vs-healthy", "n": len(paths),
    "probe_metadata_only_auroc": meta_auc, "probe_raw_auroc": raw_auc, "probe_residualized_auroc": res_auc,
    "probe_drop": round(raw_auc - res_auc, 3),
    "otcfm_raw_d": d_raw, "otcfm_raw_perm_p": p_raw,
    "otcfm_residualized_d": d_res, "otcfm_residualized_perm_p": p_res, "otcfm_residualized_null_mean": n_res,
    "dissociation_replicated": bool((raw_auc - res_auc) > 0.05 and d_res > n_res * 1.5 and p_res < 0.05),
    "note": "If probe drops under residualization but OT-CFM transport survives (p<0.05), the generative>"
            "discriminative robustness dissociation is replicated on a 2nd independent ovarian cohort.",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(res, open(OUT / "pcosgen_dissociation.json", "w"), indent=2)
print(f"\ndissociation_replicated = {res['dissociation_replicated']}")
print("saved", OUT / "pcosgen_dissociation.json")
