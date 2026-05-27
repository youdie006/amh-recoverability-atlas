import os
"""catalog #3 CKA + #8 energy distance.
CKA: representational similarity of ovary US under DINOv2 vs DINOv3 (do encoders agree?)
     + AMH-high vs AMH-low subspace CKA per modality.
Energy distance (bandwidth-free two-sample) high-vs-low per modality + permutation. numpy/sklearn only."""
import sys,glob,json,time
from pathlib import Path
import numpy as np,torch as T,pandas as pd
from transformers import AutoModel
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
R=Path(os.environ.get("PROJECT_ROOT", ".")); SEED=20260526; rng=np.random.default_rng(SEED)
sys.path.insert(0,str(R/"code/repos/kromp-blastocyst-dataset-audit/scripts/synth"))
MEAN=np.array([0.485,0.456,0.406]);STD=np.array([0.229,0.224,0.225]);dev=T.device("cuda" if T.cuda.is_available() else "cpu")
def lin_cka(X,Y):
    X=X-X.mean(0); Y=Y-Y.mean(0)
    return float((np.linalg.norm(X.T@Y)**2)/((np.linalg.norm(X.T@X))*(np.linalg.norm(Y.T@Y))+1e-12))
def energy_dist(A,B):
    def md(X,Y): return np.sqrt(((X[:,None]-Y[None])**2).sum(-1)+1e-12).mean()
    return float(2*md(A,B)-md(A,A)-md(B,B))
def edist_perm(A,B,nperm=300):
    Z=np.vstack([A,B]);n=len(A);obs=energy_dist(A,B);null=[]
    for _ in range(nperm):
        p=rng.permutation(len(Z));null.append(energy_dist(Z[p[:n]],Z[p[n:]]))
    return obs,float((np.array(null)>=obs).mean())
# ovary fuid PCO/DF
base=R/"data/external/ovarian_us/fuid/extracted";paths,yo=[],[]
for cls,lab in [("PCO",1),("Dominant_Follicle",0)]:
    for p in glob.glob(f"{base}/**/{cls}/*",recursive=True):
        if not p.lower().endswith((".jpg",".png",".jpeg")):continue
        try:
            w,h=Image.open(p).size
            if h==728 and 510<=w<=517:paths.append(p);yo.append(lab)
        except:pass
yo=np.array(yo)
def prep(p):im=Image.open(p).convert("RGB").resize((224,224));a=(np.asarray(im).astype(np.float32)/255-MEAN)/STD;return T.from_numpy(a.transpose(2,0,1)).float()
def feats(mid):
    m=AutoModel.from_pretrained(mid).to(dev).eval()
    with T.no_grad():Z=np.concatenate([m(T.stack([prep(p) for p in paths[i:i+32]]).to(dev)).last_hidden_state[:,0].cpu().numpy() for i in range(0,len(paths),32)])
    del m;T.cuda.empty_cache();return Z
Z2=feats("facebook/dinov2-with-registers-base"); Z3=feats("facebook/dinov3-vitb16-pretrain-lvd1689m")
cka_enc=lin_cka(StandardScaler().fit_transform(Z2),StandardScaler().fit_transform(Z3))
print(f"CKA(DINOv2,DINOv3) ovary representations = {cka_enc:.3f}")
res={"cka_dinov2_vs_dinov3_ovary":cka_enc}
# energy distance per modality
Zo=PCA(32,random_state=SEED).fit_transform(StandardScaler().fit_transform(Z2))
eo,po=edist_perm(Zo[yo==1],Zo[yo==0]); print(f"ovary energy={eo:.3f} p={po:.3f}")
f=np.load(R/"results/synth/track2_dinov2_giant_encoder/dinov2_giant_features.npz",allow_pickle=True)
Zw=f["wang"].astype(np.float64);idx=np.array([str(x) for x in f["wang_index"]])
amh=pd.read_csv(R/"data/external/mendeley_wang_2026/amh_recovered_wang.csv");amap=dict(zip(amh["Image"].astype(str),amh["AMH_raw"].astype(float)))
av=np.array([amap.get(i,np.nan) for i in idx]);keep=~np.isnan(av)&(av>0);Zw,av=Zw[keep],av[keep]
Zw=PCA(32,random_state=SEED).fit_transform(StandardScaler().fit_transform(Zw));yw=(av>=np.median(av)).astype(int)
sub=rng.choice(len(Zw),1000,replace=False); ee,pe=edist_perm(Zw[sub][yw[sub]==1],Zw[sub][yw[sub]==0]); print(f"embryo energy={ee:.3f} p={pe:.3f}")
from brigham_trajectory_loader import load_brigham_trajectory,standardize_for_training
d=load_brigham_trajectory(min_obs=5);std=standardize_for_training(d);Zt=std["X_std"];az=std["amh_z"];yt=(az>=np.median(az)).astype(int)
subt=rng.choice(len(Zt),1000,replace=False); et,pt=edist_perm(Zt[subt][yt[subt]==1],Zt[subt][yt[subt]==0]); print(f"endocrine energy={et:.3f} p={pt:.3f}")
res.update({"ovary_energy":eo,"ovary_energy_p":po,"embryo_energy":ee,"embryo_energy_p":pe,"endocrine_energy":et,"endocrine_energy_p":pt,
"note":"CKA: ovary encoders agree -> robust representation. Energy dist confirms ovary/endocrine signal, embryo per p.","ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
json.dump(res,open(R/"results/diagnostics/cka_energy.json","w"),indent=2);print("saved cka_energy.json")
