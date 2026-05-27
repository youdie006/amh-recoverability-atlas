import os
"""Information-localization battery (catalog #1,2,4,5): Kraskov MI + HSIC + MMD + C2ST.
Per modality, EFFECT-SIZE + HELD-OUT focused (large-n makes everything 'significant'; report magnitudes).
C2ST held-out accuracy = honest 'can a classifier separate AMH-high/low'. MI = info-content scalar.
Modalities: ovary US (PCO phenotype), embryo Wang (AMH), endocrine Brigham (AMH). numpy/sklearn only."""
import sys,glob,json,time
from pathlib import Path
import numpy as np,torch,pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import binomtest
R=Path(os.environ.get("PROJECT_ROOT", ".")); SEED=20260526; np.random.seed(SEED); rng=np.random.default_rng(SEED)
sys.path.insert(0,str(R/"code/repos/kromp-blastocyst-dataset-audit/scripts/synth"))
def rbf(X,Y,g): 
    d=((X[:,None]-Y[None])**2).sum(-1); return np.exp(-g*d)
def mmd2_perm(A,B,nperm=300):
    Z=np.vstack([A,B]); g=1.0/(np.median(((Z[:,None]-Z[None])**2).sum(-1))+1e-9)
    def mmd2(a,b): return rbf(a,a,g).mean()+rbf(b,b,g).mean()-2*rbf(a,b,g).mean()
    obs=mmd2(A,B); n=len(A); null=[]
    for _ in range(nperm):
        p=rng.permutation(len(Z)); null.append(mmd2(Z[p[:n]],Z[p[n:]]))
    return float(obs),float((np.array(null)>=obs).mean())
def hsic_perm(X,y,nperm=300):
    n=len(y); gx=1.0/(np.median(((X[:,None]-X[None])**2).sum(-1))+1e-9)
    K=rbf(X,X,gx); yy=y.reshape(-1,1).astype(float); L=(yy==yy.T).astype(float)
    H=np.eye(n)-1.0/n
    def hs(Lm): return np.trace(K@H@Lm@H)/(n*n)
    obs=hs(L); null=[]
    for _ in range(nperm):
        p=rng.permutation(n); yp=y[p].reshape(-1,1); Lp=(yp==yp.T).astype(float); null.append(hs(Lp))
    return float(obs),float((np.array(null)>=obs).mean())
def battery(name,X,ybin):
    X=np.asarray(X,float)
    # subsample for kernel methods if huge
    if len(X)>1200:
        idx=rng.choice(len(X),1200,replace=False); Xs,ys=X[idx],ybin[idx]
    else: Xs,ys=X,ybin
    c2st=cross_val_score(GradientBoostingClassifier(random_state=SEED),X,ybin,cv=5,scoring="accuracy").mean()
    bp=binomtest(int(round(c2st*len(ybin))),len(ybin),0.5,alternative="greater").pvalue
    mi=float(mutual_info_classif(X,ybin,random_state=SEED).mean())
    mmd,mmdp=mmd2_perm(Xs[ys==1],Xs[ys==0])
    hs,hsp=hsic_perm(Xs,ys)
    r={"n":int(len(X)),"C2ST_heldout_acc":float(c2st),"C2ST_p":float(bp),"MI_mean_nats":mi,
       "MMD2":mmd,"MMD_p":mmdp,"HSIC":hs,"HSIC_p":hsp}
    print(f"[{name}] C2ST acc={c2st:.3f}(p={bp:.1e}) MI={mi:.4f} MMD2={mmd:.4f}(p={mmdp:.3f}) HSIC p={hsp:.3f}")
    return r
res={}
# ovary
import torch as T
from transformers import AutoModel
from PIL import Image
MEAN=np.array([0.485,0.456,0.406]);STD=np.array([0.229,0.224,0.225]);dev=T.device("cuda" if T.cuda.is_available() else "cpu")
base=R/"data/external/ovarian_us/fuid/extracted"; paths,yo=[],[]
for cls,lab in [("PCO",1),("Dominant_Follicle",0)]:
    for p in glob.glob(f"{base}/**/{cls}/*",recursive=True):
        if not p.lower().endswith((".jpg",".png",".jpeg")): continue
        try:
            w,h=Image.open(p).size
            if h==728 and 510<=w<=517: paths.append(p);yo.append(lab)
        except: pass
yo=np.array(yo)
enc=AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(dev).eval()
def prep(p):im=Image.open(p).convert("RGB").resize((224,224));a=(np.asarray(im).astype(np.float32)/255-MEAN)/STD;return T.from_numpy(a.transpose(2,0,1)).float()
with T.no_grad(): Zo=np.concatenate([enc(T.stack([prep(p) for p in paths[i:i+32]]).to(dev)).last_hidden_state[:,0].cpu().numpy() for i in range(0,len(paths),32)])
del enc; T.cuda.empty_cache()
Zo=PCA(32,random_state=SEED).fit_transform(StandardScaler().fit_transform(Zo))
res["ovary_PCO"]=battery("ovary PCO",Zo,yo)
# embryo
f=np.load(R/"results/synth/track2_dinov2_giant_encoder/dinov2_giant_features.npz",allow_pickle=True)
Zw=f["wang"].astype(np.float64); idx=np.array([str(x) for x in f["wang_index"]])
amh=pd.read_csv(R/"data/external/mendeley_wang_2026/amh_recovered_wang.csv");amap=dict(zip(amh["Image"].astype(str),amh["AMH_raw"].astype(float)))
av=np.array([amap.get(i,np.nan) for i in idx]);keep=~np.isnan(av)&(av>0);Zw,av=Zw[keep],av[keep]
Zw=PCA(32,random_state=SEED).fit_transform(StandardScaler().fit_transform(Zw)); yw=(av>=np.median(av)).astype(int)
res["embryo_AMH"]=battery("embryo AMH",Zw,yw)
# endocrine
from brigham_trajectory_loader import load_brigham_trajectory,standardize_for_training
d=load_brigham_trajectory(min_obs=5);std=standardize_for_training(d);Zt=std["X_std"];az=std["amh_z"];yt=(az>=np.median(az)).astype(int)
res["endocrine_AMH"]=battery("endocrine AMH",Zt,yt)
res["_note"]="Within-modality effect-size + held-out C2ST (NOT cross-modality ranking; large-n inflates significance)."; res["_ts"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
json.dump(res,open(R/"results/diagnostics/info_localization.json","w"),indent=2);print("saved info_localization.json")
