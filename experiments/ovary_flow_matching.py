import os
"""Conditional Flow Matching on OVARY-US feature DISTRIBUTION (not pixel gen).
User's true vision: use FM/ODE-SDE generative models to model the DATA DISTRIBUTION conditioned on
AMH-linked reserve. Same methodology as the trajectory FM (d=1.14), applied to ovary DINOv2 features.
Model P(feature | reserve); measure reserve-conditional generated distribution shift (Cohen's d on
LDA axis + sliced-Wasserstein) + permutation null (shuffle reserve, retrain). fuid PCO vs DF (clean).
No new deps (torch/sklearn)."""
import glob,json,time,math
from pathlib import Path
import numpy as np,torch,torch.nn as nn
from PIL import Image
from transformers import AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
R=Path(os.environ.get("PROJECT_ROOT", ".")); SEED=20260526
np.random.seed(SEED); torch.manual_seed(SEED)
MEAN=np.array([0.485,0.456,0.406]);STD=np.array([0.229,0.224,0.225])
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
base=R/"data/external/ovarian_us/fuid/extracted"
paths,y=[],[]
for cls,lab in [("PCO",1),("Dominant_Follicle",0)]:
    for p in glob.glob(f"{base}/**/{cls}/*",recursive=True):
        if not p.lower().endswith((".jpg",".png",".jpeg")): continue
        try:
            w,h=Image.open(p).size
            if h==728 and 510<=w<=517: paths.append(p); y.append(lab)
        except: pass
y=np.array(y,dtype=np.float32); print(f"n={len(paths)} PCO={int(y.sum())}")
def prep(p): 
    im=Image.open(p).convert("RGB").resize((224,224)); a=(np.asarray(im).astype(np.float32)/255-MEAN)/STD
    return torch.from_numpy(a.transpose(2,0,1)).float()
enc=AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(dev).eval()
with torch.no_grad():
    Z=np.concatenate([enc(torch.stack([prep(p) for p in paths[i:i+32]]).to(dev)).last_hidden_state[:,0].cpu().numpy() for i in range(0,len(paths),32)])
del enc; torch.cuda.empty_cache()
Z=StandardScaler().fit_transform(Z); Z=PCA(32,random_state=SEED).fit_transform(Z).astype(np.float32)  # 32-d feature space
D=Z.shape[1]
class VF(nn.Module):
    def __init__(s,d=D,h=128):
        super().__init__(); s.net=nn.Sequential(nn.Linear(d+2,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t,c): return s.net(torch.cat([x,t[:,None],c[:,None]],-1))
def train_fm(Zt,ct,epochs=300):
    torch.manual_seed(SEED); m=VF().to(dev); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-4)
    Zt=torch.from_numpy(Zt).to(dev); ct=torch.from_numpy(ct).float().to(dev); n=len(Zt)
    for e in range(epochs):
        idx=torch.randperm(n,device=dev)
        for i in range(0,n,64):
            b=idx[i:i+64]; x1=Zt[b]; x0=torch.randn_like(x1); t=torch.rand(len(b),device=dev)
            xt=(1-t)[:,None]*x0+t[:,None]*x1; tgt=x1-x0
            loss=((m(xt,t,ct[b])-tgt)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return m
@torch.no_grad()
def sample(m,n,c,steps=50):
    x=torch.randn(n,D,device=dev); cc=torch.full((n,),float(c),device=dev); dt=1/steps
    for k in range(steps):
        t=torch.full((n,),(k+.5)*dt,device=dev); x=x+m(x,t,cc)*dt
    return x.cpu().numpy()
def shift(m):
    hi=sample(m,400,1); lo=sample(m,400,0)
    lda=LinearDiscriminantAnalysis().fit(np.vstack([hi,lo]),[1]*400+[0]*400)
    ph=lda.transform(hi)[:,0]; pl=lda.transform(lo)[:,0]
    psd=math.sqrt((ph.var()+pl.var())/2)+1e-9
    return float((ph.mean()-pl.mean())/psd)
m=train_fm(Z,y); d_real=shift(m)
print(f"[real] reserve-conditional FM distribution shift Cohen's d = {d_real:.3f}")
dn=[]
for k in range(15):
    rng=np.random.default_rng(SEED+k); yp=rng.permutation(y)
    dn.append(abs(shift(train_fm(Z,yp,epochs=200))))
dn=np.array(dn); p=float((dn>=abs(d_real)).mean())
res={"method":"conditional FM on ovary DINOv2 features (PCA32), reserve-conditional distribution",
     "n":len(paths),"cohens_d_real":d_real,"null_absd_mean":float(dn.mean()),"null_absd_max":float(dn.max()),
     "perm_p":p,"verdict":"reserve signal present in ovary feature DISTRIBUTION via FM" if p<0.05 else "weak",
     "ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
print(f"[null] mean|d|={dn.mean():.3f} max|d|={dn.max():.3f}  perm p={p:.3f}")
json.dump(res,open(R/"results/diagnostics/ovary_feature_fm.json","w"),indent=2)
print("saved ovary_feature_fm.json")
