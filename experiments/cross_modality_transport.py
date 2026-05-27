import os
"""Unified distribution-transport framework across modalities (FM/OT/DDBM x modality).
Apply OT(sliced-W) + DDBM-bridge to EMBRYO(Wang DINOv2) and ENDOCRINE(Brigham E2) feature distributions,
AMH high/low strata, to complete the 3-method x 3-modality matrix (ovary already done track2A25/26).
Expect: endocrine strong (FM d=1.14), embryo null. permutation nulls. No new deps."""
import sys,json,time,math
from pathlib import Path
import numpy as np,torch,torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
R=Path(os.environ.get("PROJECT_ROOT", ".")); SEED=20260526; np.random.seed(SEED); torch.manual_seed(SEED)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0,str(R/"code/repos/kromp-blastocyst-dataset-audit/scripts/synth"))

def sliced_w(A,B,D,np_=300,seed=SEED):
    g=np.random.default_rng(seed); P=g.standard_normal((D,np_)); P/=np.linalg.norm(P,axis=0,keepdims=True)
    qa=np.sort(A@P,0); qb=np.sort(B@P,0); m=min(len(A),len(B)); q=np.linspace(0,1,m)
    return float(np.abs(np.quantile(qa,q,axis=0)-np.quantile(qb,q,axis=0)).mean())
class Net(nn.Module):
    def __init__(s,d,h=128): super().__init__(); s.net=nn.Sequential(nn.Linear(d+1,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.net(torch.cat([x,t[:,None]],-1))
def bridge_d(Z,y,D,epochs=250,sig=0.3):
    torch.manual_seed(SEED); m=Net(D).to(dev); opt=torch.optim.AdamW(m.parameters(),1e-3)
    x0=torch.from_numpy(Z[y==0]).float().to(dev); x1=torch.from_numpy(Z[y==1]).float().to(dev)
    for e in range(epochs):
        i0=torch.randint(0,len(x0),(128,),device=dev); i1=torch.randint(0,len(x1),(128,),device=dev)
        a=x0[i0]; b=x1[i1]; t=torch.rand(128,device=dev)
        xt=(1-t)[:,None]*a+t[:,None]*b+sig*torch.sqrt((t*(1-t)).clamp(min=0))[:,None]*torch.randn_like(a)
        loss=((m(xt,t)-b)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x=x0.clone(); steps=50; dt=1/steps
        for k in range(steps):
            t=torch.full((len(x),),(k+.5)*dt,device=dev); x=x+(m(x,t)-x)/max(1-(k+.5)*dt,1e-2)*dt
        trans=x.cpu().numpy()
    df=Z[y==0]; lda=LinearDiscriminantAnalysis().fit(np.vstack([Z[y==1],df]),[1]*int((y==1).sum())+[0]*int((y==0).sum()))
    a=lda.transform(trans)[:,0]; b=lda.transform(df)[:,0]; psd=math.sqrt((a.var()+b.var())/2)+1e-9
    return abs(float((a.mean()-b.mean())/psd))
def run(name,Z,y):
    D=Z.shape[1]
    sw=sliced_w(Z[y==1],Z[y==0],D); rng=np.random.default_rng(SEED)
    swn=np.array([sliced_w(Z[(p:=rng.permutation(y))==1],Z[p==0],D,seed=SEED+k) for k in range(150)])
    dd=bridge_d(Z,y,D); ddn=np.array([bridge_d(Z,np.random.default_rng(SEED+k).permutation(y),D,epochs=150) for k in range(10)])
    r={"n":int(len(y)),"ot_sw":sw,"ot_p":float((swn>=sw).mean()),"ddbm_d":dd,"ddbm_p":float((ddn>=dd).mean())}
    print(f"[{name}] OT sw={sw:.3f} p={r['ot_p']:.3f} | DDBM d={dd:.3f} p={r['ddbm_p']:.3f}")
    return r
res={}
# EMBRYO: Wang DINOv2-giant features + AMH high/low
import pandas as pd
f=np.load(R/"results/synth/track2_dinov2_giant_encoder/dinov2_giant_features.npz",allow_pickle=True)
Zw=f["wang"].astype(np.float64); idx=np.array([str(x) for x in f["wang_index"]])
amh=pd.read_csv(R/"data/external/mendeley_wang_2026/amh_recovered_wang.csv"); amap=dict(zip(amh["Image"].astype(str),amh["AMH_raw"].astype(float)))
av=np.array([amap.get(i,np.nan) for i in idx]); keep=~np.isnan(av)&(av>0); Zw,av=Zw[keep],av[keep]
Zw=PCA(32,random_state=SEED).fit_transform(StandardScaler().fit_transform(Zw)).astype(np.float32)
yw=(av>=np.median(av)).astype(int)
res["embryo_Wang_AMH"]=run("embryo Wang AMH",Zw,yw)
# ENDOCRINE: Brigham E2 trajectory (standardized) + AMH high/low
from brigham_trajectory_loader import load_brigham_trajectory,standardize_for_training
d=load_brigham_trajectory(min_obs=5); std=standardize_for_training(d)
Zt=std["X_std"].astype(np.float32); amh_z=std["amh_z"]
yt=(amh_z>=np.median(amh_z)).astype(int)
res["endocrine_Brigham_AMH"]=run("endocrine Brigham AMH",Zt,yt)
res["_note"]="Transport matrix (OT+DDBM) across modalities; ovary in track2A25/26."; res["_ts"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
json.dump(res,open(R/"results/diagnostics/transport_matrix.json","w"),indent=2); print("saved transport_matrix.json")
