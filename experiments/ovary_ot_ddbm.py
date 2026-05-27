import os
"""Ovary reserve axis as DISTRIBUTION TRANSPORT: OT + DDBM (user req: connect FM->DDBM/OT).
Frame reserve as transport P(feat|DF/low) -> P(feat|PCO/high) on ovary DINOv2 features (PCA32, clean fuid).
(1) OT: sliced-Wasserstein distance(DF,PCO) + permutation null.
(2) DDBM: Brownian-bridge denoiser DF->PCO (random coupling), reverse-sample, transport Cohen's d (LDA) + perm null.
Complements FM (track2A25 d=1.71). No new deps."""
import glob,json,time,math
from pathlib import Path
import numpy as np,torch,torch.nn as nn
from PIL import Image
from transformers import AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
R=Path(os.environ.get("PROJECT_ROOT", ".")); SEED=20260526; np.random.seed(SEED); torch.manual_seed(SEED)
MEAN=np.array([0.485,0.456,0.406]);STD=np.array([0.229,0.224,0.225]); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
base=R/"data/external/ovarian_us/fuid/extracted"; paths,y=[],[]
for cls,lab in [("PCO",1),("Dominant_Follicle",0)]:
    for p in glob.glob(f"{base}/**/{cls}/*",recursive=True):
        if not p.lower().endswith((".jpg",".png",".jpeg")): continue
        try:
            w,h=Image.open(p).size
            if h==728 and 510<=w<=517: paths.append(p); y.append(lab)
        except: pass
y=np.array(y)
def prep(p): 
    im=Image.open(p).convert("RGB").resize((224,224)); a=(np.asarray(im).astype(np.float32)/255-MEAN)/STD
    return torch.from_numpy(a.transpose(2,0,1)).float()
enc=AutoModel.from_pretrained("facebook/dinov2-with-registers-base").to(dev).eval()
with torch.no_grad():
    Z=np.concatenate([enc(torch.stack([prep(p) for p in paths[i:i+32]]).to(dev)).last_hidden_state[:,0].cpu().numpy() for i in range(0,len(paths),32)])
del enc; torch.cuda.empty_cache()
Z=PCA(32,random_state=SEED).fit_transform(StandardScaler().fit_transform(Z)).astype(np.float32); D=Z.shape[1]
def sliced_w(A,B,np_=300,seed=SEED):
    g=np.random.default_rng(seed); P=g.standard_normal((D,np_)); P/=np.linalg.norm(P,axis=0,keepdims=True)
    qa=np.sort(A@P,0); qb=np.sort(B@P,0); m=min(len(A),len(B)); q=np.linspace(0,1,m)
    return float(np.abs(np.quantile(qa,q,axis=0)-np.quantile(qb,q,axis=0)).mean())
# (1) OT sliced-Wasserstein
sw=sliced_w(Z[y==1],Z[y==0]); rng=np.random.default_rng(SEED); swn=[]
for k in range(300):
    yp=rng.permutation(y); swn.append(sliced_w(Z[yp==1],Z[yp==0],seed=SEED+k))
swn=np.array(swn); p_sw=float((swn>=sw).mean())
print(f"OT sliced-Wasserstein(PCO,DF)={sw:.4f} null={swn.mean():.4f} p={p_sw:.3f}")
# (2) DDBM Brownian-bridge DF->PCO
class Net(nn.Module):
    def __init__(s,d=D,h=128): super().__init__(); s.net=nn.Sequential(nn.Linear(d+1,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,d))
    def forward(s,x,t): return s.net(torch.cat([x,t[:,None]],-1))
def train_bridge(Zf,yf,epochs=300,sig=0.3):
    torch.manual_seed(SEED); m=Net().to(dev); opt=torch.optim.AdamW(m.parameters(),1e-3)
    x0=torch.from_numpy(Zf[yf==0]).to(dev); x1=torch.from_numpy(Zf[yf==1]).to(dev); n=min(len(x0),len(x1))
    for e in range(epochs):
        i0=torch.randint(0,len(x0),(128,),device=dev); i1=torch.randint(0,len(x1),(128,),device=dev)
        a=x0[i0]; b=x1[i1]; t=torch.rand(128,device=dev)
        xt=(1-t)[:,None]*a+t[:,None]*b+sig*torch.sqrt((t*(1-t)).clamp(min=0))[:,None]*torch.randn_like(a)
        loss=((m(xt,t)-b)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()  # predict PCO endpoint
    return m
@torch.no_grad()
def transport(m,Zf,yf,steps=50):
    x=torch.from_numpy(Zf[yf==0]).to(dev).float(); dt=1/steps
    for k in range(steps):
        t=torch.full((len(x),),(k+.5)*dt,device=dev); pred1=m(x,t); x=x+(pred1-x)/max(1-(k+.5)*dt,1e-2)*dt
    return x.cpu().numpy()
def bridge_d(Zf,yf):
    m=train_bridge(Zf,yf); trans=transport(m,Zf,yf); df=Zf[yf==0]
    lda=LinearDiscriminantAnalysis().fit(np.vstack([Zf[yf==1],df]),[1]*int((yf==1).sum())+[0]*int((yf==0).sum()))
    a=lda.transform(trans)[:,0]; b=lda.transform(df)[:,0]; psd=math.sqrt((a.var()+b.var())/2)+1e-9
    return float((a.mean()-b.mean())/psd)
d_real=bridge_d(Z,y); dn=[abs(bridge_d(Z,np.random.default_rng(SEED+k).permutation(y))) for k in range(12)]
dn=np.array(dn); p_dd=float((dn>=abs(d_real)).mean())
print(f"DDBM bridge DF->PCO transport Cohen's d={d_real:.3f} null|d| mean={dn.mean():.3f} max={dn.max():.3f} p={p_dd:.3f}")
res={"ot_sliced_wasserstein":sw,"ot_null_mean":float(swn.mean()),"ot_p":p_sw,
     "ddbm_transport_d":d_real,"ddbm_null_absd_mean":float(dn.mean()),"ddbm_p":p_dd,
     "note":"Reserve axis as distribution transport on ovary features: OT distance + DDBM bridge. Complements FM d=1.71.","ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
json.dump(res,open(R/"results/diagnostics/ovary_ddbm_ot.json","w"),indent=2); print("saved ovary_ddbm_ot.json")
