"""Replay per-attack statistics from the locked raw results (assistant-prepared check)."""
import json,numpy as np,pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
W=str(ROOT/'reference')+'/'
OUT=ROOT/'verified_outputs/per_attack'; OUT.mkdir(parents=True,exist_ok=True)
RAW={'Main':W+'main_results/revised_pipeline_raw.json','B3':W+'b3_results/revised_pipeline_b3_raw.json'}
def holm(p):
    p=np.asarray(p);o=np.argsort(p,kind='stable');m=len(p);adj=np.empty(m);run=0
    for r,i in enumerate(o): run=max(run,min(1,(m-r)*p[i]));adj[i]=run
    return adj
rows=[]
for design,path in RAW.items():
    raw=json.load(open(path))
    for model in ['rf','xgb','mlp']:
        for cls in ['DoS','Bot','Infiltration']:
            c1=np.array([raw[str(s)]['results'][model]['C1']['per_class_f1'][cls] for s in range(10)])
            c2=np.array([raw[str(s)]['results'][model]['per_variant']['hybrid768']['C2']['per_class_f1'][cls] for s in range(10)])
            c3=np.array([raw[str(s)]['results'][model]['per_variant']['hybrid768']['C3']['per_class_f1'][cls] for s in range(10)])
            d=c3-c1;n=10;md=d.mean();sd=d.std(ddof=1);se=sd/np.sqrt(n);t=stats.t.ppf(.95,9)
            lo,hi=md-t*se,md+t*se
            pl=stats.t.sf((md+.01)/se,9);pu=stats.t.cdf((md-.01)/se,9)
            d2=c2-c1
            rows.append(dict(design=design,model=model,attack=cls,c1=c1.mean(),c2=c2.mean(),c3=c3.mean(),c2_c1=d2.mean(),c3_c1=md,ci90_lo=lo,ci90_hi=hi,tost_raw=max(pl,pu),
               in_001=lo>-.01 and hi<.01,in_0005=lo>-.005 and hi<.005, c2_below_c1_splits=int((d2<0).sum()), sd_c1=c1.std(ddof=1)))
df=pd.DataFrame(rows)
df['tost_holm']=np.nan
for dsg in ['Main','B3']:
    k=df.design==dsg; df.loc[k,'tost_holm']=holm(df.loc[k,'tost_raw'].values)
df['equiv_001']=(df.tost_holm<.05)&df.in_001
df.to_csv(OUT/'per_attack_tost_replay.csv',index=False)
pd.set_option('display.width',250)
print(df.to_string(float_format=lambda x:f'{x:.6g}'))
