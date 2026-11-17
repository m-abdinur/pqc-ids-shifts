"""Replay per-attack statistics from the locked raw results (assistant-prepared check)."""
import json,numpy as np,pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
W=str(ROOT/'reference')+'/'
OUT=ROOT/'verified_outputs/per_attack'; OUT.mkdir(parents=True,exist_ok=True)
RAW={'Main':W+'main_results/revised_pipeline_raw.json','B3':W+'b3_results/revised_pipeline_b3_raw.json'}
def holm(p):
    p=np.asarray(p);o=np.argsort(p);m=len(p);adj=np.empty(m);run=0
    for r,i in enumerate(o): run=max(run,min(1,(m-r)*p[i]));adj[i]=run
    return adj
def test(v):
    v=np.asarray(v,float);m=v.mean();sd=v.std(ddof=1);se=sd/np.sqrt(10);c=stats.t.ppf(.975,9)
    if np.allclose(v,v[0],rtol=0,atol=1e-15):
        return m,m,m,('const',1.0 if abs(m)<1e-15 else 0.0),np.nan
    sp=stats.shapiro(v).pvalue
    if sp>=.05: name,p='t',stats.ttest_1samp(v,0).pvalue
    else: name,p='wilcoxon',stats.wilcoxon(v,zero_method='wilcox',alternative='two-sided',method='auto').pvalue
    return m,m-c*se,m+c*se,(name,p),m/sd
rows=[]
for design,path in RAW.items():
    raw=json.load(open(path))
    for comp,cond in [('deg','C2'),('rec','C3')]:
        for model in ['rf','xgb','mlp']:
            for cls in ['DoS','Bot','Infiltration']:
                c1=np.array([raw[str(s)]['results'][model]['C1']['per_class_f1'][cls] for s in range(10)])
                cx=np.array([raw[str(s)]['results'][model]['per_variant']['hybrid768'][cond]['per_class_f1'][cls] for s in range(10)])
                m,lo,hi,(name,p),dz=test(cx-c1)
                rows.append(dict(design=design,comp=comp,model=model,attack=cls,mean=m,ci95_lo=lo,ci95_hi=hi,test=name,raw_p=p,dz=dz))
df=pd.DataFrame(rows);df['holm_p']=np.nan
for (d,c),g in df.groupby(['design','comp']): df.loc[g.index,'holm_p']=holm(g.raw_p.values)
df.to_csv(OUT/'per_attack_difference_tests_replay.csv',index=False)
pd.set_option('display.width',250)
print(df.to_string(float_format=lambda x:f'{x:.4g}'))
