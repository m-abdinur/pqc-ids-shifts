"""Replay the V6 component-control results (assistant-prepared check)."""
import json,numpy as np,pandas as pd
from pathlib import Path
import zipfile,tempfile
ROOT=Path(__file__).resolve().parents[1]
_TMP=Path(tempfile.mkdtemp(prefix="component_sources_"))
zipfile.ZipFile(ROOT/"reference/component_sources_20260923/component_sources.zip").extractall(_TMP)
OUT=ROOT/"verified_outputs/component_controls"; OUT.mkdir(parents=True,exist_ok=True)

from scipy import stats
W=str(ROOT/'reference')+'/'
MAIN=json.load(open(W+'main_results/revised_pipeline_raw.json'))
S=pd.read_csv(str(_TMP)+'/component_controls_v6/statistics/component_control_split_metrics.csv')
F=pd.read_csv(str(_TMP)+'/component_controls_v6/statistics/component_control_formal_statistics.csv')
M=['rf','xgb','mlp']
print('conditions',S.condition_key.unique().tolist(), 'rows',len(S))
# identity vs locked
def locked(s,m,k):
    r=MAIN[str(s)]['results'][m]
    return {'C1':r['C1']['weighted_f1'],'C2':r['per_variant']['hybrid768']['C2']['weighted_f1'],
            'KS_MI':r['per_variant']['hybrid768']['C3']['weighted_f1'],'KS_only':r['mi_sweep']['0.0']}.get(k)
for k in ['C1','C2','KS_only','KS_MI']:
    sub=S[S.condition_key==k]
    if sub.empty: print('no',k); continue
    diffs=[abs(r.weighted_f1-locked(r.seed,r.model_key,k)) for r in sub.itertuples()]
    print(f'{k}: vs locked max |diff| {max(diffs):.3g} over {len(diffs)}; exact {sum(d==0 for d in diffs)}')
# Table 4.18 means
piv=S.pivot_table(index='model_key',columns='condition_key',values='weighted_f1',aggfunc='mean').loc[M]
print(piv.round(4).to_string())
# contrasts replay
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);m=len(p);a=np.empty(m);run=0
    for r,i in enumerate(o): run=max(run,min(1,(m-r)*p[i]));a[i]=run
    return a
def tost(d,mg):
    md=d.mean();se=d.std(ddof=1)/np.sqrt(len(d));return max(stats.t.sf((md+mg)/se,len(d)-1),stats.t.cdf((md-mg)/se,len(d)-1))
def split_vals(k,m): return S[(S.condition_key==k)&(S.model_key==m)].sort_values('seed').weighted_f1.to_numpy()
contr={}
for _,r in F.iterrows(): contr.setdefault(r.Contrast,[]).append(r)
print('contrasts in file:',list(contr))
key={'C1':'C1','C2':'C2','KS-only':'KS_only','KS+MI':'KS_MI','MI-only threshold':'MI_only_threshold','MI-only top-k':'MI_only_topk','Random stable top-k':'random_stable_topk','Random candidate top-k':'random_candidate_topk'}
print('condition names:',S[['condition_key','Condition']].drop_duplicates().values.tolist())
names=dict(S[['Condition','condition_key']].drop_duplicates().values)
out=[]
for c,rows in contr.items():
    a,b=[x.strip() for x in c.split('−')]
    ka,kb=names.get(a),names.get(b)
    res=[]
    for r in rows:
        d=split_vals(ka,r.model_key)-split_vals(kb,r.model_key)
        sp=stats.shapiro(d).pvalue
        p=stats.ttest_1samp(d,0).pvalue if sp>=.05 else stats.wilcoxon(d).pvalue
        res.append((r,d.mean(),p,tost(d,.01),tost(d,.005)))
    hp=holm([x[2] for x in res]);h1=holm([x[3] for x in res]);h5=holm([x[4] for x in res])
    for (r,md,p,t1,t5),a1,a2,a3 in zip(res,hp,h1,h5):
        out.append(dict(contrast=c,model=r.model_key,mean=md,d_mean=abs(md-r['Mean Difference']),holm=a1,d_holm=abs(a1-r['Difference Holm p']),tost01_holm=a2,d_t01=abs(a2-r['TOST Holm p ±0.010']),tost005_holm=a3,d_t005=abs(a3-r['TOST Holm p ±0.005'])))
o=pd.DataFrame(out);pd.set_option('display.width',250)
print(o.to_string(float_format=lambda x:f'{x:.4g}'))
print('max diffs vs file:',o[['d_mean','d_holm','d_t01','d_t005']].max().to_dict())
o.to_csv(OUT/'component_contrasts_replay.csv',index=False)
