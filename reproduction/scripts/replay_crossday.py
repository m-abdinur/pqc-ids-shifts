"""Duplicate / source-day replay (assistant-prepared check). Usage: python <script> <extracted bundle folder>"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=Path(sys.argv[1]) if len(sys.argv)>1 else Path("dup_sourceday_sources")  # extracted bundle folder
OUT=ROOT/"verified_outputs/duplicates_sourceday"; OUT.mkdir(parents=True,exist_ok=True)
import pandas as pd,numpy as np,json,glob
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy import stats
B=str(SRC)+'/source_day_robustness/'
df=pd.read_pickle(B+'d5_0_provenance_reconstruction/canonical_provenance_tagged_matrix.pkl.gz')
y=df.label_id.to_numpy(); day=df.source_day.to_numpy()
DIR={'feb28_to_mar01':('2018-02-28','2018-03-01'),'mar01_to_feb28':('2018-03-01','2018-02-28')}
rows=[];checks=[]
for d,(trd,ted) in DIR.items():
    for s in range(10):
        otr,ote=train_test_split(np.arange(len(df)),test_size=.3,random_state=s,stratify=y)
        for m in ['rf','xgb','mlp']:
            a=np.load(B+f'd5_b_cross_day_generalisation/full_10_seed_run/predictions/{d}/seed_{s:02d}/{m}_predictions.npz',allow_pickle=False)
            tr,te,yt=a['train_index'],a['test_index'],a['y_true']
            ok=dict(dir=d,seed=s,model=m,
              no_overlap=len(np.intersect1d(tr,te))==0,
              ytrue_ok=np.array_equal(yt,y[te]),
              train_inf_day=set(day[tr][y[tr]==3])=={trd}, test_inf_day=set(day[te][y[te]==3])=={ted},
              other_train_same=np.array_equal(np.sort(tr[y[tr]!=3]),np.sort(otr[y[otr]!=3])),
              other_test_same=np.array_equal(np.sort(te[y[te]!=3]),np.sort(ote[y[ote]!=3])),
              n_inf_train=int((y[tr]==3).sum()), n_inf_test=int((y[te]==3).sum()))
            checks.append(ok)
            for c in ['c1','c2','c3']:
                p=a[f'{c}_pred']; pr=a[f'{c}_prob']
                assert np.array_equal(p,pr.argmax(1))
                rows.append(dict(dir=d,seed=s,model=m,cond=c.upper(),wf1=f1_score(yt,p,average='weighted'),inf_f1=f1_score(yt,p,labels=[3],average=None)[0]))
ck=pd.DataFrame(checks); print(ck.drop(columns=['dir','seed','model']).agg(lambda c: c.all() if c.dtype==bool else sorted(set(c))).to_string())
R=pd.DataFrame(rows)
M=pd.read_csv(B+'d5_b_cross_day_generalisation/full_10_seed_run/d5b_metrics_by_direction_seed_model_condition.csv')
print(M.columns.tolist()[:20])
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);m=len(p);a=np.empty(m);run=0
    for r,i in enumerate(o): run=max(run,min(1,(m-r)*p[i]));a[i]=run
    return a
def tost(dd,mg=.01):
    md=dd.mean();se=dd.std(ddof=1)/np.sqrt(len(dd));return max(stats.t.sf((md+mg)/se,len(dd)-1),stats.t.cdf((md-mg)/se,len(dd)-1))
out=[]
for metric in ['wf1','inf_f1']:
    for d in DIR:
        for m in ['rf','xgb','mlp']:
            g=R[(R.dir==d)&(R.model==m)].pivot(index='seed',columns='cond',values=metric)
            d31=(g.C3-g.C1).to_numpy(); se=d31.std(ddof=1)/np.sqrt(10); t=stats.t.ppf(.95,9)
            out.append(dict(metric=metric,dir=d,model=m,C1=g.C1.mean(),C2=g.C2.mean(),C3=g.C3.mean(),c3_c1=d31.mean(),ci90=(round(d31.mean()-t*se,4),round(d31.mean()+t*se,4)),tost=tost(d31),c3_above_c1_splits=int((d31>0).sum())))
O=pd.DataFrame(out)
for mt in ['wf1','inf_f1']:
    k=O.metric==mt; O.loc[k,'tost_holm']=holm(O.loc[k,'tost'])
pd.set_option('display.width',250); print(O.round(4).to_string())
O.to_csv(OUT/'crossday_replay.csv',index=False); R.to_csv(OUT/'crossday_split_metrics.csv',index=False)
