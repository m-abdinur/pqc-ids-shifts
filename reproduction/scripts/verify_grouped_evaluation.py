"""Independent verification of a returned grouped-evaluation run (assistant-prepared).
Usage: python verify_grouped_evaluation.py <extracted results folder> <canonical_provenance_tagged_matrix.pkl.gz>
"""
import json,hashlib,numpy as np,pandas as pd,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"verified_outputs/grouped_evaluation"; OUT.mkdir(parents=True,exist_ok=True)
from sklearn.metrics import f1_score,roc_auc_score
from scipy import stats
R=str(Path(sys.argv[1])/'results')+'/'
df=pd.read_pickle(sys.argv[2])
F=json.load(open(R+'prepared_data.json'))['features']
y=df.label_id.to_numpy()
ah=lambda v: hashlib.sha256(np.ascontiguousarray(np.asarray(v)).tobytes()).hexdigest()
assert ah(df[F].values)=='b852fe0225df0c7b3a3407c3d2c98bccfa2a6252b144f3e2e6d6ba5549df5e6e'
# independent grouping (dict of bytes -> id, no pandas factorize)
ids={}; g=np.empty(len(df),int)
for i,r in enumerate(np.ascontiguousarray(df[F].to_numpy())):
    g[i]=ids.setdefault(r.tobytes(),len(ids))
sc=json.load(open(R+'split_checks.json'))
M=pd.read_csv(R+'metrics.csv')
issues=[]; split_info={}
for rec in sc:
    s=rec['seed']
    a=np.load(R+f'predictions/seed{s}_rf_C1.npz'); te=a['test_row_positions']
    tr=np.setdiff1d(np.arange(len(df)),te)
    if ah(te)!=rec['test_positions_sha256']: issues.append(f'{s} test hash')
    cross=len(set(g[tr])&set(g[te]))
    shares=[(y[te]==c).sum()/(y==c).sum() for c in range(4)]
    split_info[s]=dict(cross=cross,shares=np.round(shares,4).tolist(),n_te=len(te))
    if cross: issues.append(f'{s} crossing {cross}')
    if not all(.29<=x<=.31 for x in shares): issues.append(f'{s} shares')
print('split issues',issues); print(split_info[0], 'crossing all zero:',all(v['cross']==0 for v in split_info.values()))
# metrics replay
d={'wf1':0,'auc':0,'pc':0}
for _,r in M.iterrows():
    a=np.load(R+r.prediction_file,allow_pickle=False)
    yt,yp,P,c,pos=a['y_true'],a['y_pred'],a['probability'],a['classes'],a['test_row_positions']
    assert list(c)==[0,1,2,3] and np.array_equal(yt,y[pos]) and np.array_equal(yp,c[P.argmax(1)]) and np.isfinite(P).all()
    assert ah(yp)==r.prediction_sha256 and ah(P)==r.probability_sha256
    ref=np.load(R+f'predictions/seed{r.seed}_rf_C1.npz')['test_row_positions']; assert np.array_equal(pos,ref)
    d['wf1']=max(d['wf1'],abs(f1_score(yt,yp,average='weighted')-r.weighted_f1))
    d['auc']=max(d['auc'],abs(roc_auc_score(yt,P,labels=c,multi_class='ovr',average='weighted')-r.roc_auc_ovr_weighted))
    pc=f1_score(yt,yp,labels=c,average=None)
    d['pc']=max(d['pc'],max(abs(pc[i]-r[n+'_f1']) for i,n in enumerate(['Benign','DoS','Bot','Infiltration'])))
print('metric replay max diffs',d,'rows',len(M))
fr=pd.DataFrame(json.load(open(R+'fit_records.json')))
print('fits',len(fr),'C1/C2 share fit:',all(M[(M.seed==s)&(M.model==m)&M.condition.isin(['C1','C2'])].fit_id.nunique()==1 for s in range(10) for m in ['rf','xgb','mlp']))
# selection vs plumbing dry run (model-independent)
a=json.load(open(R+'feature_selection.json')); b=json.load(open(ROOT/'verified_outputs/grouped_evaluation/pre_run_dry_run_feature_selection.json'))
same=all(x['stable']==z['stable'] and x['shifted']==z['shifted'] and x['selected']==z['selected'] and max(abs(x['KS_D'][k]-z['KS_D'][k]) for k in x['KS_D'])<=1e-12 and max(abs(x['MI'][k]-z['MI'][k]) for k in x['MI'])<=1e-8 for x,z in zip(a,b))
print('selection identical to local dry run:',same,[(len(x['stable']),len(x['shifted']),len(x['selected'])) for x in a])
# independent stats
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);m=len(p);out=np.empty(m);run=0
    for k,i in enumerate(o): run=max(run,min(1,(m-k)*p[i]));out[i]=run
    return out
rows=[]
for m in ['rf','xgb','mlp']:
    w=M[M.model==m].pivot(index='seed',columns='condition',values='weighted_f1')
    for name,dd in [('C2-C1',w.C2-w.C1),('C3-C1',w.C3-w.C1)]:
        dd=dd.to_numpy(); se=dd.std(ddof=1)/np.sqrt(10)
        sp=stats.shapiro(dd).pvalue; p=stats.ttest_1samp(dd,0).pvalue if sp>=.05 else stats.wilcoxon(dd).pvalue
        t=stats.t.ppf(.95,9)
        row=dict(model=m,contrast=name,C1=w.C1.mean(),C2=w.C2.mean(),C3=w.C3.mean(),mean=dd.mean(),lo90=dd.mean()-t*se,hi90=dd.mean()+t*se,mn=dd.min(),mx=dd.max(),test='t' if sp>=.05 else 'W',p=p)
        for mg in (.01,.005): row[f'tost{mg}']=max(stats.t.sf((dd.mean()+mg)/se,9),stats.t.cdf((dd.mean()-mg)/se,9))
        rows.append(row)
S=pd.DataFrame(rows)
for c in ['C2-C1','C3-C1']:
    k=S.contrast==c; S.loc[k,'holm']=holm(S.loc[k,'p'])
k=S.contrast=='C3-C1'
for mg in (.01,.005): S.loc[k,f'holm_tost{mg}']=holm(S.loc[k,f'tost{mg}'])
pd.set_option('display.width',250); print(S.to_string(float_format=lambda x:f'{x:.4g}'))
T=pd.read_csv(R+'registered_procedure_tests.csv')
print(T[['model','contrast','mean_difference','selected_test','difference_holm_p','tost_holm_p_0.01','tost_holm_p_0.005']].to_string())
print(pd.read_csv(R+'decision_rule.csv').to_string())
S.to_csv(OUT/'grouped_stats_replay.csv',index=False)
