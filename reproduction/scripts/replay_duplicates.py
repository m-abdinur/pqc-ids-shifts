"""Duplicate / source-day replay (assistant-prepared check). Usage: python <script> <extracted bundle folder>"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=Path(sys.argv[1]) if len(sys.argv)>1 else Path("dup_sourceday_sources")  # extracted bundle folder
OUT=ROOT/"verified_outputs/duplicates_sourceday"; OUT.mkdir(parents=True,exist_ok=True)
import pandas as pd,numpy as np,hashlib,json
from sklearn.model_selection import train_test_split
B=str(SRC)+'/'
df=pd.read_pickle(B+'source_day_robustness/d5_0_provenance_reconstruction/canonical_provenance_tagged_matrix.pkl.gz')
FEATS=json.load(open(str(ROOT/'reference')+'/seed0_reproduction_20260920/results/seed0_status.json'))['prepared_data']['features']
ah=lambda v: hashlib.sha256(np.ascontiguousarray(np.asarray(v)).tobytes()).hexdigest()
X=df[FEATS]; y=df['label_id'].to_numpy()
print('X hash ok', ah(X.values)=='b852fe0225df0c7b3a3407c3d2c98bccfa2a6252b144f3e2e6d6ba5549df5e6e', 'y hash ok', ah(y)=='e1f328443429dcdfa8c521b1296c72c0c14ad3c17d17543065b4f81c6ab9d15a', y.dtype)
print('matrix_row_index is 0..n-1:', (df.matrix_row_index.to_numpy()==np.arange(len(df))).all())
# exact-vector keys (bytes of float32 row)
arr=np.ascontiguousarray(X.to_numpy())
keys=np.array([r.tobytes() for r in arr],dtype=object)
k=pd.Series(keys)
gid=pd.factorize(k)[0]
df['gid']=gid
g=df.groupby('gid')
size=g.size(); nlab=g['label_id'].nunique(); nday=g['source_day'].nunique()
dupg=size[size>1]
res={'rows':len(df),'unique_vectors':int(size.shape[0]),'duplicate_groups':int(len(dupg)),'rows_in_duplicate_groups':int(dupg.sum()),
     'excess_duplicate_rows':int((dupg-1).sum()),'max_group':int(size.max()),'conflicting_label_groups':int((nlab>1).sum()),
     'rows_in_conflicting_label_groups':int(size[nlab>1].sum()),'cross_day_dup_groups':int(((nday>1)&(size>1)).sum())}
print(res)
cls={0:'Benign',1:'DoS',2:'Bot',3:'Infiltration'}
print('check class codes:',df.groupby('label_id')['label'].first().to_dict())
for c in range(4):
    sub=df[df.label_id==c]; s2=sub.groupby('gid').size()
    ingroup=int(size.loc[sub.gid].gt(1).sum())
    print(cls[c],'rows',len(sub),'unique',sub.gid.nunique(),'rows in any dup group',ingroup, round(100*ingroup/len(sub),2),'within-class excess',int((s2-1).sum()))
# train-test matches per seed and class
rows=[]
for seed in range(10):
    tr,te=train_test_split(np.arange(len(df)),test_size=.3,random_state=seed,stratify=y)
    trg=pd.DataFrame({'gid':gid[tr],'lab':y[tr],'day':df.source_day.to_numpy()[tr]})
    labs=trg.groupby('gid')['lab'].agg(set); days=trg.groupby('gid')['day'].agg(set)
    for i in te:
        pass
    teg=gid[te]; tel=y[te]; ted=df.source_day.to_numpy()[te]
    inT=pd.Series(teg).isin(labs.index).to_numpy()
    same=np.array([ (g_ in labs.index) and (l in labs[g_]) for g_,l in zip(teg,tel)])
    diff=np.array([ (g_ in labs.index) and bool(labs[g_]-{l}) for g_,l in zip(teg,tel)])
    sameday=np.array([ (g_ in days.index) and (d in days[g_]) for g_,d in zip(teg,ted)])
    for c in range(4):
        m=tel==c
        rows.append(dict(seed=seed,cls=cls[c],n=int(m.sum()),any=inT[m].mean(),same=same[m].mean(),diff=diff[m].mean(),sameday=sameday[m].mean()))
    rows.append(dict(seed=seed,cls='All',n=len(te),any=inT.mean(),same=same.mean(),diff=diff.mean(),sameday=sameday.mean()))
r=pd.DataFrame(rows)
# pooled rates (same n each seed so mean == pooled for fixed n)
pooled=r.groupby('cls').apply(lambda x: pd.Series({'n':x.n.mean(),'any':np.average(x['any'],weights=x.n),'same':np.average(x.same,weights=x.n),'diff':np.average(x['diff'],weights=x.n),'sameday':np.average(x.sameday,weights=x.n)}))
print(pooled.round(4).to_string())
print(r[r.cls=='All'][['seed','any']].assign(any=lambda d:(100*d['any']).round(2)).T.to_string())
r.to_csv(OUT/'train_test_match_replay.csv',index=False); json.dump(res,open(OUT/'dup_global_replay.json','w'),indent=2)
