"""Exploratory, descriptive: do the main C2 and C3 contrasts hold on test rows
that have no exact feature-vector copy in that split's training data?
Definition fixed before scoring: 'novel' = test row whose 70-feature vector
does not occur anywhere in the split's training rows (any label).
Uses the operator-sensitivity historical-arm predictions, which reproduce all
90 locked main weighted-F1 values exactly. No models are refit. No tests."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=Path(sys.argv[1]) if len(sys.argv)>1 else Path("dup_sourceday_sources")  # extracted bundle folder
OUT=ROOT/"verified_outputs/duplicates_sourceday"; OUT.mkdir(parents=True,exist_ok=True)

import pandas as pd,numpy as np,json
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
df=pd.read_pickle(str(SRC)+'/source_day_robustness/d5_0_provenance_reconstruction/canonical_provenance_tagged_matrix.pkl.gz')
FEATS=json.load(open(str(ROOT/'reference')+'/seed0_reproduction_20260920/results/seed0_status.json'))['prepared_data']['features']
y=df.label_id.to_numpy()
keys=pd.factorize(pd.Series([r.tobytes() for r in np.ascontiguousarray(df[FEATS].to_numpy())]))[0]
P=str(SRC.parent/'operator_sensitivity_results/results/predictions')+'/'  # extracted 20260922T180508Z_1f6bbb0b results ZIP
rows=[]
for s in range(10):
    tr,te=train_test_split(np.arange(len(df)),test_size=.3,random_state=s,stratify=y)
    novel=~np.isin(keys[te],keys[tr])
    for m in ['rf','xgb','mlp']:
        out={}
        for cond,arm in [('C1','original'),('C2','historical'),('C3','historical')]:
            a=np.load(P+f'seed{s}_{m}_{arm}_{cond}.npz',allow_pickle=False)
            assert np.array_equal(a['test_row_positions'],te) and np.array_equal(a['y_true'],y[te])
            p=a['y_pred']; yt=a['y_true']
            out[cond]=(f1_score(yt,p,average='weighted'),f1_score(yt[novel],p[novel],average='weighted'),
                       f1_score(yt,p,labels=[3],average=None)[0],f1_score(yt[novel],p[novel],labels=[3],average=None)[0])
        rows.append(dict(seed=s,model=m,novel_share=novel.mean(),
            all_C2_C1=out['C2'][0]-out['C1'][0],novel_C2_C1=out['C2'][1]-out['C1'][1],
            all_C3_C1=out['C3'][0]-out['C1'][0],novel_C3_C1=out['C3'][1]-out['C1'][1],
            novel_C1=out['C1'][1],all_C1=out['C1'][0],
            inf_all_C3_C1=out['C3'][2]-out['C1'][2],inf_novel_C3_C1=out['C3'][3]-out['C1'][3]))
R=pd.DataFrame(rows)
S=R.groupby('model').agg(novel_share=('novel_share','mean'),all_C1=('all_C1','mean'),novel_C1=('novel_C1','mean'),
    all_C2_C1=('all_C2_C1','mean'),novel_C2_C1=('novel_C2_C1','mean'),all_C3_C1=('all_C3_C1','mean'),novel_C3_C1=('novel_C3_C1','mean'),
    novel_C3_C1_min=('novel_C3_C1','min'),novel_C3_C1_max=('novel_C3_C1','max'),inf_novel_C3_C1=('inf_novel_C3_C1','mean')).loc[['rf','xgb','mlp']]
pd.set_option('display.width',250);print(S.round(4).to_string())
novel_class=[]
for s in range(10):
    tr,te=train_test_split(np.arange(len(df)),test_size=.3,random_state=s,stratify=y)
    nv=~np.isin(keys[te],keys[tr]); novel_class.append([ (nv&(y[te]==c)).sum() for c in range(4)])
print('mean novel test rows per class (Benign,DoS,Bot,Inf):',np.mean(novel_class,axis=0))
R.to_csv(OUT/'novel_rows_main_by_split.csv',index=False); S.to_csv(OUT/'novel_rows_main_summary.csv')
