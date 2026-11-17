"""Replay the D6 repeated random-stable control (assistant-prepared check)."""
import json,glob,hashlib,numpy as np,pandas as pd
from pathlib import Path
import zipfile,tempfile
ROOT=Path(__file__).resolve().parents[1]
_TMP=Path(tempfile.mkdtemp(prefix="component_sources_"))
zipfile.ZipFile(ROOT/"reference/component_sources_20260923/component_sources.zip").extractall(_TMP)
OUT=ROOT/"verified_outputs/component_controls"; OUT.mkdir(parents=True,exist_ok=True)

from scipy import stats
W=str(ROOT/'reference')+'/'
MAIN=json.load(open(W+'main_results/revised_pipeline_raw.json'))
D=str(_TMP)+'/random_stable_d6/'
FEATS=json.load(open(W+'seed0_reproduction_20260920/results/seed0_status.json'))['prepared_data']['features']
M={'rf':100,'xgb':100,'mlp':50}
rows=[];issues=[]
ck=sorted(glob.glob(D+'full_10_seed_run/checkpoints/*.json'))
assert len(ck)==30
man={}
for f in glob.glob(D+'subset_manifests/split_*/draw_*.json'):
    j=json.load(open(f)); man[(j['split_seed'],j['draw_id'])]=j
print('manifests',len(man))
sigs=set()
for s in range(10):
    stable=MAIN[str(s)]['stable_prep']; mi=MAIN[str(s)]['mi_prep']
    true_sel=[f for f in stable if mi[f]>=.05]
    draws=[man[(s,d)] for d in range(100)]
    fs=[frozenset(x['features']) for x in draws]
    if len(set(fs))!=100: issues.append(f'split {s}: duplicate subsets')
    for x in draws:
        if len(x['features'])!=len(true_sel): issues.append(f'split {s} draw {x["draw_id"]}: size')
        if not set(x['features'])<=set(stable): issues.append(f'split {s} draw {x["draw_id"]}: outside stable pool')
        if set(x['features'])==set(true_sel): issues.append(f'split {s} draw {x["draw_id"]}: equals true subset')
    for m,n in M.items():
        c=json.load(open(D+f'full_10_seed_run/checkpoints/split_{s:02d}_{m}.json'))
        if c['status']!='complete' or len(c['random_results'])!=n: issues.append(f'{s} {m}: incomplete {len(c["random_results"])}')
        ids=[r['draw_id'] for r in c['random_results']]
        if sorted(ids)!=list(range(n)): issues.append(f'{s} {m}: draw ids {sorted(ids)[:3]}...')
        for r in c['random_results']:
            if r['subset_signature']!=draws[r['draw_id']]['subset_signature']: issues.append(f'{s} {m} draw {r["draw_id"]}: signature')
        t=c['true_result']['weighted_f1']
        locked=MAIN[str(s)]['results'][m]['per_variant']['hybrid768']['C3']['weighted_f1']
        rw=np.array([r['weighted_f1'] for r in c['random_results']])
        rows.append(dict(seed=s,model=m,true=t,locked=locked,d_true=abs(t-locked),rand_mean=rw.mean(),n=len(rw),
                         pct=100*((rw<t).sum()+.5*(rw==t).sum())/len(rw),rmin=rw.min(),rmax=rw.max(),n_above=int((rw>t).sum())))
print('issues:',issues[:10],len(issues))
df=pd.DataFrame(rows)
print('true C3 vs locked C3: max |diff|',df.d_true.max(),'exact',int((df.d_true==0).sum()),'of 30')
print(df[df.d_true>0][['seed','model','true','locked','d_true']].to_string())
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);m=len(p);a=np.empty(m);run=0
    for r,i in enumerate(o): run=max(run,min(1,(m-r)*p[i]));a[i]=run
    return a
res=[]
for m in M:
    g=df[df.model==m].sort_values('seed'); d=(g.true-g.rand_mean).to_numpy()
    se=d.std(ddof=1)/np.sqrt(10);t=stats.t.ppf(.975,9)
    sp=stats.shapiro(d).pvalue
    two=stats.ttest_1samp(d,0).pvalue if sp>=.05 else stats.wilcoxon(d).pvalue
    one=stats.ttest_1samp(d,0,alternative='greater').pvalue if sp>=.05 else stats.wilcoxon(d,alternative='greater').pvalue
    res.append(dict(model=m,true=g.true.mean(),rand=g.rand_mean.mean(),diff=d.mean(),lo=d.mean()-t*se,hi=d.mean()+t*se,pos=int((d>0).sum()),shapiro=sp,p_two=two,p_one=one,
                    med_pct=g.pct.median(),min_pct=g.pct.min(),splits_with_random_above=int((g.n_above>0).sum()),total_random_above=int(g.n_above.sum()),total_draws=int(g.n.sum())))
r=pd.DataFrame(res);r['holm_two']=holm(r.p_two);r['holm_one']=holm(r.p_one)
pd.set_option('display.width',250)
print(r.to_string(float_format=lambda x:f'{x:.6g}'))
F=pd.read_csv(D+'full_10_seed_run/statistical_analysis/d6_paired_difference_statistics.csv')
print(F[['model_key','mean_true_c3','mean_random_stable','mean_difference','ci_95_lower','ci_95_upper','selected_test','alternative','raw_p_value','holm_adjusted_p','mean_true_percentile','median_true_percentile','minimum_true_percentile']].to_string())
df.to_csv(OUT/'d6_split_replay.csv',index=False); r.to_csv(OUT/'d6_statistics_replay.csv',index=False)
