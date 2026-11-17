"""Replay Tables 4.10-4.12 from locked results and the collected Drive sources (assistant-prepared check)."""
import json,zipfile,io,numpy as np,pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
Z=zipfile.ZipFile(ROOT/'reference/restrictions_sources_20260923/table410_sources.zip')
zload=lambda p: json.loads(Z.read(p))
OUT=ROOT/'verified_outputs/restrictions'; OUT.mkdir(parents=True,exist_ok=True)
from scipy import stats
W=str(ROOT/'reference')+'/'
MAIN=json.load(open(W+'main_results/revised_pipeline_raw.json'))
B3=json.load(open(W+'b3_results/revised_pipeline_b3_raw.json'))
COMB=zload('combined_timing_identifier/full_10_splits/combined_raw_10_splits.json')['splits']
TF=zload('timing_free_v6/outputs/final_10_splits_xgboost_3_2_0/timing_free_raw_10_splits.json')
SW=zload('v5_members/revised_pipeline_b3_mlp_sweep_raw.json')
M=['rf','xgb','mlp']
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);m=len(p);a=np.empty(m);run=0
    for r,i in enumerate(o): run=max(run,min(1,(m-r)*p[i]));a[i]=run
    return a
def ci(d,conf):
    d=np.asarray(d);se=d.std(ddof=1)/np.sqrt(len(d));t=stats.t.ppf(1-(1-conf)/2,len(d)-1);return d.mean()-t*se,d.mean()+t*se
def tost(d,mg):
    d=np.asarray(d);md=d.mean();se=d.std(ddof=1)/np.sqrt(len(d))
    return max(stats.t.sf((md+mg)/se,len(d)-1),stats.t.cdf((md-mg)/se,len(d)-1))
def g(raw,s,m,cond,key='weighted_f1'):
    r=raw[str(s)]['results'][m]
    return r['C1'][key] if cond=='C1' else r['per_variant']['hybrid768'][cond][key]
out={}
# identity checks: combined C1/C2 vs locked B3; timing-free C1/C2 vs locked main
idc=[];idt=[]
for s in range(10):
    for m in M:
        for cond in ['C1','C2']:
            idc.append(abs(g(COMB,s,m,cond)-g(B3,s,m,cond)))
            tfv=TF['splits'][str(s)]['classifiers'][m][cond]['weighted_f1']
            idt.append(abs(tfv-g(MAIN,s,m,cond)))
            for c in ['Benign','DoS','Bot','Infiltration']:
                idc.append(abs(g(COMB,s,m,cond,'per_class_f1')[c]-g(B3,s,m,cond,'per_class_f1')[c]))
                idt.append(abs(TF['splits'][str(s)]['classifiers'][m][cond]['per_class_f1'][c]-g(MAIN,s,m,cond,'per_class_f1')[c]))
print('combined C1/C2 vs locked B3: max F1 diff',max(idc),'n',len(idc))
print('timing-free C1/C2 vs locked main: max F1 diff',max(idt),'n',len(idt))
rows=[]
for name,raw,get3 in [('B3',B3,lambda s,m:g(B3,s,m,'C3')),
                      ('Combined',COMB,lambda s,m:g(COMB,s,m,'C3')),
                      ('Timing-free',None,lambda s,m:TF['splits'][str(s)]['classifiers'][m]['C3']['weighted_f1'])]:
    base=B3 if name!='Timing-free' else MAIN
    for m in M:
        c1=np.array([g(base,s,m,'C1') for s in range(10)]); c2=np.array([g(base,s,m,'C2') for s in range(10)])
        c3=np.array([get3(s,m) for s in range(10)]); d=c3-c1
        lo90,hi90=ci(d,.9); lo95,hi95=ci(d,.95)
        rows.append(dict(config=name,model=m,C1=c1.mean(),C2=c2.mean(),C3=c3.mean(),d=d.mean(),ci90=(lo90,hi90),ci95=(lo95,hi95),
                         tost001=tost(d,.01),tost0005=tost(d,.005),c3_gt_c2=int((c3>c2).sum())))
df=pd.DataFrame(rows)
for (cfg),gp in df.groupby('config'):
    df.loc[gp.index,'holm001']=holm(gp.tost001); df.loc[gp.index,'holm0005']=holm(gp.tost0005)
pd.set_option('display.width',250)
print(df.to_string(float_format=lambda x:f'{x:.6g}'))
# sweep
c1=np.array([SW[str(s)]['results']['mlp']['C1']['weighted_f1'] for s in range(10)])
print('sweep C1 %.6f'%c1.mean(),' vs locked B3 C1 %.6f'%np.mean([g(B3,s,'mlp','C1') for s in range(10)]))
for thr in ['0.0','0.01','0.05','0.1','0.2']:
    v=np.array([SW[str(s)]['results']['mlp']['mi_sweep'][thr] for s in range(10)]);d=v-c1
    cnt=np.mean([SW[str(s)]['results']['mlp']['selected_counts'][thr] for s in range(10)])
    lo,hi=ci(d,.9)
    print(f'  thr {thr}: C3 {v.mean():.4f} gap {d.mean():+.4f} 90%CI [{lo:+.4f},{hi:+.4f}] n={cnt:.1f} tost01 {tost(d,.01):.3g}')
sc3=np.array([SW[str(s)]['results']['mlp']['per_variant']['hybrid768']['C3']['weighted_f1'] for s in range(10)])
b3c3=np.array([g(B3,s,'mlp','C3') for s in range(10)])
print('sweep headline C3 == locked B3 MLP C3:',np.array_equal(sc3,b3c3), 'sweep mi0.05 == headline C3:', np.array_equal(np.array([SW[str(s)]['results']['mlp']['mi_sweep']['0.05'] for s in range(10)]),sc3))
df.to_csv(OUT/'restrictions_replay.csv',index=False)
