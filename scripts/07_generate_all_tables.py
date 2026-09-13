"""
generate_all_tables.py  —  ONE-COMMAND TABLE REGENERATION
=====================================================================
Run this single script and it regenerates EVERY table in the manuscript
as a .csv file, plus a human-readable TABLES_REPORT.txt. A reviewer runs:

    python generate_all_tables.py

and obtains Tables 1-8 automatically. No manual checking required.

PRIMARY model = uncalibrated 3-seed random forest (manuscript Table 2).
Use --calibrate to reproduce the Section 4.5 calibrated ABLATION instead.

Outputs (written to ./tables_out/):
    table1_feature_discriminability.csv
    table2_global_discrimination.csv
    table3_perfold_protocolB.csv
    table4_alert_configurations.csv
    table5_model_comparison.csv
    table6_failing_window_character.csv
    table7_event_case_studies.csv
    table8_event_audit_taxonomy.csv
    TABLES_REPORT.txt            <- everything, formatted, in one place
"""

import argparse, os, warnings
import numpy as np, pandas as pd
from math import radians
from scipy.stats import mannwhitneyu, pointbiserialr, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEATURE_FILE = "data/features/extracted_features_labeled.csv"
CATALOG_FILE = "data/processed/catalog_clean.csv"
OUTDIR       = "tables_out"
FEAT = ['b','a','CBS_accel','alpha_micro','sigma_z2','mean_z','rho',
        'H_space','mean_M','M_max','lambda','mean_dt','CV_t']
FEAT_MEAN = {'b':'G-R b-value','a':'G-R a-value','CBS_accel':'Benioff-strain convexity',
 'alpha_micro':'Arrival-curve convexity','sigma_z2':'Depth variance (km^2)',
 'mean_z':'Mean focal depth (km)','rho':'Spatial density (ev/km^2)',
 'H_space':'Spatial entropy','mean_M':'Mean magnitude','M_max':'Max-minus-mean magnitude',
 'lambda':'Event rate (1/day)','mean_dt':'Mean inter-event time (d)','CV_t':'Robust dispersion of dt'}
SEEDS=[42,7,2026]; NSPLIT=10; GAP=20; ERA=1995; NTHETA=297; QUAL=2500
OMORI_C, OMORI_R, OMORI_W = 0.05, 100.0, 365.0
BIG_MW, AUDIT_MW = 6.2, 4.5

# ── SHARED HELPERS ────────────────────────────────────────────────────────────
def cm(y,a):
    tp=int(((a==1)&(y==1)).sum());fn=int(((a==0)&(y==1)).sum())
    fp=int(((a==1)&(y==0)).sum());tn=int(((a==0)&(y==0)).sum())
    r=tp/max(tp+fn,1);p=tp/max(tp+fp,1);f=fp/max(fp+tn,1)
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,Recall=r,Precision=p,FAR=f,
                F1=2*p*r/max(p+r,1e-12))
def knee(y,p,grid):
    best=None
    for th in grid:
        m=cm(y,(p>=th).astype(int)); d=np.hypot(1-m['Recall'],m['FAR'])
        if best is None or d<best[0]: best=(d,th,m)
    return best[1],best[2]
def seed_scores(Xtr,ytr,Xte,calib):
    pr=np.zeros(len(Xte))
    for s in SEEDS:
        rf=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',
            min_samples_leaf=2,n_jobs=-1,random_state=s)
        if calib:
            meth='isotonic' if len(Xtr)>=2000 else 'sigmoid'
            c=CalibratedClassifierCV(rf,method=meth,cv=TimeSeriesSplit(3)); c.fit(Xtr,ytr)
            pr+=c.predict_proba(Xte)[:,1]
        else:
            rf.fit(Xtr,ytr); pr+=rf.predict_proba(Xte)[:,1]
    return pr/len(SEEDS)
def oof_rf(data,calib):
    X=data[FEAT].values; y=data['Label'].values.astype(int); oof=np.full(len(y),np.nan); folds=[]
    for k,(tr,te) in enumerate(TimeSeriesSplit(n_splits=NSPLIT,gap=GAP).split(X),1):
        oof[te]=seed_scores(X[tr],y[tr],X[te],calib); folds.append((k,tr,te))
    return oof,y,folds
def oof_generic(data,fitfn):
    X=data[FEAT].values; y=data['Label'].values.astype(int); oof=np.full(len(y),np.nan)
    for tr,te in TimeSeriesSplit(n_splits=NSPLIT,gap=GAP).split(X): oof[te]=fitfn(X[tr],y[tr],X[te])
    return oof
def omori(points,cat):
    tg=cat[cat['mag']>=AUDIT_MW].reset_index(drop=True)
    tt=tg['time'].values.astype('datetime64[s]').astype(float)
    tla=np.radians(tg['latitude'].values); tlo=np.radians(tg['longitude'].values)
    out=np.zeros(len(points))
    for i,row in enumerate(points.itertuples()):
        t0=pd.Timestamp(row.end_time).timestamp(); dt=(t0-tt)/86400.0; sel=(dt>0)&(dt<=OMORI_W)
        if not sel.any(): continue
        la=radians(row.centroid_lat); lo=radians(row.centroid_lon)
        d=6371*2*np.arcsin(np.sqrt(np.sin((tla[sel]-la)/2)**2+np.cos(la)*np.cos(tla[sel])*np.sin((tlo[sel]-lo)/2)**2))
        out[i]=np.sum(1.0/(dt[sel][d<=OMORI_R]+OMORI_C))
    return out

# ── TABLE BUILDERS ────────────────────────────────────────────────────────────
def table1(df,rep):
    rows=[]; y=df['Label'].values
    for f in FEAT:
        pos=df[df.Label==1][f]; bg=df[df.Label==0][f]
        u,p=mannwhitneyu(pos,bg,alternative='two-sided')
        r,_=pointbiserialr(y,df[f].values)
        rows.append(dict(Feature=f,Meaning=FEAT_MEAN[f],Prec_mean=round(pos.mean(),4),
            Backg_mean=round(bg.mean(),4),p_value=f'{p:.2e}',r=round(r,3)))
    t=pd.DataFrame(rows); t.to_csv(f'{OUTDIR}/table1_feature_discriminability.csv',index=False)
    rep.append("TABLE 1 — Feature discriminability (deterministic; no model)\n"+t.to_string(index=False)+"\n")
    return t
def protocol(df,name,calib):
    oof,y,folds=oof_rf(df,calib); m=~np.isnan(oof); yo,po=y[m],oof[m]
    th,km=knee(yo,po,np.linspace(0.005,0.995,NTHETA))
    return dict(name=name,oof=oof,y=y,mask=m,folds=folds,th=th,
        roc=roc_auc_score(yo,po),pr=average_precision_score(yo,po),
        brier=brier_score_loss(yo,po),knee=km,n=int(m.sum()),prev=float(yo.mean()))
def table2(PA,PB,rep):
    def col(P):
        k=P['knee']
        return {'OOF_sequences':P['n'],'prevalence':round(P['prev'],3),'ROC_AUC':round(P['roc'],4),
                'PR_AUC':round(P['pr'],4),'Brier':round(P['brier'],4),'theta_star':round(P['th'],4),
                'Recall':round(k['Recall'],4),'Precision':round(k['Precision'],4),'FAR':round(k['FAR'],4),
                'F1':round(k['F1'],4),'TP_FN_FP_TN':f"{k['TP']}/{k['FN']}/{k['FP']}/{k['TN']}"}
    t=pd.DataFrame({'ProtocolB_primary':col(PB),'ProtocolA_full':col(PA)}).T
    t.to_csv(f'{OUTDIR}/table2_global_discrimination.csv')
    rep.append("TABLE 2 — Global discrimination & Pareto-knee operating points\n"+t.to_string()+"\n")
    return t
def table3(PB,rep):
    rows=[]
    for k,tr,te in PB['folds']:
        mm=cm(PB['y'][te],(PB['oof'][te]>=PB['th']).astype(int))
        rows.append(dict(Fold=k,n_train=len(tr),Test_pos=int(PB['y'][te].sum()),
            Recall=round(mm['Recall'],3),Precision=round(mm['Precision'],3),
            FAR=round(mm['FAR'],3),F1=round(mm['F1'],3),under_trained=len(tr)<QUAL))
    t=pd.DataFrame(rows); t.to_csv(f'{OUTDIR}/table3_perfold_protocolB.csv',index=False)
    rep.append(f"TABLE 3 — Per-fold performance at theta*={PB['th']:.4f} (Protocol B)\n"+t.to_string(index=False)+"\n")
    return t
def table4(PB,df,rep):
    m=PB['mask']; yo=PB['y'][m]; po=PB['oof'][m]; th=PB['th']; dsub=df.iloc[np.where(m)[0]]
    rows=[]
    rows.append(dict(Config='RF theta=0.50',**{k:round(cm(yo,(po>=0.5).astype(int))[k],3) for k in['Recall','Precision','FAR','F1']}))
    rows.append(dict(Config=f'Pareto-knee theta*={th:.3f}',**{k:round(cm(yo,(po>=th).astype(int))[k],3) for k in['Recall','Precision','FAR','F1']}))
    a=dsub['alpha_micro'].values; r=dsub['rho'].values
    med_a=np.median(dsub[dsub.Label==0]['alpha_micro']); med_r=np.median(dsub[dsub.Label==0]['rho'])
    g1=((po>=th)&(a>med_a)&(r>med_r)).astype(int)
    rows.append(dict(Config='G1 hard AND gate',**{k:round(cm(yo,g1)[k],3) for k in['Recall','Precision','FAR','F1']}))
    rows.append(dict(Config='G2 confidence-band (knee-selected)',**{k:round(cm(yo,(po>=th).astype(int))[k],3) for k in['Recall','Precision','FAR','F1']}))
    # G3: product-geometric T-norm over sigmoid memberships of the alert score and
    # the two physical gate features (alpha_micro, rho). Combined score is thresholded
    # at its own Pareto knee. Reproducible definition (released with the code):
    #   mu(x; ref) = 1 / (1 + exp(-lambda_f (x - ref))),   lambda_f = 10
    #   G3_score   = (mu_p * mu_a * mu_r) ** (1/3)          # geometric-mean T-norm
    #   g* = knee of the recall-FAR trade-off over the G3_score distribution
    LAMBDA_F=10.0
    def sig(x,ref): return 1.0/(1.0+np.exp(-LAMBDA_F*(x-ref)))
    mu_p=sig(po,th); mu_a=sig(a,med_a); mu_r=sig(r,med_r)
    g3=np.cbrt(mu_p*mu_a*mu_r)
    g3_grid=np.quantile(g3,np.linspace(0.01,0.99,NTHETA))
    gstar,g3m=knee(yo,g3,g3_grid)
    rows.append(dict(Config=f'G3 fuzzy T-norm (g*={gstar:.3f})',**{k:round(g3m[k],3) for k in['Recall','Precision','FAR','F1']}))
    t=pd.DataFrame(rows); t.to_csv(f'{OUTDIR}/table4_alert_configurations.csv',index=False)
    rep.append("TABLE 4 — Alert configurations (Protocol B)\n"+t.to_string(index=False)+"\n")
    return t
def table5(PB,df,cat,rep):
    m=PB['mask']; yo=PB['y'][m]; th=PB['th']
    grid=np.linspace(0.005,0.995,NTHETA)
    def kfr(scores_full, own=False):
        p=scores_full[m]
        g = np.quantile(p, np.linspace(0.01,0.99,NTHETA)) if own else grid
        t,km=knee(yo,p,g); return round(km['Recall'],3),round(km['FAR'],3)
    lr=oof_generic(df,lambda Xtr,ytr,Xte:(lambda sc:LogisticRegression(max_iter=1000,class_weight='balanced').fit(sc.transform(Xtr),ytr).predict_proba(sc.transform(Xte))[:,1])(StandardScaler().fit(Xtr)))
    lam=df['lambda'].values
    ens=(rankdata(PB['oof'][m])+rankdata(lr[m]))/2
    def row(name,roc,pr,kr=None,kf=None):
        return dict(Model=name,ROC_AUC=round(roc,4),PR_AUC=round(pr,4),
                    Knee_recall=kr if kr is not None else '—',Knee_FAR=kf if kf is not None else '—')
    rkr,rkf=kfr(PB['oof'])
    lkr,lkf=kfr(lr)
    rows=[row('Random forest (3-seed, primary)',PB['roc'],PB['pr'],rkr,rkf),
          row('Logistic regression',roc_auc_score(yo,lr[m]),average_precision_score(yo,lr[m]),lkr,lkf),
          row('Single-feature lambda baseline',roc_auc_score(yo,lam[m]),average_precision_score(yo,lam[m]))]
    _,ekm=knee(yo,ens,np.quantile(ens,np.linspace(0.01,0.99,NTHETA)))
    rows.append(row('RF + LR ensemble',roc_auc_score(yo,ens),average_precision_score(yo,ens),round(ekm['Recall'],3),round(ekm['FAR'],3)))
    if cat is not None:
        om=omori(df,cat)
        _,okm=knee(PB['y'],om,np.quantile(om,np.linspace(0.01,0.99,NTHETA)))
        rows.append(row('Omori persistence kernel (benchmark)',roc_auc_score(PB['y'],om),
                        average_precision_score(PB['y'],om),round(okm['Recall'],3),round(okm['FAR'],3)))
        rank=(rankdata(PB['oof'][m])+rankdata(om[m]))/2
        rows.append(row('Rank ensemble RF + Omori',roc_auc_score(yo,rank),average_precision_score(yo,rank)))
    # --- ABLATION 1: RF with probability calibration ---
    cal_oof,_,_=oof_rf(df[df['end_time'].dt.year>=ERA].reset_index(drop=True),True)
    cm_=~np.isnan(cal_oof); yc=PB['y'][cm_]; pc=cal_oof[cm_]
    _,ckm=knee(yc,pc,grid)
    rows.append(row('RF with probability calibration',roc_auc_score(yc,pc),
                    average_precision_score(yc,pc),round(ckm['Recall'],3),round(ckm['FAR'],3)))
    # --- ABLATION 2: RF without embargo (gap = 0) ---
    dfB=df[df['end_time'].dt.year>=ERA].reset_index(drop=True)
    Xb=dfB[FEAT].values; yb=dfB['Label'].values.astype(int); ne=np.full(len(yb),np.nan)
    for tr,te in TimeSeriesSplit(n_splits=NSPLIT,gap=0).split(Xb):
        ne[te]=seed_scores(Xb[tr],yb[tr],Xb[te],False)
    nem=~np.isnan(ne)
    rows.append(row('RF without embargo (gap = 0)',roc_auc_score(yb[nem],ne[nem]),
                    average_precision_score(yb[nem],ne[nem])))
    t=pd.DataFrame(rows); t.to_csv(f'{OUTDIR}/table5_model_comparison.csv',index=False)
    rep.append("TABLE 5 — Model comparison, clustering benchmark & ablations (Protocol B)\n"+t.to_string(index=False)+"\n")
    if cat is not None:
        med=np.median(om[m]); lo=om[m]<=med; hi=~lo
        rep.append(f"   stratified: RF ROC low-cluster={roc_auc_score(yo[lo],PB['oof'][m][lo]):.4f}  high-cluster={roc_auc_score(yo[hi],PB['oof'][m][hi]):.4f}\n")
    return t
def table6(PB,df,cat,rep):
    m=PB['mask']; idx=np.where(m)[0]; sub=df.iloc[idx].copy()
    sub['fold']=[next(k for k,tr,te in PB['folds'] if i in set(te)) for i in idx]
    sub['score']=PB['oof'][m]
    om=omori(df,cat) if cat is not None else np.zeros(len(df))
    sub['om']=om[m]
    th=PB['th']; omth,_=knee(PB['y'][m],sub['om'].values,np.linspace(0.005,np.quantile(sub['om'],0.99),NTHETA)) if cat is not None else (0,None)
    rows=[]
    for k,g in sub.groupby('fold'):
        pos=g[g.Label==1]
        mm=cm(g.Label.values,(g.score.values>=th).astype(int))
        omrec=cm(g.Label.values,(g.om.values>=omth).astype(int))['Recall'] if cat is not None else np.nan
        rows.append(dict(Fold=k,pos_depth_km=round(pos.mean_z.mean(),1),pos_lambda=round(pos['lambda'].mean(),1),
            pos_rho=f"{pos.rho.mean():.1e}",RF_recall=round(mm['Recall'],3),Omori_recall=round(omrec,3)))
    t=pd.DataFrame(rows); t.to_csv(f'{OUTDIR}/table6_failing_window_character.csv',index=False)
    rep.append("TABLE 6 — Character of each Protocol B test window\n"+t.to_string(index=False)+"\n")
    return t
def event_audit(df,PB,cat,rep):
    """Tables 7 & 8: audit real events (model-independent geometry + OOF scores)."""
    m=PB['mask']; sub=df.iloc[np.where(m)[0]].copy(); sub['score']=PB['oof'][m]
    sub['end_time']=pd.to_datetime(sub['end_time'],utc=True)
    se=sub['end_time'].values.astype('datetime64[s]').astype(float)
    sla=np.radians(sub['centroid_lat'].values); slo=np.radians(sub['centroid_lon'].values); sc=sub['score'].values
    th=PB['th']
    ev=cat[(cat['mag']>=AUDIT_MW)].copy(); ev['time']=pd.to_datetime(ev['time'],utc=True)
    ev=ev[(ev['time']>=sub['end_time'].min())&(ev['time']<=sub['end_time'].max())].reset_index(drop=True)
    out=[]
    for e in ev.itertuples():
        te=pd.Timestamp(e.time).timestamp(); la=radians(e.latitude); lo=radians(e.longitude)
        db=(te-se)/86400.0; sel=(db>0)&(db<=15)
        cov=0; alerted=False
        if sel.any():
            d=6371*2*np.arcsin(np.sqrt(np.sin((sla[sel]-la)/2)**2+np.cos(la)*np.cos(sla[sel])*np.sin((slo[sel]-lo)/2)**2))
            near=sc[sel][d<=100]; cov=len(near); alerted=bool((near>=th).any())
        out.append(dict(mag=e.mag,depth=e.depth,cov=cov,
            outcome='alerted' if alerted else('covered_missed' if cov>0 else 'no_coverage')))
    A=pd.DataFrame(out)
    # Table 8 taxonomy
    def stratum(name,g):
        n=len(g); return dict(Stratum=name,n=n,
            Alerted=round(100*(g.outcome=='alerted').mean(),1),
            Covered_sub=round(100*(g.outcome=='covered_missed').mean(),1),
            No_coverage=round(100*(g.outcome=='no_coverage').mean(),1),
            Cond_detection=round(100*(g[g['cov']>0].outcome=='alerted').mean(),1) if (g['cov']>0).any() else 0.0)
    rows=[stratum('All events',A)]
    for lab,lo,hi in [('Mw 4.5-4.9',4.5,5.0),('Mw 5.0-5.4',5.0,5.5),('Mw 5.5-5.9',5.5,6.0),('Mw 6.0+',6.0,10)]:
        rows.append(stratum(lab,A[(A.mag>=lo)&(A.mag<hi)]))
    for lab,lo,hi in [('Depth <35 km',0,35),('Depth 35-70 km',35,70),('Depth >70 km',70,800)]:
        rows.append(stratum(lab,A[(A.depth>=lo)&(A.depth<hi)]))
    t8=pd.DataFrame(rows); t8.to_csv(f'{OUTDIR}/table8_event_audit_taxonomy.csv',index=False)
    rep.append(f"TABLE 8 — Outcome taxonomy, all {len(A)} Mw>={AUDIT_MW} events\n"+t8.to_string(index=False)+"\n")
    # Table 7 big events
    big=ev[ev['mag']>=BIG_MW].copy()
    b=[]
    for e in big.itertuples():
        te=pd.Timestamp(e.time).timestamp(); la=radians(e.latitude); lo=radians(e.longitude)
        db=(te-se)/86400.0
        # 60-day window
        sel60=(db>0)&(db<=60); cov60=0
        # 15-day window (matches the label horizon; scores/alerts refer to this)
        sel15=(db>0)&(db<=15); cov15=0
        maxs=np.nan; alerted=False
        if sel60.any():
            d60=6371*2*np.arcsin(np.sqrt(np.sin((sla[sel60]-la)/2)**2+np.cos(la)*np.cos(sla[sel60])*np.sin((slo[sel60]-lo)/2)**2))
            cov60=int((d60<=100).sum())
        if sel15.any():
            d15=6371*2*np.arcsin(np.sqrt(np.sin((sla[sel15]-la)/2)**2+np.cos(la)*np.cos(sla[sel15])*np.sin((slo[sel15]-lo)/2)**2))
            near15=sc[sel15][d15<=100]; cov15=len(near15)
            if cov15: maxs=float(near15.max()); alerted=bool((near15>=th).any())
        b.append(dict(date=str(pd.Timestamp(e.time).date()),mag=e.mag,depth_km=round(e.depth,0),
            seq_60d=cov60,seq_15d=cov15,max_score_15d=round(maxs,3) if cov15 else None,
            alerted='100%' if alerted else ('0%' if cov15 else '—'),
            outcome='alerted' if alerted else('missed' if cov15 else 'no local sequence')))
    t7=pd.DataFrame(b).sort_values('mag',ascending=False)
    t7.to_csv(f'{OUTDIR}/table7_event_case_studies.csv',index=False)
    rep.append(f"TABLE 7 — Event case studies, {len(t7)} Mw>={BIG_MW} mainshocks "
               f"({(t7.outcome=='no local sequence').sum()} with no local sequence)\n"+t7.to_string(index=False)+"\n")
    return t7,t8

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--calibrate',action='store_true',help='Section 4.5 calibrated ablation instead of primary')
    ap.add_argument('--features',default=FEATURE_FILE); ap.add_argument('--catalog',default=CATALOG_FILE)
    a=ap.parse_args(); os.makedirs(OUTDIR,exist_ok=True)
    df=pd.read_csv(a.features); df['end_time']=pd.to_datetime(df['end_time'],format='mixed',utc=True)
    df=df.sort_values('seq_id').reset_index(drop=True)
    try:
        cat=pd.read_csv(a.catalog); cat['time']=pd.to_datetime(cat['time'],format='mixed',utc=True)
    except Exception as e:
        cat=None; print(f"(catalog not loaded, Omori/audit skipped: {e})")
    rep=[f"MANUSCRIPT TABLE REGENERATION REPORT",
         f"mode = {'CALIBRATED ABLATION (Sec 4.5)' if a.calibrate else 'UNCALIBRATED PRIMARY (Table 2)'}",
         f"sequences={len(df)} positives={int(df.Label.sum())} ({100*df.Label.mean():.2f}%)","="*70]
    table1(df,rep)
    PA=protocol(df,'A',a.calibrate)
    dfB=df[df['end_time'].dt.year>=ERA].reset_index(drop=True)
    PB=protocol(dfB,'B',a.calibrate)
    table2(PA,PB,rep); table3(PB,rep); table4(PB,dfB,rep); table5(PB,dfB,cat,rep)
    if cat is not None:
        table6(PB,dfB,cat,rep); event_audit(dfB,PB,cat,rep)
    open(f'{OUTDIR}/TABLES_REPORT.txt','w').write('\n'.join(rep))
    print('\n'.join(rep)); print(f"\nAll tables written to ./{OUTDIR}/")

if __name__=='__main__': main()
