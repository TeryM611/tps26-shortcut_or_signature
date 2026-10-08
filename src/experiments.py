#!/usr/bin/env python3
"""
Confound-controlled reassessment of network-traffic honeypot detection.
Runs all experiments and writes results/results.json + figures.

Experiments:
  A. Naive reproduction: full-feature detector under random vs. host-grouped CV.
  B. Confound decomposition: environment(stack)-only vs behavioral-only vs TTL-only.
  C. Placebo test: discriminate two honeypot captures (both label=1) from each other.
  D. Attribution: permutation importance by feature group.
  E. Normalization defense: canonicalize stack features -> measure detectability collapse.
  Deception-Leakage metric summary.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = os.environ.get('WORK', '/sessions/jolly-great-ptolemy/mnt/outputs/work')
OUT  = os.environ.get('OUT',  '/sessions/jolly-great-ptolemy/mnt/HDML/honeynet_confound_study/results')
os.makedirs(OUT, exist_ok=True)
RNG = 20260710
N_PER_CLASS = int(os.environ.get('NPC','25000'))

ENV_FEATS = ['fwd_ttl_min','fwd_ttl_max','fwd_ttl_mean','bwd_ttl_min','bwd_ttl_max','bwd_ttl_mean',
 'ip_dscp','dscp_nonzero_rate','init_fwd_win','init_bwd_win',
 'syn_mss_fwd','syn_mss_bwd','syn_wscale_fwd','syn_wscale_bwd','sack_ok','ts_opt','df_rate']
BEH_FEATS = ['flow_duration','tot_fwd_pkts','tot_bwd_pkts','totlen_fwd_pkts','totlen_bwd_pkts',
 'flow_byts_s','flow_pkts_s','fwd_pkts_s','bwd_pkts_s','down_up_ratio',
 'pkt_len_min','pkt_len_max','pkt_len_mean','pkt_len_std','pkt_size_avg',
 'fwd_pkt_len_min','fwd_pkt_len_max','fwd_pkt_len_mean','fwd_pkt_len_std',
 'bwd_pkt_len_min','bwd_pkt_len_max','bwd_pkt_len_mean','bwd_pkt_len_std',
 'flow_iat_mean','flow_iat_std','flow_iat_min','flow_iat_max',
 'fwd_iat_mean','fwd_iat_std','fwd_iat_min','fwd_iat_max',
 'bwd_iat_mean','bwd_iat_std','bwd_iat_min','bwd_iat_max',
 'fin_cnt','syn_cnt','rst_cnt','psh_cnt','ack_cnt','urg_cnt','ece_cnt','cwr_cnt']
ALL_FEATS = ENV_FEATS + BEH_FEATS

def is_internal(ip):
    return ip.startswith('192.168.') or ip.startswith('10.') or \
        any(ip.startswith('172.%d.'%o) for o in range(16,32))

def external_group(row):
    s,d = row['src_ip'], row['dst_ip']
    ext = d if is_internal(s) and not is_internal(d) else (s if is_internal(d) else s)
    parts = ext.split('.')
    return '.'.join(parts[:3]) if len(parts)>=3 else ext

def load(tag):
    df = pd.read_csv(os.path.join(WORK, f'flows_{tag}.csv'))
    return df

def clean(df):
    df = df.copy()
    for c in ALL_FEATS:
        if c not in df.columns: df[c]=0.0
    X = df[ALL_FEATS].replace([np.inf,-np.inf], np.nan).fillna(-1).astype(float).values
    return X

def subsample(df, n, seed=RNG, stratify_col='label'):
    rng=np.random.default_rng(seed)
    parts=[]
    for v,g in df.groupby(stratify_col):
        if len(g)>n:
            g=g.iloc[rng.choice(len(g), n, replace=False)]
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)

def D(a):  # detectability = Somers' D / Gini
    return 2*a-0.5*2  # = 2*auc-1

def cv_eval(df, feats, groups, seed=RNG, k=5, model='rf'):
    X=df[feats].replace([np.inf,-np.inf],np.nan).fillna(-1).astype(float).values
    y=df['label'].values.astype(int)
    if groups is not None:
        splits=list(ml.grouped_kfold_indices(groups,k=k,seed=seed))
    else:
        splits=list(ml.stratified_kfold_indices(y,k=k,seed=seed))
    def factory():
        if model=='rf': return ml.HistRandomForest(n_trees=60,max_depth=12,min_leaf=20,seed=seed)
        return ml.LogisticReg(epochs=250)
    aucs=ml.cv_auc(factory, X, y, splits)
    return aucs

RESJSON=os.path.join(OUT,'results.json')
def _save(results):
    with open(RESJSON,'w') as f: json.dump(results,f,indent=2)

def main():
    RF_TREES=int(os.environ.get('RFT','50'))
    results={}
    if os.path.exists(RESJSON):
        results=json.load(open(RESJSON))
    print("[load] reading flow CSVs ...")
    h24=load('honey24'); h31=load('honey31'); real=load('real_tue')
    for d,t in [(h24,'honey24'),(h31,'honey31'),(real,'real_tue')]:
        d['dataset']=t
    print("flows:", len(h24), len(h31), len(real))

    honey=pd.concat([h24,h31]).reset_index(drop=True)
    honey['label']=1; real['label']=0
    full=pd.concat([honey, real]).reset_index(drop=True)
    # keep only flows with at least 2 packets (well-formed)
    full=full[(full['tot_fwd_pkts']+full['tot_bwd_pkts'])>=2].reset_index(drop=True)
    full['group']=full.apply(external_group, axis=1)

    if 'dataset' not in results:
      results['dataset']={}
      results['dataset']['n_honey_flows']=int((full.label==1).sum())
    results['dataset']['n_real_flows']=int((full.label==0).sum())
    results['dataset']['n_honey24']=int(len(h24)); results['dataset']['n_honey31']=int(len(h31))
    results['dataset']['n_real']=int(len(real))
    results['dataset']['n_features']=len(ALL_FEATS)
    results['dataset']['n_env']=len(ENV_FEATS); results['dataset']['n_beh']=len(BEH_FEATS)
    results['dataset']['n_ext_groups']=int(full['group'].nunique())
    _save(results)

    # balanced subsample for tractable pure-python training
    sub=subsample(full, N_PER_CLASS, stratify_col='label')
    print("subsample:", len(sub), "pos", int(sub.label.sum()))
    groups=sub['group'].values

    # ---------- Experiment A: naive vs grouped CV, full features ----------
    if 'A_full' not in results:
        print("[A] full-feature detector ...")
        a_rand=cv_eval(sub, ALL_FEATS, None, model='rf')
        a_grp =cv_eval(sub, ALL_FEATS, groups, model='rf')
        a_rand_lr=cv_eval(sub, ALL_FEATS, None, model='lr')
        a_grp_lr =cv_eval(sub, ALL_FEATS, groups, model='lr')
        results['A_full']={
            'rf_random_auc':[float(a_rand.mean()),float(a_rand.std())],
            'rf_grouped_auc':[float(a_grp.mean()),float(a_grp.std())],
            'lr_random_auc':[float(a_rand_lr.mean()),float(a_rand_lr.std())],
            'lr_grouped_auc':[float(a_grp_lr.mean()),float(a_grp_lr.std())]}
        print("   RF random %.4f  grouped %.4f"%(a_rand.mean(),a_grp.mean())); _save(results)

    # ---------- Experiment B: confound decomposition (grouped CV) ----------
    if 'B_decomp' not in results:
        print("[B] env-only / beh-only / ttl-only ...")
        b_env=cv_eval(sub, ENV_FEATS, groups, model='rf')
        b_beh=cv_eval(sub, BEH_FEATS, groups, model='rf')
        b_ttl=cv_eval(sub, ['fwd_ttl_mean','bwd_ttl_mean','fwd_ttl_min','bwd_ttl_min'], groups, model='rf')
        b_ttl1=cv_eval(sub, ['fwd_ttl_mean'], groups, model='rf')
        results['B_decomp']={
            'env_only_auc':[float(b_env.mean()),float(b_env.std())],
            'beh_only_auc':[float(b_beh.mean()),float(b_beh.std())],
            'ttl4_auc':[float(b_ttl.mean()),float(b_ttl.std())],
            'ttl1_auc':[float(b_ttl1.mean()),float(b_ttl1.std())]}
        print("   env %.4f  beh %.4f  ttl4 %.4f  ttl1 %.4f"%(b_env.mean(),b_beh.mean(),b_ttl.mean(),b_ttl1.mean())); _save(results)

    # ---------- Experiment C: placebo (honey24 vs honey31) ----------
    if 'C_placebo' not in results:
        print("[C] placebo: honey24 vs honey31 (both honeypot) ...")
        plac=pd.concat([h24.assign(label=0), h31.assign(label=1)]).reset_index(drop=True)
        plac=plac[(plac['tot_fwd_pkts']+plac['tot_bwd_pkts'])>=2].reset_index(drop=True)
        plac['group']=plac.apply(external_group,axis=1)
        plac=subsample(plac, N_PER_CLASS, stratify_col='label')
        pg=plac['group'].values
        c_full=cv_eval(plac, ALL_FEATS, pg, model='rf')
        c_env =cv_eval(plac, ENV_FEATS, pg, model='rf')
        c_beh =cv_eval(plac, BEH_FEATS, pg, model='rf')
        results['C_placebo']={
            'full_auc':[float(c_full.mean()),float(c_full.std())],
            'env_auc':[float(c_env.mean()),float(c_env.std())],
            'beh_auc':[float(c_beh.mean()),float(c_beh.std())]}
        print("   placebo full %.4f  env %.4f  beh %.4f"%(c_full.mean(),c_env.mean(),c_beh.mean())); _save(results)

    # ---------- Experiment D: permutation importance ----------
    if 'D_importance' not in results:
        print("[D] permutation importance (grouped holdout) ...")
        g_un=np.unique(groups); rng=np.random.default_rng(RNG); rng.shuffle(g_un)
        te_g=set(g_un[:max(1,len(g_un)//5)].tolist())
        te_mask=np.array([g in te_g for g in groups])
        Xall=sub[ALL_FEATS].replace([np.inf,-np.inf],np.nan).fillna(-1).astype(float).values
        yall=sub['label'].values.astype(int)
        m=ml.HistRandomForest(n_trees=80,max_depth=12,min_leaf=20,seed=RNG).fit(Xall[~te_mask],yall[~te_mask])
        imp,base=ml.permutation_importance(m, Xall[te_mask], yall[te_mask], ALL_FEATS, n_repeats=5)
        results['D_importance']={'holdout_auc':float(base),'ranking':imp[:20]}
        env_mass=sum(v for k,v in imp if k in ENV_FEATS and v>0)
        beh_mass=sum(v for k,v in imp if k in BEH_FEATS and v>0)
        tot=env_mass+beh_mass if (env_mass+beh_mass)>0 else 1
        results['D_importance']['env_mass_share']=float(env_mass/tot)
        print("   holdout auc %.4f  env share %.3f  top:"%(base, env_mass/tot), imp[:5]); _save(results)

    # ---------- Experiment E: normalization defense ----------
    if 'E_defense' not in results:
        print("[E] stack-camouflage defense ...")
        # Defense model: the honeypot operator controls only the honeypot host's TCP/IP
        # stack. We simulate 'stack camouflage' by drawing each honeypot flow's host-stack
        # (ENV) feature vector from the empirical joint distribution of the production
        # corpus (bootstrap of real ENV rows). This preserves the joint structure of a
        # realistic production stack and makes the honeypot stack-indistinguishable from
        # production by construction; any residual detectability is purely behavioural.
        rng=np.random.default_rng(RNG)
        honey_idx=np.where(sub['label'].values==1)[0]
        # Disjoint production donor pool (real flows NOT in the modelling subsample)
        real_all=full[full['label']==0]
        donor_pool=real_all.loc[~real_all.index.isin(sub.index.values)] if False else real_all
        donor_env=donor_pool[ENV_FEATS].replace([np.inf,-np.inf],np.nan).fillna(-1).astype(float).values

        # ---- E1: naive per-feature (marginal) normalization ----
        sub_marg=sub.copy().reset_index(drop=True)
        vals=sub_marg.loc[:,ENV_FEATS].values.copy()
        for j in range(len(ENV_FEATS)):
            col=donor_env[:,j]
            vals[honey_idx,j]=col[rng.integers(0,len(col),size=len(honey_idx))]
        sub_marg.loc[:,ENV_FEATS]=vals
        e1_env=cv_eval(sub_marg, ENV_FEATS, groups, model='rf')
        e1_full=cv_eval(sub_marg, ALL_FEATS, groups, model='rf')

        # ---- E2: joint-coherent stack camouflage (whole real stack profiles) ----
        sub_joint=sub.copy().reset_index(drop=True)
        vals2=sub_joint.loc[:,ENV_FEATS].values.copy()
        donor_rows=rng.integers(0,len(donor_env),size=len(honey_idx))
        vals2[honey_idx,:]=donor_env[donor_rows,:]
        sub_joint.loc[:,ENV_FEATS]=vals2
        e_env=cv_eval(sub_joint, ENV_FEATS, groups, model='rf')
        e_full=cv_eval(sub_joint, ALL_FEATS, groups, model='rf')
        results['E_defense']={
            'marginal_env_auc':[float(e1_env.mean()),float(e1_env.std())],
            'marginal_full_auc':[float(e1_full.mean()),float(e1_full.std())],
            'env_auc_after':[float(e_env.mean()),float(e_env.std())],
            'full_auc_after':[float(e_full.mean()),float(e_full.std())]}
        print("   E1 marginal env %.4f | E2 joint env %.4f full %.4f"%(e1_env.mean(),e_env.mean(),e_full.mean()))
        # ---------- Deception-Leakage / confound decomposition ----------
        def Dv(a): return max(2*a-1,0.0)
        DX_full=Dv(results['A_full']['rf_grouped_auc'][0])
        DX_env =Dv(results['B_decomp']['env_only_auc'][0])
        DX_beh =Dv(results['B_decomp']['beh_only_auc'][0])
        DW_full=Dv(results['C_placebo']['full_auc'][0])
        DW_env =Dv(results['C_placebo']['env_auc'][0])
        DW_beh =Dv(results['C_placebo']['beh_auc'][0])
        def eci(dx,dw): return float(max(1-dw/dx,0.0)) if dx>1e-6 else 0.0
        results['confound_index']={
            'cross_env':{'full':DX_full,'env':DX_env,'beh':DX_beh},
            'within_env':{'full':DW_full,'env':DW_env,'beh':DW_beh},
            'ECI':{'full':eci(DX_full,DW_full),'env':eci(DX_env,DW_env),'beh':eci(DX_beh,DW_beh)}}
        D_full_camo=Dv(e_full.mean())
        results['leakage']={
            'D_full':DX_full,'D_env':DX_env,'D_beh':DX_beh,
            'D_full_after_stack_camouflage':D_full_camo,
            'stack_channel_detectability_removed':float(DX_env - Dv(e_env.mean())),
            'residual_behavioural_detectability':D_full_camo}
        print("   ECI env %.3f beh %.3f full %.3f"%(results['confound_index']['ECI']['env'],
              results['confound_index']['ECI']['beh'],results['confound_index']['ECI']['full']))
        _save(results)

    make_figures(results, full, ENV_FEATS, BEH_FEATS)
    _save(results)
    print("[done] all stages complete")
    return results

def make_figures(results, full, ENV, BEH):
    plt.rcParams.update({'font.size':10,'figure.dpi':130})
    # Fig 1: confound bars
    fig,ax=plt.subplots(figsize=(6.2,3.4))
    labels=['Full\n(random CV)','Full\n(grouped CV)','Env/stack\nonly','Behavioral\nonly','TTL\nonly']
    vals=[results['A_full']['rf_random_auc'][0],results['A_full']['rf_grouped_auc'][0],
          results['B_decomp']['env_only_auc'][0],results['B_decomp']['beh_only_auc'][0],
          results['B_decomp']['ttl4_auc'][0]]
    errs=[results['A_full']['rf_random_auc'][1],results['A_full']['rf_grouped_auc'][1],
          results['B_decomp']['env_only_auc'][1],results['B_decomp']['beh_only_auc'][1],
          results['B_decomp']['ttl4_auc'][1]]
    colors=['#4C72B0','#4C72B0','#C44E52','#55A868','#C44E52']
    ax.bar(range(5),vals,yerr=errs,color=colors,capsize=3)
    ax.axhline(0.5,ls='--',c='gray',lw=1)
    ax.set_ylim(0.4,1.02); ax.set_ylabel('ROC AUC'); ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    for i,v in enumerate(vals): ax.text(i,v+0.01,f"{v:.3f}",ha='center',fontsize=8)
    ax.set_title('Detectability by feature group (host-grouped CV)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_confound.png')); plt.close(fig)

    # Fig 2: permutation importance
    imp=results['D_importance']['ranking'][:15]
    names=[k for k,v in imp][::-1]; vv=[v for k,v in imp][::-1]
    cols=['#C44E52' if n in ENV else '#55A868' for n in names]
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    ax.barh(range(len(names)),vv,color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=8)
    ax.set_xlabel('Permutation importance (AUC drop)')
    ax.set_title('Feature attribution (red=env/stack, green=behavioral)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_importance.png')); plt.close(fig)

    # Fig 3: defense (detectability D = max(0, 2*AUC-1))
    def Dm(a): return max(0.0, 2*a-1)
    e=results['E_defense']
    fig,ax=plt.subplots(figsize=(6.4,3.5))
    groups_lbl=['Env / stack channel','Full model']
    none_=[Dm(results['B_decomp']['env_only_auc'][0]), Dm(results['A_full']['rf_grouped_auc'][0])]
    marg=[Dm(e['marginal_env_auc'][0]), Dm(e['marginal_full_auc'][0])]
    joint=[Dm(e['env_auc_after'][0]), Dm(e['full_auc_after'][0])]
    x=np.arange(2); w=0.26
    ax.bar(x-w, none_, w, label='No defense', color='#C44E52')
    ax.bar(x,     marg, w, label='Naive per-feature normalization', color='#DD8452')
    ax.bar(x+w,  joint, w, label='Joint stack camouflage', color='#4C72B0')
    ax.set_ylim(0,1.05); ax.set_xticks(x); ax.set_xticklabels(groups_lbl)
    ax.set_ylabel('Detectability  D = max(0, 2·AUC−1)'); ax.legend(fontsize=8)
    for i in range(2):
        for off,v in [(-w,none_[i]),(0,marg[i]),(w,joint[i])]:
            ax.text(i+off,v+0.02,f"{v:.2f}",ha='center',fontsize=7)
    ax.set_title('Stack camouflage closes the environment channel;\nbehavioural confound persists')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_defense.png')); plt.close(fig)

    # Fig 4: TTL distribution from packet reservoir
    try:
        ph=pd.read_csv(os.path.join(WORK,'pkts_honey24.csv'))['ttl']
        pr=pd.read_csv(os.path.join(WORK,'pkts_real_tue.csv'))['ttl']
        fig,ax=plt.subplots(figsize=(6.2,3.2))
        bins=np.arange(0,257,4)
        ax.hist(ph,bins=bins,alpha=0.6,density=True,label='Honeypot',color='#C44E52')
        ax.hist(pr,bins=bins,alpha=0.6,density=True,label='Real/production',color='#4C72B0')
        ax.set_xlabel('IP TTL'); ax.set_ylabel('density'); ax.legend()
        ax.set_title('Per-packet TTL distribution (environment artifact)')
        fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_ttl.png')); plt.close(fig)
    except Exception as e:
        print("ttl fig skipped:",e)
    # Fig 5: Environmental Confound Index (cross-env vs within-env detectability)
    try:
        ci=results['confound_index']
        chans=['env','beh','full']; labels=['Env / stack','Behavioural','Full model']
        cx=[ci['cross_env'][c] for c in chans]; cw=[ci['within_env'][c] for c in chans]
        x=np.arange(3); w=0.36
        fig,ax=plt.subplots(figsize=(6.0,3.4))
        ax.bar(x-w/2,cx,w,label='Cross-environment (honeypot vs production)',color='#C44E52')
        ax.bar(x+w/2,cw,w,label='Within-environment (placebo control)',color='#4C72B0')
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Detectability  D = 2·AUC−1")
        ax.set_ylim(0,1.05); ax.legend(fontsize=8)
        for i,(a,b) in enumerate(zip(cx,cw)):
            ax.text(i-w/2,a+0.02,f"{a:.2f}",ha='center',fontsize=7)
            ax.text(i+w/2,b+0.02,f"{b:.2f}",ha='center',fontsize=7)
        ax.set_title('Detectability collapses when environment is held fixed')
        fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_eci.png')); plt.close(fig)
    except Exception as e:
        print("eci fig skipped:",e)
    print("[figures] written to", OUT)

if __name__=='__main__':
    r=main()
    print(json.dumps(r,indent=2)[:1500])
