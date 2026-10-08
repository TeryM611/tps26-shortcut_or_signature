#!/usr/bin/env python3
"""Experiments I & J: model-agnosticity (XGBoost reproduces the RF decomposition)
and SHAP explainability, compared against v1's permutation importance.
Runs in the isolated `hdml_ext` conda env (xgboost, shap, sklearn).
Self-contained: re-implements the minimal data loading / grouping / host-grouped
CV needed, rather than importing the base-env pipeline (different Python envs)."""
import os, re, sys, json, glob
import numpy as np, pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = os.environ.get('WORK', 'data')
OUT = os.environ.get('OUT', 'results')
os.makedirs(OUT, exist_ok=True)
RNG = 20260710
NPC = int(os.environ.get('NPC', '20000'))
RESJSON = os.path.join(OUT, 'results_xgb_shap.json')

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
    return ip.startswith('192.168.') or ip.startswith('10.') or any(ip.startswith('172.%d.' % o) for o in range(16, 32))

def external_group(row):
    s, d = row['src_ip'], row['dst_ip']
    ext = d if is_internal(s) and not is_internal(d) else (s if is_internal(d) else s)
    parts = ext.split('.')
    return '.'.join(parts[:3]) if len(parts) >= 3 else ext

def _day_from_tag(tag):
    m = re.match(r'honey(\d+)', tag)
    if m: return int(m.group(1))
    if tag == 'honey24': return 24
    if tag == 'honey31': return 31
    return None

def load_honeypot_pool():
    files = sorted(glob.glob(os.path.join(WORK, 'flows_honey*.csv')))
    frames = []
    for f in files:
        tag = os.path.basename(f)[len('flows_'):-4]
        if tag == 'honey16_s2':
            continue
        day = _day_from_tag(tag)
        if day is None:
            continue
        df = pd.read_csv(f)
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool['label'] = 1
    return pool

def load_real_tue():
    df = pd.read_csv(os.path.join(WORK, 'flows_real_tue.csv'))
    df['label'] = 0
    return df

def prep(df):
    df = df[(df['tot_fwd_pkts'] + df['tot_bwd_pkts']) >= 2].reset_index(drop=True)
    df['group'] = df.apply(external_group, axis=1)
    return df

def subsample(df, n, seed=RNG, stratify_col='label'):
    rng = np.random.default_rng(seed)
    parts = []
    for v, g in df.groupby(stratify_col):
        if len(g) > n:
            g = g.iloc[rng.choice(len(g), n, replace=False)]
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)

def grouped_kfold_indices(groups, k=5, seed=RNG):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    folds = [uniq[i::k] for i in range(k)]
    for i in range(k):
        test_g = set(folds[i].tolist())
        test_mask = np.array([g in test_g for g in groups])
        yield np.where(~test_mask)[0], np.where(test_mask)[0]

def auc_score(y_true, score):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, score)

def xgb_cv_auc(df, feats, groups, k=5, seed=RNG):
    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(-1).astype(float).values
    y = df['label'].values.astype(int)
    aucs = []
    for tr, te in grouped_kfold_indices(groups, k=k, seed=seed):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric='auc', n_jobs=4, random_state=seed, verbosity=0)
        m.fit(X[tr], y[tr])
        s = m.predict_proba(X[te])[:, 1]
        aucs.append(auc_score(y[te], s))
    return np.array(aucs)

def main():
    results = json.load(open(RESJSON)) if os.path.exists(RESJSON) else {}
    print("[load] honeypot pool + CIC-IDS2017 ...")
    honey = prep(load_honeypot_pool())
    real = prep(load_real_tue())
    full = pd.concat([honey.assign(label=1), real.assign(label=0)], ignore_index=True)
    sub = subsample(full, NPC, stratify_col='label')
    groups = sub['group'].values
    print(f"xgboost subsample: {len(sub)} flows, {len(np.unique(groups))} external groups")

    if 'I_xgb_decomp' not in results:
        print("[I] XGBoost decomposition (model-agnosticity check) ...")
        i_full = xgb_cv_auc(sub, ALL_FEATS, groups)
        i_env = xgb_cv_auc(sub, ENV_FEATS, groups)
        i_beh = xgb_cv_auc(sub, BEH_FEATS, groups)
        results['I_xgb_decomp'] = {
            'n_used': int(len(sub)),
            'full_auc': [float(i_full.mean()), float(i_full.std())],
            'env_auc': [float(i_env.mean()), float(i_env.std())],
            'beh_auc': [float(i_beh.mean()), float(i_beh.std())]}
        print("   XGB full %.4f env %.4f beh %.4f" % (i_full.mean(), i_env.mean(), i_beh.mean()))
        json.dump(results, open(RESJSON, 'w'), indent=2)

    if 'J_shap' not in results:
        print("[J] SHAP explainability on grouped hold-out ...")
        g_un = np.unique(groups); rng = np.random.default_rng(RNG); rng.shuffle(g_un)
        te_g = set(g_un[:max(1, len(g_un)//5)].tolist())
        te_mask = np.array([g in te_g for g in groups])
        X = sub[ALL_FEATS].replace([np.inf, -np.inf], np.nan).fillna(-1).astype(float).values
        y = sub['label'].values.astype(int)
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric='auc', n_jobs=4, random_state=RNG, verbosity=0)
        m.fit(X[~te_mask], y[~te_mask])
        holdout_auc = float(auc_score(y[te_mask], m.predict_proba(X[te_mask])[:, 1]))

        explainer = shap.TreeExplainer(m)
        # subsample the holdout for SHAP compute cost
        Xte = X[te_mask]
        if len(Xte) > 4000:
            idx = np.random.default_rng(RNG).choice(len(Xte), 4000, replace=False)
            Xte_s = Xte[idx]
        else:
            Xte_s = Xte
        sv = explainer.shap_values(Xte_s)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)
        ranking = [(ALL_FEATS[i], float(mean_abs[i])) for i in order]
        env_mass = sum(v for k, v in ranking if k in ENV_FEATS)
        beh_mass = sum(v for k, v in ranking if k in BEH_FEATS)
        tot = env_mass + beh_mass if (env_mass + beh_mass) > 0 else 1
        results['J_shap'] = {
            'holdout_auc': holdout_auc,
            'ranking_top20': ranking[:20],
            'env_mass_share': float(env_mass / tot)}
        print("   holdout auc %.4f  SHAP env share %.3f  top5:" % (holdout_auc, env_mass/tot), ranking[:5])
        json.dump(results, open(RESJSON, 'w'), indent=2)

        # Figures
        plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
        shap.summary_plot(sv, Xte_s, feature_names=ALL_FEATS, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_shap_summary.png')); plt.close()

        top_env_feat = next((k for k, v in ranking if k in ENV_FEATS), ALL_FEATS[0])
        fi = ALL_FEATS.index(top_env_feat)
        plt.figure(figsize=(6.2, 4.0))
        shap.dependence_plot(fi, sv, Xte_s, feature_names=ALL_FEATS, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_shap_dependence.png')); plt.close()
        print("   figures written: fig_shap_summary.png, fig_shap_dependence.png (top env feat: %s)" % top_env_feat)

    print("[done] experiments_xgb_shap complete")
    return results

if __name__ == '__main__':
    r = main()
    print(json.dumps(r, indent=2)[:2000])
