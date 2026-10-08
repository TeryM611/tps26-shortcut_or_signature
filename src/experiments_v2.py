#!/usr/bin/env python3
"""Extended experiments (F-H): multi-day honeypot pool, day-grouped placebo
distribution, and generalization to a second independent production-like
corpus (FIRST-2015). Builds on v1's experiments.py / ml.py."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml
from experiments import ENV_FEATS, BEH_FEATS, ALL_FEATS, external_group, cv_eval, subsample, RNG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_pool import load_honeypot_pool, load_first2015, load_real_tue

WORK = os.environ.get('WORK', 'data')
OUT = os.environ.get('OUT', 'results')
os.makedirs(OUT, exist_ok=True)
NPC = int(os.environ.get('NPC', '20000'))
NPC_PAIR = int(os.environ.get('NPC_PAIR', '3000'))
N_PAIRS = int(os.environ.get('NPAIRS', '50'))

RESJSON = os.path.join(OUT, 'results_v2.json')

def _save(r):
    json.dump(r, open(RESJSON, 'w'), indent=2)

def prep(df):
    df = df[(df['tot_fwd_pkts'] + df['tot_bwd_pkts']) >= 2].reset_index(drop=True)
    df['group'] = df.apply(external_group, axis=1)
    return df

def main():
    results = json.load(open(RESJSON)) if os.path.exists(RESJSON) else {}
    print("[load] honeypot pool (all days) ...")
    honey_pool, days_seen = load_honeypot_pool(WORK)
    honey_pool = prep(honey_pool)
    real_tue = prep(load_real_tue(WORK))
    print(f"honeypot pool: {len(honey_pool)} flows across {len(days_seen)} days; real_tue: {len(real_tue)}")

    results.setdefault('pool_meta', {})
    results['pool_meta']['n_honey_pool_flows'] = int(len(honey_pool))
    results['pool_meta']['n_honey_days'] = int(len(days_seen))
    results['pool_meta']['honey_days'] = sorted(str(d) for d in days_seen.keys())
    results['pool_meta']['day_segment_map'] = {str(k): v for k, v in days_seen.items()}
    _save(results)

    if 'F_full_pool' not in results:
        print("[F] full honeypot pool vs CIC-IDS2017 ...")
        fdf = pd.concat([honey_pool.assign(label=1), real_tue.assign(label=0)], ignore_index=True)
        sub = subsample(fdf, NPC, stratify_col='label')
        groups = sub['group'].values
        f_full = cv_eval(sub, ALL_FEATS, groups, model='rf')
        f_env = cv_eval(sub, ENV_FEATS, groups, model='rf')
        f_beh = cv_eval(sub, BEH_FEATS, groups, model='rf')
        f_ttl = cv_eval(sub, ['fwd_ttl_mean', 'bwd_ttl_mean', 'fwd_ttl_min', 'bwd_ttl_min'], groups, model='rf')
        results['F_full_pool'] = {
            'n_pool_used': int(len(sub)),
            'full_auc': [float(f_full.mean()), float(f_full.std())],
            'env_auc': [float(f_env.mean()), float(f_env.std())],
            'beh_auc': [float(f_beh.mean()), float(f_beh.std())],
            'ttl_auc': [float(f_ttl.mean()), float(f_ttl.std())]}
        print("   full %.4f env %.4f beh %.4f ttl %.4f" % (f_full.mean(), f_env.mean(), f_beh.mean(), f_ttl.mean()))
        _save(results)

    if 'G_day_placebo' not in results:
        print("[G] day-grouped placebo across many honeypot-day pairs ...")
        days = sorted(honey_pool['capture_day'].unique().tolist())
        rng = np.random.default_rng(RNG)
        all_pairs = [(a, b) for i, a in enumerate(days) for b in days[i+1:]]
        rng.shuffle(all_pairs)
        pairs = all_pairs[:N_PAIRS]
        full_aucs = []; env_aucs = []; beh_aucs = []; used_pairs = []
        for (da, db) in pairs:
            A = honey_pool[honey_pool['capture_day'] == da]
            B = honey_pool[honey_pool['capture_day'] == db]
            if len(A) < 50 or len(B) < 50:
                continue
            plac = pd.concat([A.assign(label=0), B.assign(label=1)], ignore_index=True)
            n_each = min(NPC_PAIR, plac[plac.label == 0].shape[0], plac[plac.label == 1].shape[0])
            if n_each < 30:
                continue
            plac_sub = subsample(plac, n_each, stratify_col='label')
            pg = plac_sub['group'].values
            if len(np.unique(pg)) < 10:
                continue
            try:
                c_full = cv_eval(plac_sub, ALL_FEATS, pg, model='rf')
                c_env = cv_eval(plac_sub, ENV_FEATS, pg, model='rf')
                c_beh = cv_eval(plac_sub, BEH_FEATS, pg, model='rf')
            except Exception as e:
                print("   pair (%s,%s) skipped: %s" % (da, db, e))
                continue
            full_aucs.append(float(c_full.mean())); env_aucs.append(float(c_env.mean())); beh_aucs.append(float(c_beh.mean()))
            used_pairs.append([str(da), str(db), int(n_each)])
            print("   pair (%s,%s) n=%d full %.3f env %.3f beh %.3f" % (da, db, n_each, c_full.mean(), c_env.mean(), c_beh.mean()))
        results['G_day_placebo'] = {
            'n_pairs': len(used_pairs), 'pairs': used_pairs,
            'full_auc_mean_std': [float(np.mean(full_aucs)), float(np.std(full_aucs))],
            'env_auc_mean_std': [float(np.mean(env_aucs)), float(np.std(env_aucs))],
            'beh_auc_mean_std': [float(np.mean(beh_aucs)), float(np.std(beh_aucs))],
            'full_aucs': full_aucs, 'env_aucs': env_aucs, 'beh_aucs': beh_aucs}
        print("   G summary: full %.3f+-%.3f env %.3f+-%.3f beh %.3f+-%.3f (n=%d pairs)" % (
            np.mean(full_aucs), np.std(full_aucs), np.mean(env_aucs), np.std(env_aucs),
            np.mean(beh_aucs), np.std(beh_aucs), len(used_pairs)))
        _save(results)

    print("[load] FIRST-2015 corpus ...")
    first2015 = prep(load_first2015(WORK))
    results['pool_meta']['n_first2015_flows'] = int(len(first2015))
    results['pool_meta']['n_first2015_days'] = int(first2015['capture_day'].nunique())
    _save(results)

    if 'H_honey_vs_first2015' not in results:
        print("[H-a] honeypot pool vs FIRST-2015 ...")
        hdf = pd.concat([honey_pool.assign(label=1), first2015.assign(label=0)], ignore_index=True)
        sub = subsample(hdf, NPC, stratify_col='label')
        groups = sub['group'].values
        h_full = cv_eval(sub, ALL_FEATS, groups, model='rf')
        h_env = cv_eval(sub, ENV_FEATS, groups, model='rf')
        h_beh = cv_eval(sub, BEH_FEATS, groups, model='rf')
        results['H_honey_vs_first2015'] = {
            'n_used': int(len(sub)),
            'full_auc': [float(h_full.mean()), float(h_full.std())],
            'env_auc': [float(h_env.mean()), float(h_env.std())],
            'beh_auc': [float(h_beh.mean()), float(h_beh.std())]}
        print("   full %.4f env %.4f beh %.4f" % (h_full.mean(), h_env.mean(), h_beh.mean()))
        _save(results)

    if 'H_first2015_vs_cicids' not in results:
        print("[H-b] production-vs-production placebo: CIC-IDS2017 Tuesday vs FIRST-2015 ...")
        pdf = pd.concat([real_tue.assign(label=0), first2015.assign(label=1)], ignore_index=True)
        n_each = min(NPC, pdf[pdf.label == 0].shape[0], pdf[pdf.label == 1].shape[0])
        sub = subsample(pdf, n_each, stratify_col='label')
        groups = sub['group'].values
        p_full = cv_eval(sub, ALL_FEATS, groups, model='rf')
        p_env = cv_eval(sub, ENV_FEATS, groups, model='rf')
        p_beh = cv_eval(sub, BEH_FEATS, groups, model='rf')
        results['H_first2015_vs_cicids'] = {
            'n_used': int(len(sub)),
            'full_auc': [float(p_full.mean()), float(p_full.std())],
            'env_auc': [float(p_env.mean()), float(p_env.std())],
            'beh_auc': [float(p_beh.mean()), float(p_beh.std())]}
        print("   full %.4f env %.4f beh %.4f" % (p_full.mean(), p_env.mean(), p_beh.mean()))
        _save(results)

    make_figures(results)
    print("[done] experiments_v2 complete")
    return results

def make_figures(results):
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
    g = results['G_day_placebo']
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(g['env_aucs'], bins=15, alpha=0.7, color='#C44E52', label='Env/stack channel')
    ax.hist(g['full_aucs'], bins=15, alpha=0.5, color='#4C72B0', label='Full model')
    ax.axvline(0.5, ls='--', c='gray')
    ax.set_xlabel('Placebo AUC (honeypot day A vs honeypot day B)'); ax.set_ylabel('count of day-pairs')
    ax.legend(fontsize=8); ax.set_title(f"Day-grouped placebo distribution (n={g['n_pairs']} day pairs)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_day_placebo.png')); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    labels = ['Honeypot vs\nCIC-IDS2017\n(v1, n=2 days)', 'Honeypot pool vs\nCIC-IDS2017\n(F, all days)',
              'Honeypot pool vs\nFIRST-2015\n(H-a)', 'FIRST-2015 vs\nCIC-IDS2017\n(H-b, prod-vs-prod)']
    env_vals = [0.99999, results['F_full_pool']['env_auc'][0], results['H_honey_vs_first2015']['env_auc'][0],
                results['H_first2015_vs_cicids']['env_auc'][0]]
    full_vals = [1.000, results['F_full_pool']['full_auc'][0], results['H_honey_vs_first2015']['full_auc'][0],
                 results['H_first2015_vs_cicids']['full_auc'][0]]
    x = np.arange(4); w = 0.35
    ax.bar(x - w/2, env_vals, w, label='Env/stack-only', color='#C44E52')
    ax.bar(x + w/2, full_vals, w, label='Full model', color='#4C72B0')
    ax.axhline(0.5, ls='--', c='gray', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.4, 1.05); ax.set_ylabel('ROC AUC (host-grouped CV)')
    ax.legend(fontsize=8); ax.set_title('The confound generalizes across independent corpus pairs')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_generalization.png')); plt.close(fig)

if __name__ == '__main__':
    r = main()
    print(json.dumps(r, indent=2)[:2000])
