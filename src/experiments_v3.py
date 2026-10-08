#!/usr/bin/env python3
"""Experiment K: same-environment CIC-IDS2017 positive control.

Motivation: every placebo elsewhere in this study (C_placebo, G_day_placebo,
H_first2015_vs_cicids) collapses the env/stack channel to chance. That is the
correct result for testing a confound, but on its own it cannot distinguish
"no confound signal" from "no statistical power at all." UNB's own CIC-IDS2017
documentation states all five capture days (Mon-Fri, July 3-7 2017) share the
same testbed, victim/attacker hosts, and network configuration - only the
injected attack content differs by day. That gives a same-environment,
different-behavior positive control: Monday is pure benign; Friday adds
botnet/port-scan/DDoS; Tuesday (already in this project) adds brute-force
FTP/SSH. If the behavioral channel detects attack-day-vs-Monday while the
env/stack channel stays near chance (same hosts), that demonstrates the
pipeline has power to find real signal when it is genuinely present - which
is exactly what makes the chance results elsewhere credible nulls rather than
blind spots.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import ENV_FEATS, BEH_FEATS, ALL_FEATS, external_group, cv_eval, subsample, RNG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_pool import load_cicids_day, load_real_tue

WORK = os.environ.get('WORK', 'data')
OUT = os.environ.get('OUT', 'results')
os.makedirs(OUT, exist_ok=True)
NPC = int(os.environ.get('NPC', '20000'))

RESJSON = os.path.join(OUT, 'results_v3.json')

def _save(r):
    json.dump(r, open(RESJSON, 'w'), indent=2)

def prep(df):
    df = df[(df['tot_fwd_pkts'] + df['tot_bwd_pkts']) >= 2].reset_index(drop=True)
    df['group'] = df.apply(external_group, axis=1)
    return df

def run_pair(name_a, df_a, name_b, df_b, npc):
    pdf = pd.concat([df_a.assign(label=0), df_b.assign(label=1)], ignore_index=True)
    n_each = min(npc, pdf[pdf.label == 0].shape[0], pdf[pdf.label == 1].shape[0])
    sub = subsample(pdf, n_each, stratify_col='label')
    groups = sub['group'].values
    p_full = cv_eval(sub, ALL_FEATS, groups, model='rf')
    p_env = cv_eval(sub, ENV_FEATS, groups, model='rf')
    p_beh = cv_eval(sub, BEH_FEATS, groups, model='rf')
    p_ttl = cv_eval(sub, ['fwd_ttl_mean', 'bwd_ttl_mean', 'fwd_ttl_min', 'bwd_ttl_min'], groups, model='rf')
    res = {
        'pair': f'{name_a}_vs_{name_b}',
        'same_testbed': True,
        'n_used': int(len(sub)),
        'full_auc': [float(p_full.mean()), float(p_full.std())],
        'env_auc': [float(p_env.mean()), float(p_env.std())],
        'beh_auc': [float(p_beh.mean()), float(p_beh.std())],
        'ttl_auc': [float(p_ttl.mean()), float(p_ttl.std())]}
    print("   %s vs %s: full %.4f env %.4f beh %.4f ttl %.4f" % (
        name_a, name_b, p_full.mean(), p_env.mean(), p_beh.mean(), p_ttl.mean()))
    return res

def main():
    results = json.load(open(RESJSON)) if os.path.exists(RESJSON) else {}
    print("[load] CIC-IDS2017 Monday, Friday, Tuesday ...")
    mon = prep(load_cicids_day(WORK, 'cicids_mon', 'cicids_mon'))
    fri = prep(load_cicids_day(WORK, 'cicids_fri', 'cicids_fri'))
    tue = prep(load_real_tue(WORK))
    print(f"monday: {len(mon)} flows | friday: {len(fri)} flows | tuesday: {len(tue)} flows")

    results['pool_meta'] = results.get('pool_meta', {})
    results['pool_meta'].update({
        'n_monday_flows': int(len(mon)), 'n_friday_flows': int(len(fri)), 'n_tuesday_flows': int(len(tue)),
        'note': 'Monday/Friday/Tuesday share the same CIC-IDS2017 testbed, hosts, and network '
                'configuration per UNB documentation (captured July 3-7 2017); only injected '
                'attack content differs by day. Monday=pure benign, Friday=botnet/port-scan/DDoS, '
                'Tuesday=brute-force FTP/SSH.'})
    _save(results)

    if 'K_a_monday_vs_friday' not in results:
        print("[K-a] same-environment positive control: Monday (benign) vs Friday (botnet/portscan/DDoS) ...")
        results['K_a_monday_vs_friday'] = run_pair('monday', mon, 'friday', fri, NPC)
        _save(results)

    if 'K_b_monday_vs_tuesday' not in results:
        print("[K-b] same-environment positive control: Monday (benign) vs Tuesday (brute-force FTP/SSH) ...")
        results['K_b_monday_vs_tuesday'] = run_pair('monday', mon, 'tuesday', tue, NPC)
        _save(results)

    make_figure(results)
    print("[done] experiments_v3 complete")
    return results

def make_figure(results):
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
    ka, kb = results['K_a_monday_vs_friday'], results['K_b_monday_vs_tuesday']
    labels = ['Monday vs Friday\n(botnet/portscan/DDoS)', 'Monday vs Tuesday\n(brute-force FTP/SSH)']
    env_vals = [ka['env_auc'][0], kb['env_auc'][0]]
    beh_vals = [ka['beh_auc'][0], kb['beh_auc'][0]]
    full_vals = [ka['full_auc'][0], kb['full_auc'][0]]
    x = np.arange(2); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x - w, env_vals, w, label='Env/stack-only', color='#C44E52')
    ax.bar(x, beh_vals, w, label='Behavioral-only', color='#55A868')
    ax.bar(x + w, full_vals, w, label='Full model', color='#4C72B0')
    ax.axhline(0.5, ls='--', c='gray', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 1.05); ax.set_ylabel('ROC AUC (host-grouped CV)')
    for i in range(2):
        for off, v in [(-w, env_vals[i]), (0, beh_vals[i]), (w, full_vals[i])]:
            ax.text(i + off, v + 0.02, f"{v:.2f}", ha='center', fontsize=7)
    ax.legend(fontsize=8)
    ax.set_title('Same-environment positive control: same testbed, different attack day')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_positive_control.png')); plt.close(fig)

if __name__ == '__main__':
    r = main()
    print(json.dumps(r, indent=2)[:2000])
