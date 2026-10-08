#!/usr/bin/env python3
"""Experiment L: cross-corpus transfer -- an operational falsification.

The AUC decomposition (Sec. Confound) shows honeypot-vs-production corpora are
separable on environment features. A skeptic can still argue that this
separability *is* the honeypot signal. We settle it operationally, in the idiom
of distribution-shift/spurious-correlation testing: if a "honeypot detector"
has learned honeypot-ness, it must generalize -- pointed at a *different,
independent* production corpus it has never seen, it should label that corpus
production, not honeypot. If instead it learned corpus identity ("looks like the
one production capture I trained on" vs. "looks like the honeypot capture"), it
will systematically flag an unseen production corpus as honeypot.

We train honeypot(1) vs production-A(0), verify low in-domain error under
host-grouped evaluation, then measure the false-positive rate on production-B --
an independent, genuinely non-honeypot corpus whose true label is 0. A high
transfer FPR is direct operational proof of a corpus-identity classifier.
Symmetric A<->B and a stack-only (ENV) variant included.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml
from experiments import ENV_FEATS, BEH_FEATS, ALL_FEATS, external_group, subsample, RNG
from load_pool import load_honeypot_pool, load_first2015, load_real_tue

WORK = os.environ.get('WORK', 'data')
OUT = os.environ.get('OUT', 'results')
os.makedirs(OUT, exist_ok=True)
NPC = int(os.environ.get('NPC', '20000'))
RESJSON = os.path.join(OUT, 'results_v4.json')


def _save(r):
    json.dump(r, open(RESJSON, 'w'), indent=2)


def prep(df):
    df = df[(df['tot_fwd_pkts'] + df['tot_bwd_pkts']) >= 2].reset_index(drop=True)
    df['group'] = df.apply(external_group, axis=1)
    return df


def _mat(df, feats):
    return df[feats].replace([np.inf, -np.inf], np.nan).fillna(-1).astype(float).values


def grouped_split(df, frac_test=0.3, seed=RNG):
    """Split by external /24 group so no host straddles train/test."""
    g = df['group'].values
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_te = max(1, int(len(uniq) * frac_test))
    te_groups = set(uniq[:n_te].tolist())
    te = np.array([x in te_groups for x in g])
    return ~te, te


def transfer(name_a, prod_a, name_b, prod_b, honey, feats, feat_tag, seed=RNG):
    """Train honeypot(1) vs prod_a(0); report in-domain metrics (grouped hold-out)
    and the false-positive rate transferred onto independent production prod_b."""
    # balanced training corpus: honeypot vs production-A
    n_each = min(NPC, len(honey), len(prod_a))
    h = honey.sample(n=n_each, random_state=seed) if len(honey) > n_each else honey.copy()
    pa = prod_a.sample(n=n_each, random_state=seed) if len(prod_a) > n_each else prod_a.copy()
    train_all = pd.concat([h.assign(label=1), pa.assign(label=0)], ignore_index=True)
    tr, te = grouped_split(train_all, 0.3, seed)
    Xtr, ytr = _mat(train_all[tr], feats), train_all[tr]['label'].values.astype(int)
    Xte, yte = _mat(train_all[te], feats), train_all[te]['label'].values.astype(int)
    m = ml.HistRandomForest(n_trees=80, max_depth=12, min_leaf=20, seed=seed).fit(Xtr, ytr)
    # in-domain performance (same two corpora, held-out hosts)
    s_te = m.predict_proba(Xte)
    idom = ml.binary_metrics(yte, s_te, thr=0.5)
    idom_auc = ml.auc(yte, s_te)
    # in-domain false positive rate: production-A hosts labelled honeypot
    pa_mask = (yte == 0)
    idom_prod_fpr = float(((s_te[pa_mask] >= 0.5)).mean()) if pa_mask.sum() else float('nan')
    # TRANSFER: predict on independent production-B (true label 0)
    Xb = _mat(prod_b, feats)
    s_b = m.predict_proba(Xb)
    transfer_fpr = float((s_b >= 0.5).mean())            # fraction of real prod-B called honeypot
    transfer_meanp = float(s_b.mean())                    # mean P(honeypot) on real prod-B
    # honeypot recall on held-out honeypot (model still flags honeypots?)
    h_mask = (yte == 1)
    honey_recall = float((s_te[h_mask] >= 0.5).mean()) if h_mask.sum() else float('nan')
    res = {
        'train': f'honeypot_vs_{name_a}', 'transfer_to': name_b, 'features': feat_tag,
        'n_train_each': int(n_each),
        'in_domain_auc': idom_auc,
        'in_domain_accuracy': idom['accuracy'],
        'in_domain_honeypot_recall': honey_recall,
        'in_domain_production_fpr': idom_prod_fpr,
        'transfer_production_fpr': transfer_fpr,
        'transfer_mean_p_honeypot': transfer_meanp,
        'n_transfer_prod_flows': int(len(prod_b))}
    print(f"   [{feat_tag}] train honey-vs-{name_a}: in-domain AUC {idom_auc:.4f}, "
          f"honeypot recall {honey_recall:.3f}, in-domain prod-FPR {idom_prod_fpr:.3f}")
    print(f"            -> TRANSFER onto {name_b} (real production): "
          f"{transfer_fpr*100:.1f}% flagged HONEYPOT (mean P={transfer_meanp:.3f})")
    return res


def main():
    results = json.load(open(RESJSON)) if os.path.exists(RESJSON) else {}
    print("[load] corpora ...")
    honey, _ = load_honeypot_pool(WORK); honey = prep(honey)
    cicids = prep(load_real_tue(WORK))          # CIC-IDS2017 Tuesday (production A)
    first = prep(load_first2015(WORK))          # FIRST-2015 (production B)
    print(f"honeypot {len(honey)} | CIC-IDS2017 {len(cicids)} | FIRST-2015 {len(first)}")

    results['meta'] = {
        'design': 'Train honeypot(1) vs production-A(0); measure false-positive rate '
                  'when the frozen model is applied to an independent production-B corpus '
                  '(true label 0). High transfer FPR = corpus-identity classifier, not '
                  'honeypot detector.',
        'production_A_first_pass': 'CIC-IDS2017 Tuesday', 'production_B': 'FIRST-2015'}

    runs = []
    print("[L1] train honeypot vs CIC-IDS2017 -> transfer onto FIRST-2015")
    runs.append(transfer('cicids', cicids, 'first2015', first, honey, ALL_FEATS, 'all'))
    runs.append(transfer('cicids', cicids, 'first2015', first, honey, ENV_FEATS, 'env'))
    print("[L2] train honeypot vs FIRST-2015 -> transfer onto CIC-IDS2017")
    runs.append(transfer('first2015', first, 'cicids', cicids, honey, ALL_FEATS, 'all'))
    runs.append(transfer('first2015', first, 'cicids', cicids, honey, ENV_FEATS, 'env'))
    results['L_transfer'] = runs

    # ---- stack-profile homogeneity: why the honeypot cluster is distinctive ----
    def homog(df):
        key = (df['fwd_ttl_mean'].round().astype(int).astype(str) + '_'
               + df['init_fwd_win'].astype(str) + '_' + df['syn_mss_fwd'].astype(str))
        vc = key.value_counts(normalize=True)
        ent = float(-(vc * np.log2(vc)).sum())
        return {'n_flows': int(len(df)), 'n_distinct_stack_profiles': int(key.nunique()),
                'top1_profile_share': float(vc.iloc[0]), 'stack_entropy_bits': ent}
    results['M_homogeneity'] = {
        'honeypot': homog(honey), 'cicids_tue': homog(cicids), 'first2015': homog(first),
        'note': 'stack profile = (rounded fwd TTL, initial fwd window, SYN MSS). Lower '
                'entropy / higher top-1 share = a more infrastructure-uniform corpus. The '
                'honeypot corpus is a near-monolithic single-stack cluster, which is why a '
                'honeypot-trained detector rejects unseen production corpora without having '
                'learned any portable notion of honeypot-ness.'}
    _save(results)
    make_figure(results)
    print("[homogeneity] honey top-1 %.3f (%.2f bits) | cicids %.3f | first2015 %.3f"
          % (results['M_homogeneity']['honeypot']['top1_profile_share'],
             results['M_homogeneity']['honeypot']['stack_entropy_bits'],
             results['M_homogeneity']['cicids_tue']['top1_profile_share'],
             results['M_homogeneity']['first2015']['top1_profile_share']))
    print("[done] experiments_v4 complete")
    return results


def make_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = results['L_transfer']
    # show, per run, in-domain production FPR (near 0) vs transfer production FPR (high)
    labels = [f"train: H vs {r['train'].split('_vs_')[1]}\n"
              f"test on {r['transfer_to']}\n({r['features']} feats)" for r in runs]
    idom = [r['in_domain_production_fpr'] for r in runs]
    trans = [r['transfer_production_fpr'] for r in runs]
    x = np.arange(len(runs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ax.bar(x - w/2, idom, w, label='In-domain production FPR\n(same corpus, held-out hosts)', color='#55A868')
    ax.bar(x + w/2, trans, w, label='Transfer production FPR\n(independent production corpus)', color='#C44E52')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Fraction of REAL production\nflagged as honeypot')
    for i, v in enumerate(trans):
        ax.text(i + w/2, v + 0.02, f"{v*100:.0f}%", ha='center', fontsize=8)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title('A honeypot detector calls an unseen production network "honeypot"')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_transfer.png')); plt.close(fig)


if __name__ == '__main__':
    r = main()
    print(json.dumps(r['L_transfer'], indent=2)[:1500])
