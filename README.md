# Artifact — "Shortcut or Signature?" (IEEE TPS-ISA 2026)


## What's here

```
src/           extraction + analysis Python (12 files)
results/       cached experiment outputs (5 JSON files, all numbers reported in the paper)
README.md      this file
```

**What's not here, and why:**
- Raw pcap captures — sizes in the tens of GB; every raw source used is
  publicly available (see *Data sources* below).
- Derived flow CSVs — 523 MB total; regenerable from the raw pcaps with
  the extractors in `src/`. On request we can share the exact extracted
  CSVs directly.

## Data sources (all public)

| Source                                | Used for                                | Where to get it                                          |
|---|---|---|
| **CIC-IDS2017**                       | production and same-testbed corpora     | https://www.unb.ca/cic/datasets/ids-2017.html            |
| **CIC-Honeynet Project (2019)**       | honeypot corpus (30 capture days)       | https://www.unb.ca/cic/research/honeynet.html            |
| **FIRST-2015 Hands-on Net. Forensics**| second production-like corpus           | https://www.first.org/conference/2015/program\#phands-on-network-forensics |

## Reproduce (from raw pcaps)

1. Fetch the three data sources above; place under any local directory,
   then update these two paths:
   - `src/extract_all_days.py` — `PCAP_DIR = "/path/to/CICHoneynet_pcap"`
   - `src/extract_first2015.py` — `ROOT = "/path/to/FIRST-2015_Hands-on_Network_Forensics_PCAP"`
2. Run the extractors:
   ```
   python src/extract_flows.py         # honeypot vs CIC-IDS2017 (Sections III–VII)
   python src/extract_all_days.py      # honeypot 30-day pool (Section IX Experiment F)
   python src/extract_first2015.py     # FIRST-2015 flows (Section IX Experiment H)
   ```
3. Run the analyses (each writes a `results/results*.json`):
   ```
   python src/experiments.py           # main confound decomposition (results.json)
   python src/experiments_v2.py        # 30-day pool + 50-pair placebo (results_v2.json)
   python src/experiments_v3.py        # positive control K-a / K-b (results_v3.json)
   python src/experiments_v4.py        # cross-corpus transfer test (results_v4.json)
   python src/experiments_xgb_shap.py  # XGBoost + SHAP replication (results_xgb_shap.json)
   ```

Total wall-clock runtime on a modest laptop is roughly one to two hours,
dominated by pcap extraction.

## Reproduce (from cached results only)

Every headline number in the paper is in `results/`. Numbers can be
re-derived from these JSON files without re-running any experiment:

| Paper claim                                   | Comes from                                   |
|---|---|
| Env-only AUC 0.99999, beh-only 0.9987, TTL 0.9906  | `results.json` → `B_decomp`              |
| Same-env placebo AUC 0.489 → chance             | `results.json` → `C_placebo`             |
| Permutation importance 94.6% env mass           | `results.json` → `D_importance`          |
| ECI: stack 1.00, full 0.72                      | `results.json` → `E_eci` and Table       |
| 30-day pool placebo distribution 0.500 ± 0.037  | `results_v2.json` → `G_50pair_placebo`   |
| H-a and H-b generalization (0.984 headline)     | `results_v2.json` → `H_generalize`       |
| K-a / K-b positive control AUCs                 | `results_v3.json`                         |
| Transfer FPRs 0.07% / 5.45% / env-only 0.54%    | `results_v4.json` → `L_transfer`          |
| XGBoost env AUC 0.99999, SHAP 77.7% env mass    | `results_xgb_shap.json`                   |

## Source file map

| File                          | Section in paper                                |
|---|---|
| `src/extract_flows.py`        | Section III — flow extraction pipeline           |
| `src/pcap_parse.py`           | packet → flow logic (shared by all extractors)  |
| `src/extract_chunked.py`      | chunked pcap reader for the 30-day pool         |
| `src/extract_all_days.py`     | Section IX Experiment F                         |
| `src/extract_first2015.py`    | Section IX Experiment H (FIRST-2015)            |
| `src/load_pool.py`            | pool assembly, shared by v2/v3/v4               |
| `src/ml.py`                   | model training, host-grouped CV, permutation importance |
| `src/experiments.py`          | Sections V–VII (confound, placebo, ECI, defense)|
| `src/experiments_v2.py`       | Section IX generalization + 50-pair placebo     |
| `src/experiments_v3.py`       | Section IX K-a / K-b positive control           |
| `src/experiments_v4.py`       | Section IX cross-corpus transfer                |
| `src/experiments_xgb_shap.py` | Section IX XGBoost + SHAP replication           |

## Dependencies

Python ≥ 3.10, plus: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`,
`scapy` (or `dpkt`) for pcap parsing. No exotic pinning; a fresh venv with
`pip install numpy pandas scikit-learn xgboost shap scapy` is enough.

## AI-assistance disclosure

Consistent with the paper's AI Disclosure section, an AI assistant
(Claude, Anthropic) helped implement the extraction and analysis code. 
All code in `src/` and all cached numbers in `results/` are authors' own.

