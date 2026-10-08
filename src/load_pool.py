#!/usr/bin/env python3
"""Load and aggregate the extended multi-day honeypot pool and the FIRST-2015
independent production-like corpus, on top of the v1 single-day data."""
import os, re, glob, pandas as pd

def _day_from_tag(tag):
    m = re.match(r'honey(\d+)', tag)
    if m: return int(m.group(1))
    if tag == 'honey24': return 24
    if tag == 'honey31': return 31
    return None

def load_honeypot_pool(data_dir, exclude=('honey16_s2',)):
    """All honeypot day/segment CSVs (v1's honey24/honey31 plus the new days),
    tagged with a capture-day integer so same-day segments never split across
    CV/placebo groups. Excludes confirmed-duplicate segments."""
    files = sorted(glob.glob(os.path.join(data_dir, 'flows_honey*.csv')))
    frames = []
    days_seen = {}
    for f in files:
        tag = os.path.basename(f)[len('flows_'):-4]
        if tag in exclude:
            continue
        day = _day_from_tag(tag)
        if day is None:
            continue
        df = pd.read_csv(f)
        df['capture_day'] = day
        df['source_tag'] = tag
        frames.append(df)
        days_seen.setdefault(day, []).append(tag)
    pool = pd.concat(frames, ignore_index=True)
    pool['label'] = 1
    return pool, days_seen

def load_first2015(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, 'flows_first_*.csv')))
    frames = []
    for f in files:
        tag = os.path.basename(f)[len('flows_'):-4]  # first_<date>_<epoch>
        m = re.match(r'first_(\d{4}-\d{2}-\d{2})_', tag)
        date = m.group(1) if m else tag
        df = pd.read_csv(f)
        df['capture_day'] = date
        df['source_tag'] = tag
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool['label'] = 0
    return pool

def load_real_tue(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'flows_real_tue.csv'))
    df['capture_day'] = 'cicids_tue'
    df['source_tag'] = 'real_tue'
    df['label'] = 0
    return df

def load_cicids_day(data_dir, tag, day_name):
    """Load a single-tag CIC-IDS2017 day extracted via extract_chunked.py
    (e.g. tag='cicids_mon' -> data/flows_cicids_mon.csv). Same testbed/hosts
    as real_tue (Tuesday) per UNB's CIC-IDS2017 documentation - only the
    injected attack content differs by day."""
    df = pd.read_csv(os.path.join(data_dir, f'flows_{tag}.csv'))
    df['capture_day'] = day_name
    df['source_tag'] = tag
    return df
