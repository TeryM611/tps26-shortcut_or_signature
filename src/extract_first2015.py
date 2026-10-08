#!/usr/bin/env python3
"""Extract all FIRST-2015 Hands-on Network Forensics pcap files (snort.log.*)
into data/flows_first_<date>_<epoch>.csv, label=0 (independent production-like
corpus, distinct network/time/purpose from CIC-IDS2017)."""
import os, subprocess, sys, json, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT = "/path/to/FIRST-2015_Hands-on_Network_Forensics_PCAP"
DATA = os.path.join(HERE, 'data')
LOGPATH = os.path.join(DATA, 'extract_first2015.log')

def build_jobs():
    jobs = []
    for path in sorted(glob.glob(os.path.join(ROOT, '*', 'snort.log.*'))):
        date = os.path.basename(os.path.dirname(path))
        epoch = os.path.basename(path).split('.')[-1]
        tag = f"first_{date}_{epoch}"
        jobs.append((path, tag))
    return jobs

def extract_one(pcap, tag, budget=90, max_chunks=800):
    log_lines = []
    for i in range(max_chunks):
        p = subprocess.run([sys.executable, os.path.join(HERE, 'src', 'extract_chunked.py'),
                             pcap, '0', tag, DATA, str(budget)],
                            capture_output=True, text=True)
        try:
            obj = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            log_lines.append(f"{tag}: PARSE_FAIL rc={p.returncode} stderr={p.stderr[:500]}")
            break
        if obj.get('done'):
            log_lines.append(f"{tag}: DONE npkt={obj.get('total_pkts')} ntcp={obj.get('total_tcp')} "
                              f"nflows={obj.get('nflows')} resync_events={obj.get('resync_events')} "
                              f"resync_bytes={obj.get('resync_bytes_skipped')} "
                              f"unresolvable={obj.get('unresolvable_corruption')}")
            break
    else:
        log_lines.append(f"{tag}: GIVEUP after {max_chunks} chunks")
    return tag, log_lines

def main():
    jobs = build_jobs()
    print(f"[extract_first2015] {len(jobs)} pcap files to process", flush=True)
    with open(LOGPATH, 'a') as logf:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(extract_one, pcap, tag): tag for pcap, tag in jobs}
            for fut in as_completed(futs):
                tag, lines = fut.result()
                for l in lines:
                    print(l, flush=True)
                    logf.write(l + '\n'); logf.flush()
    print("ALL_FIRST2015_DONE", flush=True)
    with open(LOGPATH, 'a') as logf:
        logf.write("ALL_FIRST2015_DONE\n")

if __name__ == '__main__':
    main()
