#!/usr/bin/env python3
"""Extract all remaining CICHoneynet pcap days/segments (24July, 31July already
done) into data/flows_honey<DD>_s<N>.csv, in parallel, resuming each until done.
"""
import os, re, subprocess, sys, json
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) if False else \
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PCAP_DIR = "/path/to/CICHoneynet_pcap"
DATA = os.path.join(HERE, 'data')
LOGPATH = os.path.join(DATA, 'extract_all_days.log')
DONE_FILES = {"CICHoneynet_24July..pcap", "CICHoneynet_31July..pcap"}

def build_jobs():
    jobs = []
    for fn in sorted(os.listdir(PCAP_DIR)):
        if fn in DONE_FILES or not fn.lower().endswith('.pcap'):
            continue
        m = re.search(r'_(\d+)(?:-(\d+)|\((\d+)\))?\s*July', fn)
        if not m:
            continue
        day = int(m.group(1)); seg = m.group(2) or m.group(3) or '1'
        tag = f"honey{day:02d}_s{seg}"
        jobs.append((os.path.join(PCAP_DIR, fn), tag))
    return jobs

def extract_one(pcap, tag, budget=60, max_chunks=500):
    log_lines = []
    for i in range(max_chunks):
        p = subprocess.run([sys.executable, os.path.join(HERE, 'src', 'extract_chunked.py'),
                             pcap, '1', tag, DATA, str(budget)],
                            capture_output=True, text=True)
        line = (p.stdout or '').strip() + ((' STDERR:' + p.stderr.strip()) if p.stderr.strip() else '')
        log_lines.append(f"{tag}: {line}")
        try:
            obj = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            log_lines.append(f"{tag}: PARSE_FAIL rc={p.returncode}")
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
    print(f"[extract_all_days] {len(jobs)} pcap files to process", flush=True)
    with open(LOGPATH, 'a') as logf:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(extract_one, pcap, tag): tag for pcap, tag in jobs}
            for fut in as_completed(futs):
                tag, lines = fut.result()
                for l in lines:
                    print(l, flush=True)
                    logf.write(l + '\n'); logf.flush()
    print("ALL_EXTRACTION_DONE", flush=True)
    with open(LOGPATH, 'a') as logf:
        logf.write("ALL_EXTRACTION_DONE\n")

if __name__ == '__main__':
    main()
