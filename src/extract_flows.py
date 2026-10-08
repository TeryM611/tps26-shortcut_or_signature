#!/usr/bin/env python3
"""
Streaming bidirectional flow featurizer over a pcap/pcapng.
Bounded memory: idle flows are evicted (flow timeout) and written to CSV.
Also keeps a reservoir sample of per-packet records for packet-level analysis.

Usage:
  python extract_flows.py <pcap> <label 0|1> <dataset_tag> <out_dir>

Outputs (in out_dir):
  flows_<tag>.csv     one row per flow
  pkts_<tag>.csv      reservoir sample of packets
  meta_<tag>.json     processing stats
"""
import sys, os, csv, json, math, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcap_parse import iter_packets, parse_ip_tcp

FLOW_TIMEOUT = 120.0        # seconds idle -> evict
EVICT_EVERY = 200000        # packets between eviction sweeps
RESERVOIR = 1500000         # packet sample cap
random.seed(1337)

# accumulator slot indices
class Acc:
    __slots__ = ('first_ts','last_ts','fwd_key',
                 'f_cnt','b_cnt','f_byt','b_byt',
                 'a_sum','a_sq','a_min','a_max',
                 'f_sum','f_sq','f_min','f_max',
                 'b_sum','b_sq','b_min','b_max',
                 'last_all','last_f','last_b',
                 'iat_n','iat_sum','iat_sq','iat_min','iat_max',
                 'fiat_n','fiat_sum','fiat_sq','fiat_min','fiat_max',
                 'biat_n','biat_sum','biat_sq','biat_min','biat_max',
                 'fin','syn','rst','psh','ack','urg','ece','cwr',
                 'init_fwin','init_bwin',
                 'f_ttl_sum','f_ttl_n','f_ttl_min','f_ttl_max',
                 'b_ttl_sum','b_ttl_n','b_ttl_min','b_ttl_max',
                 'dscp_fwd','dscp_nonzero',
                 'mss_f','mss_b','ws_f','ws_b','sackok','tsopt','df_cnt','proto')
    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, 0)
        self.a_min=self.f_min=self.b_min=1e18
        self.iat_min=self.fiat_min=self.biat_min=1e18
        self.f_ttl_min=self.b_ttl_min=999
        self.init_fwin=-1; self.init_bwin=-1
        self.mss_f=-1; self.mss_b=-1; self.ws_f=-1; self.ws_b=-1
        self.sackok=0; self.tsopt=0
        self.last_all=-1.0; self.last_f=-1.0; self.last_b=-1.0
        self.dscp_fwd=-1

FLOW_COLS = ['src_ip','dst_ip','src_port','dst_port','proto','first_ts','flow_duration',
 'tot_fwd_pkts','tot_bwd_pkts','totlen_fwd_pkts','totlen_bwd_pkts','flow_byts_s','flow_pkts_s',
 'fwd_pkts_s','bwd_pkts_s','down_up_ratio',
 'pkt_len_min','pkt_len_max','pkt_len_mean','pkt_len_std','pkt_size_avg',
 'fwd_pkt_len_min','fwd_pkt_len_max','fwd_pkt_len_mean','fwd_pkt_len_std',
 'bwd_pkt_len_min','bwd_pkt_len_max','bwd_pkt_len_mean','bwd_pkt_len_std',
 'flow_iat_mean','flow_iat_std','flow_iat_min','flow_iat_max',
 'fwd_iat_mean','fwd_iat_std','fwd_iat_min','fwd_iat_max',
 'bwd_iat_mean','bwd_iat_std','bwd_iat_min','bwd_iat_max',
 'fin_cnt','syn_cnt','rst_cnt','psh_cnt','ack_cnt','urg_cnt','ece_cnt','cwr_cnt',
 # --- environment / host-stack fingerprint features ---
 'fwd_ttl_min','fwd_ttl_max','fwd_ttl_mean','bwd_ttl_min','bwd_ttl_max','bwd_ttl_mean',
 'ip_dscp','dscp_nonzero_rate','init_fwd_win','init_bwd_win',
 'syn_mss_fwd','syn_mss_bwd','syn_wscale_fwd','syn_wscale_bwd','sack_ok','ts_opt','df_rate',
 'label','dataset']

PKT_COLS = ['ttl','dscp','ecn','win','flags','mss','wscale','sackok','tsopt','df','total_len','proto','direction','label','dataset']

def _std(n, s, sq):
    if n < 2: return 0.0
    v = (sq - s*s/n)/(n)
    return math.sqrt(v) if v > 0 else 0.0

def finalize(key, a, label, tag, w):
    src_ip, sp, dst_ip, dp = key_to_endpoints(key, a)
    dur = max(a.last_ts - a.first_ts, 0.0)
    totp = a.f_cnt + a.b_cnt
    totb = a.f_byt + a.b_byt
    flow_byts_s = totb/dur if dur>0 else 0.0
    flow_pkts_s = totp/dur if dur>0 else 0.0
    fwd_pkts_s = a.f_cnt/dur if dur>0 else 0.0
    bwd_pkts_s = a.b_cnt/dur if dur>0 else 0.0
    down_up = (a.b_cnt/a.f_cnt) if a.f_cnt>0 else 0.0
    a_mean = a.a_sum/totp if totp>0 else 0.0
    f_mean = a.f_sum/a.f_cnt if a.f_cnt>0 else 0.0
    b_mean = a.b_sum/a.b_cnt if a.b_cnt>0 else 0.0
    row = [src_ip,dst_ip,sp,dp,a.proto,round(a.first_ts,6),round(dur,6),
      a.f_cnt,a.b_cnt,a.f_byt,a.b_byt,round(flow_byts_s,4),round(flow_pkts_s,4),
      round(fwd_pkts_s,4),round(bwd_pkts_s,4),round(down_up,4),
      (0 if a.a_min>1e17 else a.a_min),a.a_max,round(a_mean,3),round(_std(totp,a.a_sum,a.a_sq),3),round(a_mean,3),
      (0 if a.f_min>1e17 else a.f_min),a.f_max,round(f_mean,3),round(_std(a.f_cnt,a.f_sum,a.f_sq),3),
      (0 if a.b_min>1e17 else a.b_min),a.b_max,round(b_mean,3),round(_std(a.b_cnt,a.b_sum,a.b_sq),3),
      round(a.iat_sum/a.iat_n,6) if a.iat_n>0 else 0.0, round(_std(a.iat_n,a.iat_sum,a.iat_sq),6),
      (0 if a.iat_min>1e17 else round(a.iat_min,6)), round(a.iat_max,6),
      round(a.fiat_sum/a.fiat_n,6) if a.fiat_n>0 else 0.0, round(_std(a.fiat_n,a.fiat_sum,a.fiat_sq),6),
      (0 if a.fiat_min>1e17 else round(a.fiat_min,6)), round(a.fiat_max,6),
      round(a.biat_sum/a.biat_n,6) if a.biat_n>0 else 0.0, round(_std(a.biat_n,a.biat_sum,a.biat_sq),6),
      (0 if a.biat_min>1e17 else round(a.biat_min,6)), round(a.biat_max,6),
      a.fin,a.syn,a.rst,a.psh,a.ack,a.urg,a.ece,a.cwr,
      (0 if a.f_ttl_min>900 else a.f_ttl_min),a.f_ttl_max,
      round(a.f_ttl_sum/a.f_ttl_n,2) if a.f_ttl_n>0 else 0,
      (0 if a.b_ttl_min>900 else a.b_ttl_min),a.b_ttl_max,
      round(a.b_ttl_sum/a.b_ttl_n,2) if a.b_ttl_n>0 else 0,
      a.dscp_fwd if a.dscp_fwd>=0 else 0, round(a.dscp_nonzero/totp,4) if totp>0 else 0.0,
      a.init_fwin, a.init_bwin, a.mss_f, a.mss_b, a.ws_f, a.ws_b, a.sackok, a.tsopt,
      round(a.df_cnt/totp,4) if totp>0 else 0.0,
      label, tag]
    w.writerow(row)

def key_to_endpoints(key, a):
    (ipA, pA, ipB, pB) = key
    # a.fwd_key holds the initiator endpoints tuple
    if a.fwd_key:
        (s,sp,d,dp) = a.fwd_key
        return s, sp, d, dp
    return ipA, pA, ipB, pB

def run(pcap, label, tag, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    flows = {}
    fpath = os.path.join(out_dir, f'flows_{tag}.csv')
    ppath = os.path.join(out_dir, f'pkts_{tag}.csv')
    fh = open(fpath,'w',newline=''); fw = csv.writer(fh); fw.writerow(FLOW_COLS)
    reservoir = []; seen_pkt = 0
    npkt=0; ntcp=0; t0=time.time(); last_evict_pkt=0; last_ts=[0.0]
    MAXP = int(os.environ.get('MAXP','0'))

    def maybe_evict(now):
        cutoff = now - FLOW_TIMEOUT
        dead = [k for k,a in flows.items() if a.last_ts < cutoff]
        for k in dead:
            finalize(k, flows[k], label, tag, fw)
            del flows[k]

    for ts, data, orig, incl, lt in iter_packets(pcap):
        npkt += 1
        if MAXP and npkt > MAXP: break
        r = parse_ip_tcp(data, 0, lt)
        if not r: continue
        if r[0] != 'tcp':
            continue
        (_, s_ip, d_ip, s_pt, d_pt, ttl, dscp, ecn, ip_id, df, mf, ihl, tlen,
         win, flags, dataofs, mss, wscale, sackok, tsopt) = r
        ntcp += 1
        # canonical bidirectional key
        if (s_ip, s_pt) <= (d_ip, d_pt):
            key = (s_ip, s_pt, d_ip, d_pt)
        else:
            key = (d_ip, d_pt, s_ip, s_pt)
        a = flows.get(key)
        if a is None:
            a = Acc(); a.first_ts = ts; a.proto = 6
            a.fwd_key = (s_ip, s_pt, d_ip, d_pt)   # initiator direction
            flows[key] = a
        a.last_ts = ts
        # direction relative to initiator
        fwd = (a.fwd_key[0] == s_ip and a.fwd_key[1] == s_pt)
        L = orig
        # all pkt length stats
        a.a_sum += L; a.a_sq += L*L
        if L < a.a_min: a.a_min = L
        if L > a.a_max: a.a_max = L
        # IAT all
        if a.last_all >= 0:
            iat = ts - a.last_all
            if iat >= 0:
                a.iat_n += 1; a.iat_sum += iat; a.iat_sq += iat*iat
                if iat < a.iat_min: a.iat_min = iat
                if iat > a.iat_max: a.iat_max = iat
        a.last_all = ts
        # flags
        if flags & 0x01: a.fin += 1
        if flags & 0x02: a.syn += 1
        if flags & 0x04: a.rst += 1
        if flags & 0x08: a.psh += 1
        if flags & 0x10: a.ack += 1
        if flags & 0x20: a.urg += 1
        if flags & 0x40: a.ece += 1
        if flags & 0x80: a.cwr += 1
        if df: a.df_cnt += 1
        if dscp != 0: a.dscp_nonzero += 1
        if fwd:
            a.f_cnt += 1; a.f_byt += L
            a.f_sum += L; a.f_sq += L*L
            if L < a.f_min: a.f_min = L
            if L > a.f_max: a.f_max = L
            a.f_ttl_sum += ttl; a.f_ttl_n += 1
            if ttl < a.f_ttl_min: a.f_ttl_min = ttl
            if ttl > a.f_ttl_max: a.f_ttl_max = ttl
            if a.dscp_fwd < 0: a.dscp_fwd = dscp
            if a.last_f >= 0:
                iat = ts - a.last_f
                if iat >= 0:
                    a.fiat_n += 1; a.fiat_sum += iat; a.fiat_sq += iat*iat
                    if iat < a.fiat_min: a.fiat_min = iat
                    if iat > a.fiat_max: a.fiat_max = iat
            a.last_f = ts
            if (flags & 0x02):  # SYN from initiator
                if a.init_fwin < 0: a.init_fwin = win
                if mss>=0: a.mss_f = mss
                if wscale>=0: a.ws_f = wscale
                if sackok>0: a.sackok = 1
                if tsopt>0: a.tsopt = 1
        else:
            a.b_cnt += 1; a.b_byt += L
            a.b_sum += L; a.b_sq += L*L
            if L < a.b_min: a.b_min = L
            if L > a.b_max: a.b_max = L
            a.b_ttl_sum += ttl; a.b_ttl_n += 1
            if ttl < a.b_ttl_min: a.b_ttl_min = ttl
            if ttl > a.b_ttl_max: a.b_ttl_max = ttl
            if a.last_b >= 0:
                iat = ts - a.last_b
                if iat >= 0:
                    a.biat_n += 1; a.biat_sum += iat; a.biat_sq += iat*iat
                    if iat < a.biat_min: a.biat_min = iat
                    if iat > a.biat_max: a.biat_max = iat
            a.last_b = ts
            if (flags & 0x02):  # SYN-ACK from responder
                if a.init_bwin < 0: a.init_bwin = win
                if mss>=0: a.mss_b = mss
                if wscale>=0: a.ws_b = wscale
        # reservoir sample of packets
        seen_pkt += 1
        rec = [ttl,dscp,ecn,win,flags,mss,wscale,sackok,tsopt,df,orig,6,(1 if fwd else 0),label,tag]
        if len(reservoir) < RESERVOIR:
            reservoir.append(rec)
        else:
            j = random.randint(0, seen_pkt-1)
            if j < RESERVOIR:
                reservoir[j] = rec
        # periodic eviction
        if npkt - last_evict_pkt >= EVICT_EVERY:
            maybe_evict(ts)
            last_evict_pkt = npkt

    # flush remaining
    for k,a in flows.items():
        finalize(k, a, label, tag, fw)
    fh.close()
    # write reservoir
    with open(ppath,'w',newline='') as ph:
        pw = csv.writer(ph); pw.writerow(PKT_COLS)
        pw.writerows(reservoir)
    meta = {'pcap':os.path.basename(pcap),'label':label,'dataset':tag,
            'n_packets':npkt,'n_tcp':ntcp,'n_pkt_sampled':len(reservoir),
            'seconds':round(time.time()-t0,1)}
    with open(os.path.join(out_dir,f'meta_{tag}.json'),'w') as mh:
        json.dump(meta,mh,indent=2)
    print(json.dumps(meta))

if __name__ == '__main__':
    pcap, label, tag, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    run(pcap, label, tag, out_dir)
