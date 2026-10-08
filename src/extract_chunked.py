#!/usr/bin/env python3
"""
Resumable, time-boxed flow extractor (works within short call budgets).
Processes a pcap for up to TIME_BUDGET seconds per invocation, checkpoints the
byte offset, and appends flows + sampled packets. Re-run until state['done'].

State file: <out>/state_<tag>.json
Flows appended: <out>/flows_<tag>.csv
Packet sample (systematic 1-in-K): <out>/pkts_<tag>.csv

Note: active flows are flushed at the end of each chunk, so a small fraction of
flows crossing a chunk boundary are split. This is immaterial for aggregate
statistical modelling and keeps memory + runtime bounded.

Usage: python extract_chunked.py <pcap> <label> <tag> <out_dir> [time_budget_s]
"""
import sys, os, csv, json, struct, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcap_parse import parse_ip_tcp, find_resync_offset, MAX_SNAPLEN
import extract_flows as ef

PKT_SAMPLE_K = 50   # keep 1 packet in K

def read_header(path):
    with open(path,'rb') as f:
        magic=f.read(4)
    if magic in (b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4', b'\x4d\x3c\xb2\xa1', b'\xa1\xb2\x3c\x4d'):
        with open(path,'rb') as f:
            gh=f.read(24)
        endian='<' if magic in (b'\xd4\xc3\xb2\xa1',b'\x4d\x3c\xb2\xa1') else '>'
        nano = magic in (b'\x4d\x3c\xb2\xa1',b'\xa1\xb2\x3c\x4d')
        linktype=struct.unpack(endian+'I',gh[20:24])[0]
        return {'fmt':'classic','endian':endian,'nano':nano,'linktype':linktype,'data_start':24}
    elif magic==b'\x0a\x0d\x0d\x0a':
        return {'fmt':'ng'}
    raise ValueError('unknown magic')

def iter_classic_off(path, hdr, start_off, first_ts=None, resync_log=None):
    endian=hdr['endian']; nano=hdr['nano']; lt=hdr['linktype']
    rec=struct.Struct(endian+'IIII')
    filesize=os.path.getsize(path)
    with open(path,'rb') as f:
        f.seek(start_off if start_off>0 else hdr['data_start'])
        read=f.read
        while True:
            pos=f.tell()
            h=read(16)
            if len(h)<16: break
            ts_s,ts_frac,incl,orig=rec.unpack(h)
            valid = (0 < incl <= MAX_SNAPLEN) and (orig >= incl)
            if valid and first_ts is not None:
                ts_check = ts_s+(ts_frac/1e9 if nano else ts_frac/1e6)
                valid = (first_ts - 5*365*86400) <= ts_check <= (first_ts + 5*365*86400)
            if valid:
                data = read(incl)
                if len(data) == incl:
                    ts=ts_s+(ts_frac/1e9 if nano else ts_frac/1e6)
                    if first_ts is None: first_ts = ts
                    yield ts,data,orig,incl,lt,f.tell()
                    continue
                # header looked plausible but payload was cut short: genuine
                # end-of-capture truncation, not corruption. Stop cleanly.
                break
            # header itself is implausible -> corruption, not EOF (unless there
            # isn't even room left for a 16-byte header).
            if pos + 16 > filesize:
                break
            new_off = find_resync_offset(path, endian, nano, pos, first_ts, filesize)
            if new_off is None:
                if resync_log is not None:
                    resync_log.append({'unresolvable': True, 'stopped_at': pos})
                break
            if resync_log is not None:
                resync_log.append({'bad_offset': pos, 'resume_offset': new_off,
                                    'skipped_bytes': new_off - pos})
            f.seek(new_off)

def iter_ng_off(path, start_off, ifinfo):
    # ifinfo: dict with 'lt' and 'tsr' lists (persisted). Parse IDBs as encountered.
    with open(path,'rb') as f:
        f.seek(start_off if start_off>0 else 0)
        read=f.read
        lts=ifinfo['lt']; tsrs=ifinfo['tsr']
        while True:
            d0=read(4)
            if len(d0)<4: break
            btype=struct.unpack('<I',d0)[0]
            blb=read(4)
            if len(blb)<4: break
            blen=struct.unpack('<I',blb)[0]
            if blen<12: break
            body=read(blen-12)
            if len(body)<blen-12: break
            read(4)
            if btype==0x00000001:  # IDB
                linktype=struct.unpack('<H',body[0:2])[0]; tsr=1e-6
                ooff=8
                while ooff+4<=len(body):
                    oc,ol=struct.unpack('<HH',body[ooff:ooff+4])
                    ov=body[ooff+4:ooff+4+ol]
                    if oc==0: break
                    if oc==9 and len(ov)>=1:
                        r=ov[0]; tsr=2.0**(-(r&0x7f)) if r&0x80 else 10.0**(-r)
                    ooff+=4+ol+((4-(ol%4))%4)
                lts.append(linktype); tsrs.append(tsr)
            elif btype==0x00000006:  # EPB
                ifid,th,tl,caplen,origlen=struct.unpack('<IIIII',body[0:20])
                pkt=body[20:20+caplen]
                tsr=tsrs[ifid] if ifid<len(tsrs) else 1e-6
                lt=lts[ifid] if ifid<len(lts) else 1
                ts=((th<<32)|tl)*tsr
                yield ts,pkt,origlen,caplen,lt,f.tell()
            elif btype==0x00000003:  # SPB
                origlen=struct.unpack('<I',body[0:4])[0]; pkt=body[4:]
                lt=lts[0] if lts else 1
                yield 0.0,pkt,origlen,len(pkt),lt,f.tell()

def run(pcap,label,tag,out_dir,budget):
    os.makedirs(out_dir,exist_ok=True)
    spath=os.path.join(out_dir,f'state_{tag}.json')
    fpath=os.path.join(out_dir,f'flows_{tag}.csv')
    ppath=os.path.join(out_dir,f'pkts_{tag}.csv')
    if os.path.exists(spath):
        state=json.load(open(spath))
    else:
        hdr=read_header(pcap)
        state={'offset':0,'npkt':0,'ntcp':0,'nflows':0,'nsamp':0,'done':False,'hdr':hdr,
               'ifinfo':{'lt':[],'tsr':[]},'first_ts':None,'resync':[]}
    if state.get('done'):
        print(json.dumps({'tag':tag,'done':True,'npkt':state['npkt'],'ntcp':state['ntcp'],'nflows':state['nflows'],
                           'resync_events':len(state.get('resync',[])),
                           'resync_bytes_skipped':sum(r['skipped_bytes'] for r in state.get('resync',[]))}))
        return
    hdr=state['hdr']
    state.setdefault('first_ts', None); state.setdefault('resync', [])
    new_file = not os.path.exists(fpath)
    fh=open(fpath,'a',newline=''); fw=csv.writer(fh)
    if new_file: fw.writerow(ef.FLOW_COLS)
    ph=open(ppath,'a',newline=''); pw=csv.writer(ph)
    if not os.path.exists(ppath) or os.path.getsize(ppath)==0: pw.writerow(ef.PKT_COLS)

    flows={}
    t0=time.time(); processed=0; ntcp0=state['ntcp']; npkt0=state['npkt']
    resync_log=[]
    if hdr['fmt']=='classic':
        it=iter_classic_off(pcap,hdr,state['offset'],first_ts=state['first_ts'],resync_log=resync_log)
    else:
        it=iter_ng_off(pcap,state['offset'],state['ifinfo'])
    last_off=state['offset']
    for ts,data,orig,incl,lt,off in it:
        if state['first_ts'] is None: state['first_ts']=ts
        state['npkt']+=1; processed+=1
        r=parse_ip_tcp(data,0,lt)
        last_off=off
        if r and r[0]=='tcp':
            state['ntcp']+=1
            ef_update(flows,r,ts,orig,label,tag,pw,state)
        if (processed & 0x3FFF)==0:  # every ~16k pkts check time
            if time.time()-t0>budget:
                break
    # flush active flows for this chunk
    for k,a in flows.items():
        ef.finalize(k,a,label,tag,fw)
        state['nflows']+=1
    fh.close(); ph.close()
    # determine done: did we reach EOF? if loop ended by break vs exhaustion
    # detect EOF by comparing offset to filesize
    fsize=os.path.getsize(pcap)
    reached_eof = last_off>=fsize-8
    unresolvable = any(r.get('unresolvable') for r in resync_log)
    if processed==0 or reached_eof or unresolvable:
        state['done']=True
    state['offset']=last_off
    state['resync'] = state.get('resync', []) + [r for r in resync_log if not r.get('unresolvable')]
    state['resync_unresolvable'] = state.get('resync_unresolvable', False) or unresolvable
    if unresolvable:
        state['truncated_at_offset'] = [r['stopped_at'] for r in resync_log if r.get('unresolvable')][0]
    json.dump(state,open(spath,'w'))
    print(json.dumps({'tag':tag,'chunk_pkts':processed,'total_pkts':state['npkt'],
        'total_tcp':state['ntcp'],'nflows':state['nflows'],'offset':state['offset'],
        'fsize':fsize,'done':state['done'],'sec':round(time.time()-t0,1),
        'resync_events':len(state['resync']),
        'resync_bytes_skipped':sum(r['skipped_bytes'] for r in state['resync']),
        'unresolvable_corruption':unresolvable}))

def ef_update(flows,r,ts,orig,label,tag,pw,state):
    (_, s_ip,d_ip,s_pt,d_pt,ttl,dscp,ecn,ip_id,df,mf,ihl,tlen,win,flags,dataofs,mss,wscale,sackok,tsopt)=r
    if (s_ip,s_pt)<=(d_ip,d_pt): key=(s_ip,s_pt,d_ip,d_pt)
    else: key=(d_ip,d_pt,s_ip,s_pt)
    a=flows.get(key)
    if a is None:
        a=ef.Acc(); a.first_ts=ts; a.proto=6; a.fwd_key=(s_ip,s_pt,d_ip,d_pt); flows[key]=a
    a.last_ts=ts
    fwd=(a.fwd_key[0]==s_ip and a.fwd_key[1]==s_pt)
    L=orig
    a.a_sum+=L; a.a_sq+=L*L
    if L<a.a_min: a.a_min=L
    if L>a.a_max: a.a_max=L
    if a.last_all>=0:
        iat=ts-a.last_all
        if iat>=0:
            a.iat_n+=1; a.iat_sum+=iat; a.iat_sq+=iat*iat
            if iat<a.iat_min: a.iat_min=iat
            if iat>a.iat_max: a.iat_max=iat
    a.last_all=ts
    if flags&0x01: a.fin+=1
    if flags&0x02: a.syn+=1
    if flags&0x04: a.rst+=1
    if flags&0x08: a.psh+=1
    if flags&0x10: a.ack+=1
    if flags&0x20: a.urg+=1
    if flags&0x40: a.ece+=1
    if flags&0x80: a.cwr+=1
    if df: a.df_cnt+=1
    if dscp!=0: a.dscp_nonzero+=1
    if fwd:
        a.f_cnt+=1; a.f_byt+=L; a.f_sum+=L; a.f_sq+=L*L
        if L<a.f_min: a.f_min=L
        if L>a.f_max: a.f_max=L
        a.f_ttl_sum+=ttl; a.f_ttl_n+=1
        if ttl<a.f_ttl_min: a.f_ttl_min=ttl
        if ttl>a.f_ttl_max: a.f_ttl_max=ttl
        if a.dscp_fwd<0: a.dscp_fwd=dscp
        if a.last_f>=0:
            iat=ts-a.last_f
            if iat>=0:
                a.fiat_n+=1; a.fiat_sum+=iat; a.fiat_sq+=iat*iat
                if iat<a.fiat_min: a.fiat_min=iat
                if iat>a.fiat_max: a.fiat_max=iat
        a.last_f=ts
        if flags&0x02:
            if a.init_fwin<0: a.init_fwin=win
            if mss>=0: a.mss_f=mss
            if wscale>=0: a.ws_f=wscale
            if sackok>0: a.sackok=1
            if tsopt>0: a.tsopt=1
    else:
        a.b_cnt+=1; a.b_byt+=L; a.b_sum+=L; a.b_sq+=L*L
        if L<a.b_min: a.b_min=L
        if L>a.b_max: a.b_max=L
        a.b_ttl_sum+=ttl; a.b_ttl_n+=1
        if ttl<a.b_ttl_min: a.b_ttl_min=ttl
        if ttl>a.b_ttl_max: a.b_ttl_max=ttl
        if a.last_b>=0:
            iat=ts-a.last_b
            if iat>=0:
                a.biat_n+=1; a.biat_sum+=iat; a.biat_sq+=iat*iat
                if iat<a.biat_min: a.biat_min=iat
                if iat>a.biat_max: a.biat_max=iat
        a.last_b=ts
        if flags&0x02:
            if a.init_bwin<0: a.init_bwin=win
            if mss>=0: a.mss_b=mss
            if wscale>=0: a.ws_b=wscale
    # systematic packet sample
    if state['npkt'] % PKT_SAMPLE_K == 0:
        pw.writerow([ttl,dscp,ecn,win,flags,mss,wscale,sackok,tsopt,df,orig,6,(1 if fwd else 0),label,tag])
        state['nsamp']+=1

if __name__=='__main__':
    pcap,label,tag,out=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
    budget=float(sys.argv[5]) if len(sys.argv)>5 else 35.0
    run(pcap,label,tag,out,budget)
