#!/usr/bin/env python3
"""
Pure-Python streaming pcap / pcapng parser.
Extracts per-packet IP/TCP features to CSV with low, bounded memory.
No third-party deps (struct + stdlib only).

Fields per TCP/IP packet:
  ts, src_ip, dst_ip, proto, src_port, dst_port,
  ip_ttl, ip_dscp, ip_ecn, ip_id, ip_df, ip_mf, ip_hdr_len, ip_total_len,
  tcp_win, tcp_flags, tcp_dataofs, orig_len, cap_len,
  opt_mss, opt_wscale, opt_sackok, opt_ts   (TCP options; -1 = absent)
"""
import struct, socket, sys, os

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW1 = 101   # raw IP
LINKTYPE_RAW12 = 12
LINKTYPE_LINUX_SLL = 113
LINKTYPE_LINUX_SLL2 = 276
LINKTYPE_NULL = 0

def _ip4(b):
    return socket.inet_ntoa(b)

def parse_ip_tcp(data, l2off, linktype):
    """Given raw frame bytes, return a feature tuple or None (non TCP/IPv4)."""
    try:
        if linktype == LINKTYPE_ETHERNET:
            if len(data) < 14: return None
            eth_type = struct.unpack('!H', data[12:14])[0]
            off = 14
            # VLAN 802.1Q / QinQ
            while eth_type == 0x8100 or eth_type == 0x88a8:
                if len(data) < off+4: return None
                eth_type = struct.unpack('!H', data[off+2:off+4])[0]
                off += 4
            if eth_type != 0x0800:  # not IPv4
                return None
        elif linktype in (LINKTYPE_RAW1, LINKTYPE_RAW12):
            off = 0
        elif linktype == LINKTYPE_NULL:
            off = 4
        elif linktype == LINKTYPE_LINUX_SLL:
            if len(data) < 16: return None
            proto_l3 = struct.unpack('!H', data[14:16])[0]
            if proto_l3 != 0x0800: return None
            off = 16
        else:
            if len(data) < 14: return None
            eth_type = struct.unpack('!H', data[12:14])[0]
            if eth_type != 0x0800: return None
            off = 14

        if len(data) < off+20: return None
        vihl = data[off]
        version = vihl >> 4
        if version != 4: return None
        ihl = (vihl & 0x0f) * 4
        if ihl < 20: return None
        tos = data[off+1]
        dscp = tos >> 2
        ecn = tos & 0x3
        total_len = struct.unpack('!H', data[off+2:off+4])[0]
        ip_id = struct.unpack('!H', data[off+4:off+6])[0]
        flags_frag = struct.unpack('!H', data[off+6:off+8])[0]
        df = (flags_frag >> 14) & 1
        mf = (flags_frag >> 13) & 1
        ttl = data[off+8]
        proto = data[off+9]
        src_ip = _ip4(data[off+12:off+16])
        dst_ip = _ip4(data[off+16:off+20])
        if proto != 6:  # TCP only
            return ('nontcp', proto, src_ip, dst_ip, ttl, dscp, ecn, ip_id, df, mf, ihl, total_len)
        toff = off + ihl
        if len(data) < toff+20: return None
        src_port, dst_port = struct.unpack('!HH', data[toff:toff+4])
        dataofs = (data[toff+12] >> 4) * 4
        flags = data[toff+13]           # FIN,SYN,RST,PSH,ACK,URG,ECE,CWR
        win = struct.unpack('!H', data[toff+14:toff+16])[0]
        # TCP options
        opt_mss = opt_wscale = opt_sackok = opt_ts = -1
        ooff = toff + 20
        oend = toff + dataofs
        while ooff < oend and ooff < len(data):
            kind = data[ooff]
            if kind == 0: break        # EOL
            if kind == 1:              # NOP
                ooff += 1; continue
            if ooff+1 >= len(data): break
            olen = data[ooff+1]
            if olen < 2: break
            if kind == 2 and olen == 4 and ooff+4 <= len(data):
                opt_mss = struct.unpack('!H', data[ooff+2:ooff+4])[0]
            elif kind == 3 and olen == 3 and ooff+3 <= len(data):
                opt_wscale = data[ooff+2]
            elif kind == 4:
                opt_sackok = 1
            elif kind == 8:
                opt_ts = 1
            ooff += olen
        return ('tcp', src_ip, dst_ip, src_port, dst_port, ttl, dscp, ecn, ip_id,
                df, mf, ihl, total_len, win, flags, dataofs,
                opt_mss, opt_wscale, opt_sackok, opt_ts)
    except Exception:
        return None


MAX_SNAPLEN = 262144   # sane upper bound on any real capture's per-packet length
RESYNC_WINDOW = 4 * 1024 * 1024  # bytes scanned in-memory when a record looks corrupt

def _valid_hdr(ts_s, incl, orig, first_ts, nano):
    if not (0 < incl <= MAX_SNAPLEN): return False, None
    if orig < incl: return False, None
    ts = ts_s
    if first_ts is not None and not (first_ts - 5*365*86400 <= ts <= first_ts + 5*365*86400):
        return False, None
    return True, ts

def find_resync_offset(path, endian, nano, bad_off, first_ts, filesize, window=RESYNC_WINDOW):
    """Scan forward in-memory from bad_off for the next plausible classic-pcap
    record header (sane incl/orig, timestamp near first_ts, and a plausible
    follow-on record), to recover from a corrupted/misaligned record instead of
    stalling forever. Returns an absolute file offset, or None if none found."""
    rec = struct.Struct(endian+'IIII')
    with open(path, 'rb') as f:
        f.seek(bad_off)
        buf = f.read(min(window, max(0, filesize - bad_off)))
    n = len(buf)
    for i in range(1, max(0, n - 16)):
        try:
            ts_s, ts_frac, incl, orig = rec.unpack_from(buf, i)
        except struct.error:
            break
        ts = ts_s + (ts_frac/1e9 if nano else ts_frac/1e6)
        ok, _ = _valid_hdr(ts, incl, orig, first_ts, nano)
        if not ok:
            continue
        end_local = i + 16 + incl
        if end_local + 16 <= n:
            try:
                ts_s2, ts_frac2, incl2, orig2 = rec.unpack_from(buf, end_local)
            except struct.error:
                continue
            if not (0 < incl2 <= MAX_SNAPLEN) or orig2 < incl2:
                continue
        elif end_local > n:
            continue
        return bad_off + i
    return None

def iter_classic(path, resync_log=None):
    with open(path, 'rb') as f:
        gh = f.read(24)
        if len(gh) < 24: return
        magic = gh[:4]
        if magic == b'\xd4\xc3\xb2\xa1':
            endian = '<'; nano = False
        elif magic == b'\xa1\xb2\xc3\xd4':
            endian = '>'; nano = False
        elif magic == b'\x4d\x3c\xb2\xa1':
            endian = '<'; nano = True
        elif magic == b'\xa1\xb2\x3c\x4d':
            endian = '>'; nano = True
        else:
            raise ValueError('not classic pcap')
        linktype = struct.unpack(endian+'I', gh[20:24])[0]
        rec = struct.Struct(endian+'IIII')
        filesize = os.path.getsize(path)
        first_ts = None
        read = f.read
        while True:
            pos = f.tell()
            hdr = read(16)
            if len(hdr) < 16: break
            ts_s, ts_frac, incl, orig = rec.unpack(hdr)
            ts_full = ts_s + (ts_frac/1e9 if nano else ts_frac/1e6)
            ok, _ = _valid_hdr(ts_full, incl, orig, first_ts, nano)
            if ok:
                data = read(incl)
                if len(data) == incl:
                    if first_ts is None: first_ts = ts_full
                    yield ts_full, data, orig, incl, linktype
                    continue
                break  # plausible header, short payload at true EOF: clean stop
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


def iter_pcapng(path):
    with open(path, 'rb') as f:
        read = f.read
        if_linktypes = []
        if_tsresol = []
        data0 = read(4)
        while data0:
            if len(data0) < 4: break
            btype = struct.unpack('<I', data0)[0]
            blen_b = read(4)
            if len(blen_b) < 4: break
            blen = struct.unpack('<I', blen_b)[0]
            if blen < 12: break
            body = read(blen - 12)
            if len(body) < blen - 12: break
            _tail = read(4)  # trailing block total length
            if btype == 0x0A0D0D0A:  # SHB
                pass
            elif btype == 0x00000001:  # IDB
                linktype = struct.unpack('<H', body[0:2])[0]
                tsr = 1e-6
                ooff = 8
                while ooff + 4 <= len(body):
                    ocode, olen = struct.unpack('<HH', body[ooff:ooff+4])
                    oval = body[ooff+4:ooff+4+olen]
                    if ocode == 0: break
                    if ocode == 9 and len(oval) >= 1:
                        r = oval[0]
                        if r & 0x80:
                            tsr = 2.0 ** (-(r & 0x7f))
                        else:
                            tsr = 10.0 ** (-r)
                    ooff += 4 + olen + ((4 - (olen % 4)) % 4)
                if_linktypes.append(linktype)
                if_tsresol.append(tsr)
            elif btype == 0x00000006:  # Enhanced Packet Block
                ifid, th, tl, caplen, origlen = struct.unpack('<IIIII', body[0:20])
                pkt = body[20:20+caplen]
                tsr = if_tsresol[ifid] if ifid < len(if_tsresol) else 1e-6
                lt = if_linktypes[ifid] if ifid < len(if_linktypes) else LINKTYPE_ETHERNET
                ts = ((th << 32) | tl) * tsr
                yield ts, pkt, origlen, caplen, lt
            elif btype == 0x00000003:  # Simple Packet Block
                origlen = struct.unpack('<I', body[0:4])[0]
                pkt = body[4:]
                lt = if_linktypes[0] if if_linktypes else LINKTYPE_ETHERNET
                yield 0.0, pkt, origlen, len(pkt), lt
            data0 = read(4)


def iter_packets(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
    if magic in (b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\xc3\xd4', b'\x4d\x3c\xb2\xa1', b'\xa1\xb2\x3c\x4d'):
        yield from iter_classic(path)
    elif magic == b'\x0a\x0d\x0d\x0a':
        yield from iter_pcapng(path)
    else:
        raise ValueError('unknown pcap magic %r' % magic)


if __name__ == '__main__':
    import time
    path = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    t0 = time.time()
    n=0; ntcp=0; lt=None
    ttl_hist={}
    for ts, data, orig, incl, lt in iter_packets(path):
        n += 1
        r = parse_ip_tcp(data, 0, lt)
        if r and r[0]=='tcp':
            ntcp += 1
            ttl_hist[r[5]] = ttl_hist.get(r[5],0)+1
        if n >= N: break
    dt = time.time()-t0
    print(f"parsed {n} pkts ({ntcp} tcp) in {dt:.2f}s -> {n/dt:,.0f} pkt/s ; linktype={lt}")
    top = sorted(ttl_hist.items(), key=lambda x:-x[1])[:6]
    print("top TTLs:", top)
