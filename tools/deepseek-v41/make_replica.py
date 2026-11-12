#!/usr/bin/env python3
"""Write an experts-only replica of a DeepSeek V4.1 GGUF for DS4_ARGODRIVE_REPLICAS.

The reader only ever asks a replica for bytes of the routed expert tensors (ffn_gate_exps, ffn_up_exps,
ffn_down_exps of every layer). This writes a sparse file of the model's full size that holds exactly those tensors at
their own offsets and nothing else, so a second or third drive needs room for the experts only (289 GB of the
501.6 GB October checkpoint). Every range is read back uncached and compared with the source before the file gets its
final name.

usage: make_replica.py <model.gguf> <replica path> [--threads N] [--list]"""
import fcntl, hashlib, os, struct, sys, threading, time
F_NOCACHE = 48
args = [a for a in sys.argv[1:] if not a.startswith('--')]
opts = sys.argv[1:]
threads = int(opts[opts.index('--threads') + 1]) if '--threads' in opts else 8
src = args[0]

def tensor_table(path):
    f = open(path, 'rb')
    magic, ver, nt, nkv = struct.unpack('<4sIQQ', f.read(24))
    if magic != b'GGUF': sys.exit('not a GGUF file')
    def rstr():
        n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode('utf-8', 'replace')
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    align = 32
    def skip(t):
        nonlocal align
        if t == 8: rstr(); return None
        if t == 9:
            et, n = struct.unpack('<IQ', f.read(12))
            if et == 8:
                for _ in range(n): rstr()
            else: f.seek(size[et] * n, 1)
            return None
        raw = f.read(size[t]); return raw
    for _ in range(nkv):
        k = rstr(); t, = struct.unpack('<I', f.read(4)); raw = skip(t)
        if k == 'general.alignment' and raw is not None: align = struct.unpack('<I', raw)[0]
    infos = []
    for _ in range(nt):
        name = rstr(); nd, = struct.unpack('<I', f.read(4))
        dims = struct.unpack('<%dQ' % nd, f.read(8 * nd)); typ, off = struct.unpack('<IQ', f.read(12))
        infos.append([name, dims, typ, off])
    data = (f.tell() + align - 1) // align * align
    total = os.path.getsize(path)
    order = sorted(range(len(infos)), key=lambda i: infos[i][3])
    for j, i in enumerate(order):
        end = infos[order[j + 1]][3] if j + 1 < len(order) else total - data
        infos[i].append(end - infos[i][3])           # upper bound of the tensor's bytes (to the next tensor)
    f.close()
    return data, infos

data, infos = tensor_table(src)
want = [(data + off, span, name) for name, dims, typ, off, span in infos
        if any(k in name for k in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'))]
want.sort()
total = sum(n for _, n, _ in want)
print('%d tensors in the file, %d routed expert tensors, %.1f GB of %.1f GB' % (
    len(infos), len(want), total / 1e9, os.path.getsize(src) / 1e9))
if '--list' in opts:
    for off, n, name in want[:6] + want[-3:]: print('  %14d %12d %s' % (off, n, name))
    sys.exit(0)
dst = args[1]
if os.path.exists(dst) or os.path.exists(dst + '.part'): sys.exit('replica path exists')
st = os.statvfs(os.path.dirname(dst))
if st.f_bavail * st.f_frsize < total + (20 << 30): sys.exit('not enough room for %.1f GB' % (total / 1e9))
size = os.path.getsize(src)
sfd = os.open(src, os.O_RDONLY); fcntl.fcntl(sfd, F_NOCACHE, 1)
dfd = os.open(dst + '.part', os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644); fcntl.fcntl(dfd, F_NOCACHE, 1)
os.ftruncate(dfd, size)
BLOCK = 64 << 20
jobs = [(off + p, min(BLOCK, n - p)) for off, n, _ in want for p in range(0, n, BLOCK)]
lock = threading.Lock(); state = {'i': 0, 'bytes': 0, 'err': None}; t0 = time.time()
def run(verify):
    def worker():
        while True:
            with lock:
                if state['err'] or state['i'] >= len(jobs): return
                off, n = jobs[state['i']]; state['i'] += 1
            try:
                a = os.pread(sfd, n, off)
                if len(a) != n: raise OSError('short read of the source at %d' % off)
                if verify:
                    b = os.pread(dfd, n, off)
                    if a != b: raise OSError('replica differs from the source at %d' % off)
                else:
                    mv = memoryview(a); w = 0
                    while w < n: w += os.pwrite(dfd, mv[w:], off + w)
            except OSError as e:
                with lock: state['err'] = str(e)
                return
            with lock:
                state['bytes'] += n
    state.update(i=0, bytes=0); t = time.time()
    ts = [threading.Thread(target=worker) for _ in range(threads)]
    [x.start() for x in ts]; [x.join() for x in ts]
    if state['err']:
        os.close(dfd); os.unlink(dst + '.part'); sys.exit(('verification' if verify else 'copy') + ' failed: ' + state['err'])
    print('%s: %.1f GB in %.0f s (%.2f GB/s)' % ('verified' if verify else 'copied', state['bytes'] / 1e9, time.time() - t, state['bytes'] / 1e9 / (time.time() - t)), flush=True)
run(False)
fcntl.fcntl(dfd, 51)          # F_FULLFSYNC
run(True)
os.close(dfd); os.close(sfd)
os.rename(dst + '.part', dst)
print('replica ready: %s (%d bytes logical, %.1f GB written)' % (dst, size, total / 1e9))
