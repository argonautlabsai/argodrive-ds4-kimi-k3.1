#!/usr/bin/env python3
"""Qwen proposal sidecar for ds4's Kimi K3 decode (steps 79, 82, 86, 89 and 94b, experimental). ds4 sends the committed K3 token ids; this answers with up to `requested` K3 ids or none.
Rules from deltafin's qwen_draft.rs and provider_qwen.cpp: greedy; generated rows stop at, and include, end-of-text; a proposal is cut at the first row below the
confidence gate; translation goes through the decoded K3 text and must keep the history as an exact prefix; a failed round trip gives no proposal (soft skip);
budget requested + 2 clamped to [3, MAX_NEW], with one retry at MAX_NEW when the budget ran out short of `requested`; MAX_NEW is 20, or
DS4_K3_QWEN_ROWS + 4 when that is more (blocks of up to 32 rows since step 120). Generation stops early at the first row below the gate,
because nothing after it can be used, and the KV cache and predicted rows are reused as long as the text agrees. A proposal shorter than DS4_K3_QWEN_KMIN drafts is
dropped, and ds4 runs a block of 1 + n rows for any other. While ds4 works on a block or a plain step, the sidecar can carry its greedy sequence on past the history
(DS4_K3_QWEN_SPEC) and stops as soon as the next request arrives. Binary protocol on stdin/stdout, little endian:
  H                                          -> h  f32 load_seconds, u32 vram_mb
  B  u32 n_prompt                            -> b  (a new request: empty history; the prompt's length, which the test fixtures need)
  Q  u32 n_new, n_new * u32 ids, u32 requested -> q  u32 nd, nd * u32 ids, f32 min_conf, f32 seconds, u32 steps, u32 flags (1 soft skip, 2 confidence stop, 4 retried)
  X                                          -> exits
Environment: DS4_K3_QWEN_DIR (the Qwen3-0.6B-Base checkpoint), DS4_K3_QWEN_META (the folder with K3's tiktoken.model and tokenizer_config.json), DS4_K3_QWEN_CONF (gate, default 0.6),
DS4_K3_QWEN_ADAPT=lo,hi,down,up (default 0.6,0.95,0.02,0.1 except on macOS, 0 or off to disable: the gate starts at DS4_K3_QWEN_CONF, drops by `down` after a proposal whose drafts were all kept and rises by `up` after
one that kept fewer than half, within [lo, hi]), DS4_K3_QWEN_KMIN (fewest drafts worth a block, default 2), DS4_K3_QWEN_SPEC=after_block,after_step (rows to have
ready past the history, default 17,3), DS4_K3_QWEN_TOKENIZE (k3_tokenize.py, default: next to this file), DS4_K3_QWEN_DEVICE (default cuda), DS4_K3_QWEN_KEEP (default 1: a prefix that
agrees with everything consumed keeps the rows computed past it; DS4_K3_QWEN_SPEC then defaults to 2 x DS4_K3_QWEN_ROWS + 3 rows after a block, prepared only after a
proposal of at least DS4_K3_QWEN_SPEC_MIN drafts, default 16),
DS4_K3_QWEN_CTX (the most tokens of history the model is given, default 4096 except on macOS, 0 = all).
Needs torch with CUDA, transformers, tokenizers, tiktoken and regex.
Test fixtures, never set in a real run: DS4_K3_QWEN_TRUTH=<ids file> proposes the true continuation (prompt ids are not in the file: the first id of the file follows the
prompt), DS4_K3_QWEN_FLEN=<n> makes those proposals n drafts long, DS4_K3_QWEN_CORRUPT=<every>:<index> wrongs draft <index> of every <every>-th proposal, DS4_K3_QWEN_BAD=<n> answers malformed at proposal n, DS4_K3_QWEN_DIE=<n> exits at proposal n, DS4_K3_QWEN_HOG=1 takes all the free VRAM at the first proposal, keeps it and proposes nothing."""
import importlib.util, json, os, re, select, struct, sys, time
t_load0 = time.time()
proto_in = os.fdopen(0, "rb", 0)   # unbuffered, so the idle check below sees every waiting byte
proto_out = os.fdopen(os.dup(1), "wb", 0)
os.dup2(2, 1)   # anything else that prints goes to ds4's stderr, never into the protocol
import torch
import tiktoken
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
QW = os.environ["DS4_K3_QWEN_DIR"]; META = os.environ["DS4_K3_QWEN_META"]
TOKPY = os.environ.get("DS4_K3_QWEN_TOKENIZE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_tokenize.py")
GATE = float(os.environ.get("DS4_K3_QWEN_CONF", "0.6"))
_adapt = os.environ.get("DS4_K3_QWEN_ADAPT", "" if sys.platform == "darwin" else "0.6,0.95,0.02,0.1")   # default on CUDA since 10 Nov 2026; 0 or off switches it off
ADAPT = None if _adapt.strip() in ("", "0", "off") else [float(x) for x in _adapt.split(",")]
KMIN = int(os.environ.get("DS4_K3_QWEN_KMIN", "2"))
SLACK, EOS = 2, 151643
ROWS = int(os.environ.get("DS4_K3_QWEN_ROWS", "64"))   # 64 on CUDA with steps 128, 129, 131; on macOS since 11 Nov 2026 (it must match ds4's default)
MAX_NEW = max(20, ROWS + 4)   # step 120: the budget follows the block width
# step 122: a new prefix that agrees with everything consumed keeps the rows computed past it (they are its greedy continuation), so
# the rows prepared while ds4 worked are answered without a forward. 0 restores the crop.
KEEP = os.environ.get("DS4_K3_QWEN_KEEP", "1").strip() not in ("", "0", "off")
# rows to have ready past the history: after a block the next one starts about ROWS tokens later and asks for up to ROWS + SLACK more
SPEC = [int(x) for x in os.environ.get("DS4_K3_QWEN_SPEC", f"{max(17, 2 * ROWS + 3)},3" if KEEP else "17,3").split(",")]
# step 137: the deep rows are prepared only after a proposal of at least DS4_K3_QWEN_SPEC_MIN drafts (default 16, 0 = always), 17 rows
# otherwise. Preparing 131 rows after every block ran beside ds4's own GPU work: on everyday prompts, where blocks stay short, the
# blocks took 4-6 % longer for nothing (11 Nov 2026: 2.05 -> 2.09 and 2.35 -> 2.42 tok/s; record and 512-token prompts unchanged).
SPEC_MIN = int(os.environ.get("DS4_K3_QWEN_SPEC_MIN", "16"))
# step 131: a block wider than 32 rows only when its first half is likely to hold. A 64-row block costs about 4.7 s more than a
# 32-row one on the CUDA test machine and saves about 4.5 s when it is kept whole, so going wide pays when the first 31 drafts are kept with a
# probability above one half; that probability is taken as the product of Qwen's confidences over the rows behind those drafts.
# DS4_K3_QWEN_WIDE=rows,theta (default 32,0.5; 0 or off: every proposal may fill the block). Replayed on the record, code and
# 512-token runs of 10 Nov 2026: the prose runs keep every wide block, the code run loses the 64-row block that kept 23 of 63.
_wd = os.environ.get("DS4_K3_QWEN_WIDE", "32,0.5").strip()
WIDE = None if _wd in ("", "0", "off") else [float(x) for x in _wd.split(",")]
if WIDE and (len(WIDE) < 2 or ROWS <= WIDE[0]): WIDE = None
# step 136: the model is given at most DS4_K3_QWEN_CTX tokens of history (default 4096 on CUDA, 0 = all, the default on macOS, where
# the model shares the machine's memory). A longer history is cut at one token for the whole request, so the rows already computed
# stay valid; a proposal is checked by ds4 whatever it was made from. The cache costs about 0.11 MB of VRAM per token: beside a
# 16k-context target the forward over a 13k-token prompt ran out of memory.
CTX = int(os.environ.get("DS4_K3_QWEN_CTX", "0" if sys.platform == "darwin" else "4096"))
dev = os.environ.get("DS4_K3_QWEN_DEVICE", "cuda")
OOM_ERR = tuple(e for e in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if e) or (MemoryError,)
spec = importlib.util.spec_from_file_location("k3tok", TOKPY); kt = importlib.util.module_from_spec(spec); spec.loader.exec_module(kt); kt.META = META
RANKS = kt.load_ranks(); INV = {v: k for k, v in RANKS.items()}
SPECIAL = {int(k): v["content"] for k, v in json.load(open(os.path.join(META, "tokenizer_config.json")))["added_tokens_decoder"].items()}
SP_INV = {v: k for k, v in SPECIAL.items()}
SP_RX = re.compile("(" + "|".join(re.escape(s) for s in sorted(SP_INV, key=len, reverse=True)) + ")")
ENC = tiktoken.Encoding(name="k3", pat_str=kt.PATTERN.pattern, mergeable_ranks=RANKS, special_tokens={})
def k3_decode(ids):
    out = bytearray()
    for i in ids: out += SPECIAL[i].encode() if i in SPECIAL else INV.get(i, b"")
    return out.decode("utf-8", "replace")
def k3_encode(text):
    ids = []
    for part in SP_RX.split(text):
        if not part: continue
        if part in SP_INV: ids.append(SP_INV[part])
        else: ids.extend(ENC.encode_ordinary(part))
    return ids
QT = Tokenizer.from_file(os.path.join(QW, "tokenizer.json"))
q_encode = lambda t: QT.encode(t, add_special_tokens=False).ids
q_decode = lambda ids: QT.decode(ids, skip_special_tokens=False)
model = AutoModelForCausalLM.from_pretrained(QW, dtype=torch.float16, local_files_only=True).to(dev).eval()
class Qwen:
    """Greedy rows over one growing sequence (step 82). The KV cache and the rows already predicted stay valid for as long as the new prefix agrees with what was
    consumed, so a position usually costs one forward. pred[i] is the greedy (token, probability) after the first i tokens. Numerics can differ from a full forward in the last bits, which can
    only change whether a block is tried: every emitted token is still the target's own."""
    def __init__(s):
        s.ids, s.pred, s.kv, s.forwards = [], {}, None, 0
    def _forward(s, new):
        s.forwards += 1
        out = model(input_ids=torch.tensor([new], device=dev), past_key_values=s.kv, use_cache=True, logits_to_keep=1)
        s.kv = out.past_key_values; s.ids.extend(new)
        lg = out.logits[0, -1].float(); pr = torch.softmax(lg, -1); i = int(lg.argmax())
        s.pred[len(s.ids)] = (i, float(pr[i]))
    @torch.inference_mode()
    def seek(s, prefix):
        n = min(len(prefix), len(s.ids)); lcp = 0
        while lcp < n and prefix[lcp] == s.ids[lcp]: lcp += 1
        if lcp == len(prefix) and lcp not in s.pred: lcp -= 1     # the row after the whole prefix is needed and was never predicted
        if KEEP and lcp == len(prefix): return                     # step 122: every row past the prefix continues it greedily: keep the cache
        if lcp < len(s.ids):
            if lcp <= 0: s.kv = None; s.ids = []; s.pred = {}; lcp = 0
            else:
                s.kv.crop(lcp); s.ids = s.ids[:lcp]
                for k in [k for k in s.pred if k > lcp]: del s.pred[k]
        if lcp < len(prefix): s._forward(prefix[lcp:])
    @torch.inference_mode()
    def speculate(s, upto):
        """While no request waits on stdin, extend the greedy sequence towards row `upto`; the next request reuses what it can and crops the rest."""
        while len(s.ids) < upto and s.kv is not None and not select.select([0], [], [], 0)[0]:
            t, _ = s.pred[len(s.ids)]
            if t == EOS: return
            s._forward([t])
    @torch.inference_mode()
    def rows(s, start, budget, gate):
        """Rows from position start, up to budget, stopping at end-of-text and at the first row under the gate."""
        g, p = [], []; i = start
        while len(g) < budget:
            if i not in s.pred: s._forward([g[-1]])
            t, c = s.pred[i]
            g.append(t); p.append(c); i += 1
            if t == EOS or c < gate: return g, p, True
        return g, p, False
QW_STATE = Qwen()
def once(h, n_h, prefix, g, p, requested, gate):
    acc = next((i for i, x in enumerate(p) if x < gate), len(g))
    tr = k3_encode(q_decode(prefix + g[:acc]))
    if tr[:n_h] != h: return None, acc < len(g)
    d = tr[n_h:][:requested]
    if WIDE and len(d) > WIDE[0] - 1:     # step 131
        lo = int(WIDE[0]) - 1; n_all = max(1, len(tr) - n_h)
        rows = min(acc, max(1, -(-lo * acc // n_all)))     # Qwen rows behind the first lo drafts, by proportion
        pr = 1.0
        for x in p[:rows]: pr *= x
        if pr < WIDE[1]: return d[:lo], True     # counted as a confidence stop: no retry for more rows
    return d, acc < len(g)
@torch.inference_mode()
def propose(h, requested, gate=None):
    global LAST_PLEN, CUT
    gate = GATE if gate is None else gate
    flags = 0; steps = 0
    LAST_PLEN = None
    canon = k3_decode(h)
    if k3_encode(canon) != h: return [], None, 0, 1
    prefix = q_encode(canon)
    if not prefix: return [], None, 0, 1
    if CUT is None: CUT = max(0, len(prefix) - CTX) if CTX > 0 else 0   # step 136: fixed at the request's first proposal
    mp = prefix[CUT:] if CUT else prefix     # what the model is given; `prefix` stays whole for the round trip in once()
    LAST_PLEN = len(mp)   # step 122: the speculation after the answer starts from here, without encoding the history again
    QW_STATE.seek(mp)
    budget = max(3, min(MAX_NEW, requested + SLACK))
    g, p, done = QW_STATE.rows(len(mp), budget, gate)
    d, stopped = once(h, len(h), prefix, g, p, requested, gate)
    if d is None: return [], (min(p) if p else None), len(g), 1
    if len(d) < requested and not stopped and not done and budget != MAX_NEW:
        flags |= 4
        g, p, done = QW_STATE.rows(len(mp), MAX_NEW, gate)
        d, stopped = once(h, len(h), prefix, g, p, requested, gate)
        if d is None: return [], (min(p) if p else None), len(g), flags | 1
    if stopped: flags |= 2
    return d, (min(p) if p else None), len(g), flags
def rd(n):
    b = b""
    while len(b) < n:
        c = proto_in.read(n - len(b))
        if not c: sys.exit(0)
        b += c
    return b
FX_T = [int(x) for x in open(os.environ["DS4_K3_QWEN_TRUTH"]).read().split()] if os.environ.get("DS4_K3_QWEN_TRUTH") else None
FX_C = os.environ.get("DS4_K3_QWEN_CORRUPT"); FX_EV, FX_IX = (int(x) for x in FX_C.split(":")) if FX_C else (0, 0)
FX_BAD = int(os.environ.get("DS4_K3_QWEN_BAD", "0")); FX_DIE = int(os.environ.get("DS4_K3_QWEN_DIE", "0")); FX_LEN = int(os.environ.get("DS4_K3_QWEN_FLEN", "0"))
FX_HOG = os.environ.get("DS4_K3_QWEN_HOG", "").strip() not in ("", "0")   # test fixture: at the first proposal take all the free VRAM, keep it and propose nothing
LAST_PLEN = None
CUT = None      # step 136, per request: tokens of the history's front the model is not given
NO_MEM = False  # step 136, per request: a forward ran out of VRAM; no proposals until the next request
HOG = []
DUMP = os.environ.get("DS4_K3_QWEN_DUMP")   # diagnostics: every request's token ids as ds4 sent them ("B <n_prompt>", then "Q <ids...>" lines), for offline replays
def prefix_len(h):
    return LAST_PLEN if LAST_PLEN is not None else max(0, len(q_encode(k3_decode(h))) - (CUT or 0))
def out_of_memory():
    """step 136: a forward ran out of VRAM (a long history beside a target that left little). Called outside the except block, so
    the failed forward's tensors are gone: drop the cache, hand the memory back and stay quiet until the next request."""
    global NO_MEM
    NO_MEM = True
    QW_STATE.kv = None; QW_STATE.ids = []; QW_STATE.pred = {}
    if dev == "cuda": torch.cuda.empty_cache()
    sys.stderr.write("qwen sidecar: out of VRAM on this request's history; no proposals until the next request\n"); sys.stderr.flush()
def main():
    global CUT, NO_MEM
    h = []; n_q = 0; n_prompt = None; gate = GATE; last = []
    QW_STATE.seek(q_encode("Warm-up of the model: the first call compiles nothing but allocates a lot.")); QW_STATE.rows(len(QW_STATE.ids), 4, 0.0)
    QW_STATE.kv = None; QW_STATE.ids = []; QW_STATE.pred = {}
    if dev == "cuda": torch.cuda.synchronize()
    while True:
        c = rd(1)
        if c == b"H":
            proto_out.write(b"h" + struct.pack("<fI", time.time() - t_load0, int(torch.cuda.memory_reserved() >> 20) if dev == "cuda" else 0))
        elif c == b"B":
            n_prompt = struct.unpack("<I", rd(4))[0]   # ds4 may take plain steps before its first proposal
            if DUMP:
                with open(DUMP, "a") as df: df.write(f"B {n_prompt}\n")
            h = []; n_q = 0; gate = GATE; last = []; QW_STATE.kv = None; QW_STATE.ids = []; QW_STATE.pred = {}; proto_out.write(b"b")
            CUT = None; NO_MEM = False
        elif c == b"Q":
            t0 = time.time()
            n = struct.unpack("<I", rd(4))[0]
            new = list(struct.unpack(f"<{n}I", rd(4 * n)))
            req = struct.unpack("<I", rd(4))[0]
            if n_prompt is None: n_prompt = len(new) - 1
            if DUMP:
                with open(DUMP, "a") as df: df.write("Q " + " ".join(map(str, new)) + "\n")
            h.extend(new); n_q += 1
            if ADAPT and last:     # new holds the drafts that were kept and the next anchor
                a = 0
                while a < len(last) and a < len(new) and new[a] == last[a]: a += 1
                if a == len(last): gate = max(ADAPT[0], gate - ADAPT[2])
                elif 2 * a < len(last): gate = min(ADAPT[1], gate + ADAPT[3])
            if FX_DIE and n_q == FX_DIE: os._exit(3)
            if FX_T is not None:
                k = len(h) - n_prompt     # generated so far, the anchor included
                m = min(FX_LEN, req) if FX_LEN else req
                d = FX_T[k:k + m]; mc, st, fl = 1.0, 0, 0
                if len(d) < m: d = []
                elif FX_EV and n_q % FX_EV == 0 and FX_IX < len(d): d = list(d); d[FX_IX] = 13 if d[FX_IX] != 13 else 11
            elif FX_HOG:
                if not HOG and dev == "cuda":
                    for size in (256 << 20, 16 << 20, 1 << 20):
                        try:
                            while True: HOG.append(torch.empty(size, dtype=torch.uint8, device=dev))
                        except OOM_ERR: pass
                    sys.stderr.write(f"qwen sidecar: fixture holds {sum(t.numel() for t in HOG) >> 20} MiB of VRAM, {torch.cuda.mem_get_info()[0] >> 20} MiB left free\n"); sys.stderr.flush()
                d, mc, st, fl = [], None, 0, 1
            elif NO_MEM:
                d, mc, st, fl = [], None, 0, 1
            else:
                oom = False
                try: d, mc, st, fl = propose(h, req, gate)
                except OOM_ERR: oom = True
                if oom: out_of_memory(); d, mc, st, fl = [], None, 0, 1
            if len(d) < min(KMIN, req): d = []
            last = list(d)
            if FX_BAD and n_q == FX_BAD: proto_out.write(b"q" + struct.pack("<I", 99)); continue
            proto_out.write(b"q" + struct.pack(f"<I{len(d)}IffII", len(d), *d, -1.0 if mc is None else mc, time.time() - t0, st, fl))
            if FX_T is None and not FX_HOG and not NO_MEM and (SPEC[0] or SPEC[1]) and QW_STATE.ids:
                oom = False
                deep = SPEC[0] if (not SPEC_MIN or len(d) >= SPEC_MIN) else min(SPEC[0], 17)   # step 137
                try: QW_STATE.speculate(prefix_len(h) + (deep if d else SPEC[1]))
                except OOM_ERR: oom = True
                if oom: out_of_memory()
        elif c == b"X":
            return
        else:
            sys.stderr.write(f"qwen sidecar: unknown command {c!r}\n"); return
if __name__ == "__main__":
    main()
