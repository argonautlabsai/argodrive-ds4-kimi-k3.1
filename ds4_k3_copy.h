/* ds4_k3_copy.h: drafts copied from the context (step 160), the record of drafts the model turned down (step 161) and
   the rule that sizes a draft from measured costs (step 162). 19 Nov 2026.

   When the last tokens of the context occurred earlier in it (in the prompt or in the answer so far), the tokens that
   followed there are a draft of what comes next: repeated keys of a JSON answer, a figure or a name quoted from the
   prompt, code being edited. No draft model is involved. A draft only chooses which rows a verification block
   computes; the block keeps the tokens that equal the model's own choice and continues from the first that does not,
   so the answer is the plain decode's whatever is proposed here.

   On this engine a drafted row is not free: its experts are read like any other row's, about half a plain step each
   (19 Nov 2026, the two-card server). So a copy is drafted only when its tokens are likely enough to be right:
     - every draft token is put in a class: the length of the match behind it, and whether other earlier occurrences
       of the same context agree with it, disagree, or the model turned this token down in this context before;
     - how often each class was right is counted as the answers arrive (every step scores the first token of that
       step's candidate against the model's choice, drafted or not), starting from the priors below;
     - the time of a block of copied rows is fitted against its rows, in plain steps, from the blocks that ran;
     - a draft is the length with the best expected saving, and it is drafted only if that saving is positive.

   Included by ds4.c. tools/kimi-k3/copy_replay.c runs the same code over recorded token streams without a GPU. */
#ifndef DS4_K3_COPY_H
#define DS4_K3_COPY_H
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K3_COPY_K 3u          /* the indexed match: the last three tokens */
#define K3_COPY_LCAP 32u      /* match lengths are counted up to this */
#define K3_COPY_MAXD 63u      /* the longest draft */
#define K3_COPY_WALK 96u      /* earlier occurrences looked at, the most recent first */
#define K3_COPY_HBITS 18u     /* slots of the three-token index */
#define K3_COPY_RBITS 15u     /* slots of the record of choices */
#define K3_COPY_NLB 6u        /* match-length classes: 3, 4, 5-6, 7-10, 11-18, 19 and more */
#define K3_COPY_NCLS 4u       /* 0 one earlier occurrence, 1 several that agree, 2 they disagree, 3 turned down here before */
#define K3_COPY_NCELL (K3_COPY_NLB * K3_COPY_NCLS)

/* how often a copied token was the model's choice, by class, before anything is measured (weight: 6 observations) */
static const double k3_copy_prior[K3_COPY_NCLS][K3_COPY_NLB] = {
    {0.45, 0.55, 0.65, 0.75, 0.85, 0.92},
    {0.70, 0.78, 0.84, 0.90, 0.94, 0.97},
    {0.20, 0.25, 0.30, 0.40, 0.50, 0.60},
    {0.08, 0.10, 0.12, 0.15, 0.20, 0.25},
};
#define K3_COPY_PRIOR_W 6.0
/* a block of copied rows costs a + b x rows plain steps, before anything is measured (the two-card server, 19 Nov 2026) */
#define K3_COPY_COST_A 0.8
#define K3_COPY_COST_B 0.5

typedef struct { uint32_t tag; uint16_t wins, losses; } k3_copy_ref;
typedef struct { uint32_t pos, len; } k3_copy_cand;
typedef struct {
    /* settings, read once */
    int mode;                 /* 0 off, 1 drafts, 2 shadow: candidates are scored and what the rule would have drafted is counted, no blocks */
    uint32_t max_d;           /* the longest draft: DS4_K3_COPY_ROWS - 1 */
    double margin;            /* DS4_K3_COPY_MARGIN: the expected saving a draft must show, in plain steps */
    double first;             /* DS4_K3_COPY_FIRST: a copy expected to give this many tokens is drafted without asking the draft model */
    double fix_a, fix_b;      /* DS4_K3_COPY_COST=a,b: the block cost, not fitted */
    int trace, cut_qwen;      /* DS4_K3_COPY_TRACE; DS4_K3_REFUTE_QWEN */
    /* the request */
    bool active;
    int special_base;
    int *hist; int32_t *prev; uint32_t n, cap, n_prompt;
    int32_t *head; k3_copy_ref *ref;
    /* the candidate of this step: its tokens, their classes, their keys in the record */
    bool cand_on;             /* its first token is scored against the next token of the answer */
    uint32_t p_n, p_d; double p_exp;
    int p_tok[K3_COPY_MAXD]; uint8_t p_cell[K3_COPY_MAXD]; uint64_t p_key[K3_COPY_MAXD];
    uint32_t p_src, p_len;
    /* shadow: the block the rule would have run */
    bool v_on; uint32_t v_n, v_at; int v_tok[K3_COPY_MAXD];
    /* learned over the process */
    double q_s[K3_COPY_NCELL], q_t[K3_COPY_NCELL];
    double t1;                /* a plain step, seconds */
    double r1, rx, ry, rxx, rxy;   /* block time in plain steps against rows */
    /* the request's counts */
    uint32_t st_steps, st_found, st_scored, st_first_ok, st_blocks, st_drafted, st_acc, st_full, st_zero, st_prompt, st_answer;
    uint32_t st_ref, st_swapped, st_qloss, st_qcut, st_qcut_right, st_qcut_lost;
    uint32_t sv_blocks, sv_drafted, sv_acc; double sv_saved, st_block_s;
} k3_copy;

static inline uint32_t k3_copy_h3(const int *t) {
    uint64_t h = ((uint64_t)(uint32_t)t[0] + 1u) * 0x9e3779b97f4a7c15ull;
    h = (h ^ (uint32_t)t[1]) * 0xc2b2ae3d27d4eb4full; h ^= h >> 31;
    h = (h ^ (uint32_t)t[2]) * 0x165667b19e3779f9ull; h ^= h >> 29;
    return (uint32_t)(h >> (64u - K3_COPY_HBITS));
}
/* the record's key: the three tokens before a position and a token proposed or chosen there */
static inline uint64_t k3_copy_key(const int *ctx3, int tok) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t)(uint32_t)ctx3[0];
    h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h ^= (uint64_t)(uint32_t)ctx3[1];
    h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 33; h ^= (uint64_t)(uint32_t)ctx3[2];
    h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h ^= (uint64_t)(uint32_t)tok;
    h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 33;
    return h;
}
static inline k3_copy_ref *k3_copy_ref_get(k3_copy *c, uint64_t key, bool make) {
    k3_copy_ref *r = &c->ref[key >> (64u - K3_COPY_RBITS)];
    const uint32_t tag = (uint32_t)key | 1u;
    if (r->tag == tag) return r;
    if (!make) return NULL;
    r->tag = tag; r->wins = 0; r->losses = 0;   /* the slot's earlier key is forgotten */
    return r;
}
static inline bool k3_copy_refuted(const k3_copy *c, uint64_t key) {
    const k3_copy_ref *r = k3_copy_ref_get((k3_copy *)c, key, false);
    return r && r->losses > r->wins;
}
static inline uint32_t k3_copy_lb(uint32_t len) { return len <= 3u ? 0u : len == 4u ? 1u : len <= 6u ? 2u : len <= 10u ? 3u : len <= 18u ? 4u : 5u; }
static inline double k3_copy_q(const k3_copy *c, uint32_t cell) {
    return (c->q_s[cell] + K3_COPY_PRIOR_W * k3_copy_prior[cell / K3_COPY_NLB][cell % K3_COPY_NLB]) / (c->q_t[cell] + K3_COPY_PRIOR_W);
}
static inline void k3_copy_score(k3_copy *c, uint32_t cell, bool right) {
    if (c->q_t[cell] >= 512.0) { c->q_t[cell] *= 0.5; c->q_s[cell] *= 0.5; }   /* old counts fade */
    c->q_t[cell] += 1.0; if (right) c->q_s[cell] += 1.0;
}
/* a block of rows costs a + b x rows plain steps: the fit over the copy blocks so far, with the prior as six of them */
static void k3_copy_cost(const k3_copy *c, double *a, double *b) {
    if (c->fix_b > 0.0) { *a = c->fix_a; *b = c->fix_b; return; }
    const double w = 3.0, y2 = K3_COPY_COST_A + 2.0 * K3_COPY_COST_B, y10 = K3_COPY_COST_A + 10.0 * K3_COPY_COST_B;
    const double s1 = c->r1 + 2.0 * w, sx = c->rx + w * 12.0, sy = c->ry + w * (y2 + y10), sxx = c->rxx + w * 104.0, sxy = c->rxy + w * (2.0 * y2 + 10.0 * y10);
    const double det = s1 * sxx - sx * sx;
    double bb = det > 1e-9 ? (s1 * sxy - sx * sy) / det : K3_COPY_COST_B;
    if (bb < 0.10) bb = 0.10;
    if (bb > 1.50) bb = 1.50;
    double aa = (sy - bb * sx) / s1;
    if (aa < 0.0) aa = 0.0;
    if (aa > 2.5) aa = 2.5;
    *a = aa; *b = bb;
}
static void k3_copy_settings(k3_copy *c) {
    const char *e = getenv("DS4_K3_COPY");
    c->mode = !e || !e[0] || !strcmp(e, "0") || !strcmp(e, "off") ? 0 : !strcmp(e, "shadow") ? 2 : 1;
    e = getenv("DS4_K3_COPY_ROWS");
    int rows = e ? atoi(e) : 16;
    if (rows < 2) rows = 2;
    if (rows > (int)K3_COPY_MAXD + 1) rows = (int)K3_COPY_MAXD + 1;
    c->max_d = (uint32_t)rows - 1u;
    e = getenv("DS4_K3_COPY_MARGIN"); c->margin = e ? atof(e) : 0.10;
    e = getenv("DS4_K3_COPY_FIRST"); c->first = e ? atof(e) : 3.0;
    c->fix_a = c->fix_b = 0.0;
    e = getenv("DS4_K3_COPY_COST");
    if (e) { double a = 0.0, b = 0.0; if (sscanf(e, "%lf,%lf", &a, &b) == 2 && b > 0.0) { c->fix_a = a; c->fix_b = b; } }
    c->trace = getenv("DS4_K3_COPY_TRACE") != NULL;
    e = getenv("DS4_K3_REFUTE_QWEN"); c->cut_qwen = e && e[0] && strcmp(e, "0") != 0;
}
static void k3_copy_append(k3_copy *c, int tok) {
    if (c->n == c->cap) {
        c->cap = c->cap ? c->cap * 2u : 8192u;
        c->hist = realloc(c->hist, sizeof(int) * c->cap); c->prev = realloc(c->prev, sizeof(int32_t) * c->cap);
        if (!c->hist || !c->prev) { fprintf(stderr, "ds4: K3 copy: no memory for the context index\n"); abort(); }
    }
    const uint32_t i = c->n++;
    c->hist[i] = tok; c->prev[i] = -2;   /* -2: no three-token match ends here */
    if (i + 1u >= K3_COPY_K && tok >= 0 && c->hist[i - 1u] >= 0 && c->hist[i - 2u] >= 0) {
        const uint32_t h = k3_copy_h3(c->hist + i - 2u);
        c->prev[i] = c->head[h]; c->head[h] = (int32_t)i;
    }
}
/* a new request: the context is the prompt. Positions in skip ranges (image rows: every image's ids are the same
   placeholder) are never matched or copied. */
static void k3_copy_begin(k3_copy *c, const int *ids, uint32_t n_ids, int special_base, const uint32_t *skip_pos, const uint32_t *skip_n, uint32_t n_skip) {
    c->active = false; c->cand_on = false; c->v_on = false; c->p_n = c->p_d = 0;
    if (c->mode == 0) return;
    if (!c->head) {
        c->head = malloc(sizeof(int32_t) << K3_COPY_HBITS); c->ref = malloc(sizeof(k3_copy_ref) << K3_COPY_RBITS);
        if (!c->head || !c->ref) { fprintf(stderr, "ds4: K3 copy: no memory for the context index; off\n"); c->mode = 0; return; }
    }
    memset(c->head, 0xff, sizeof(int32_t) << K3_COPY_HBITS);
    memset(c->ref, 0, sizeof(k3_copy_ref) << K3_COPY_RBITS);
    c->n = 0; c->special_base = special_base;
    c->st_steps = c->st_found = c->st_scored = c->st_first_ok = c->st_blocks = c->st_drafted = c->st_acc = c->st_full = c->st_zero = c->st_prompt = c->st_answer = 0;
    c->st_ref = c->st_swapped = c->st_qloss = c->st_qcut = c->st_qcut_right = c->st_qcut_lost = 0;
    c->sv_blocks = c->sv_drafted = c->sv_acc = 0; c->sv_saved = 0.0; c->st_block_s = 0.0;
    for (uint32_t i = 0; i < n_ids; i++) {
        int t = ids[i];
        for (uint32_t s = 0; s < n_skip; s++) if (i >= skip_pos[s] && i - skip_pos[s] < skip_n[s]) { t = -1; break; }
        k3_copy_append(c, t);
    }
    c->n_prompt = n_ids;
    c->active = true;
}
static void k3_copy_shadow_end(k3_copy *c) {   /* the block the rule would have run: its drafts, those the answer confirmed */
    double a, b; k3_copy_cost(c, &a, &b);
    c->sv_blocks++; c->sv_drafted += c->v_n; c->sv_acc += c->v_at;
    c->sv_saved += (double)(c->v_at + 1u) - (a + b * (double)(c->v_n + 1u));
    c->v_on = false;
}
/* a token of the answer, as the model chose it: recorded, scored against the candidate of the step before, indexed */
static void k3_copy_push(k3_copy *c, int tok) {
    if (!c->active) return;
    if (c->n >= K3_COPY_K) {
        k3_copy_ref *r = k3_copy_ref_get(c, k3_copy_key(c->hist + c->n - 3u, tok), true);
        if (r->wins < 65535u) r->wins++;
    }
    if (c->cand_on) {
        c->cand_on = false;
        const bool right = tok == c->p_tok[0];
        k3_copy_score(c, c->p_cell[0], right);
        c->st_scored++;
        if (right) c->st_first_ok++;
        else { k3_copy_ref *r = k3_copy_ref_get(c, c->p_key[0], true); if (r->losses < 65535u) r->losses++; c->st_ref++; }
    }
    if (c->v_on) {
        if (c->v_at < c->v_n && tok == c->v_tok[c->v_at]) c->v_at++;   /* a draft the block would have kept */
        else k3_copy_shadow_end(c);                                    /* the block's own next token */
    }
    k3_copy_append(c, tok);
}
/* the candidate of this step. Returns the number of tokens to draft (0: none) and writes them to out. */
static uint32_t k3_copy_propose(k3_copy *c, int *out, uint32_t want) {
    c->cand_on = false; c->p_n = 0; c->p_d = 0; c->p_exp = 0.0;
    if (!c->active || want == 0u || c->n <= K3_COPY_K) return 0u;
    c->st_steps++;
    const uint32_t n = c->n, e = n - 1u;
    if (c->prev[e] == -2) return 0u;
    k3_copy_cand cand[K3_COPY_WALK]; uint32_t nc = 0, walked = 0;
    for (int32_t p = c->prev[e]; p >= 0 && walked < K3_COPY_WALK; p = c->prev[p], walked++) {
        uint32_t len = 0;
        while (len < K3_COPY_LCAP && (uint32_t)p >= len && c->hist[(uint32_t)p - len] >= 0 && c->hist[(uint32_t)p - len] == c->hist[e - len]) len++;
        if (len >= K3_COPY_K) { cand[nc].pos = (uint32_t)p; cand[nc].len = len; nc++; }   /* shorter: another context in the same slot */
    }
    if (nc == 0u) return 0u;
    c->st_found++;
    uint32_t best = 0;
    for (uint32_t i = 1; i < nc; i++) if (cand[i].len > cand[best].len) best = i;   /* the longest match; among equals the most recent */
    int w[3] = {c->hist[n - 3u], c->hist[n - 2u], c->hist[n - 1u]};
    if (k3_copy_refuted(c, k3_copy_key(w, c->hist[cand[best].pos + 1u]))) {
        /* step 161: the model turned this continuation down here before; another occurrence that continues otherwise takes its place */
        const int t0 = c->hist[cand[best].pos + 1u];
        uint32_t alt = nc;
        for (uint32_t i = 0; i < nc; i++) {
            const int ti = c->hist[cand[i].pos + 1u];
            if (i == best || ti == t0 || ti < 0 || k3_copy_refuted(c, k3_copy_key(w, ti))) continue;
            if (alt == nc || cand[i].len > cand[alt].len) alt = i;
        }
        if (alt < nc) { best = alt; c->st_swapped++; }
    }
    const uint32_t bp = cand[best].pos, bl = cand[best].len;
    const uint32_t vmin = (bl + 1u) / 2u > K3_COPY_K ? (bl + 1u) / 2u : K3_COPY_K;   /* an occurrence votes if its match is at least half the best's */
    uint8_t alive[K3_COPY_WALK];
    for (uint32_t i = 0; i < nc; i++) alive[i] = i != best && cand[i].len >= vmin;
    uint32_t maxn = want < c->max_d ? want : c->max_d;
    if (maxn > K3_COPY_MAXD) maxn = K3_COPY_MAXD;
    double ca, cb; k3_copy_cost(c, &ca, &cb);
    double P = 1.0, A = 0.0, best_gain = 0.0, best_exp = 0.0; uint32_t best_d = 0;
    for (uint32_t j = 0; j < maxn; j++) {
        const uint32_t src = bp + 1u + j;
        const int tok = src < n ? c->hist[src] : c->p_tok[src - n];   /* past the end of the context: the draft continues from itself */
        if (tok < 0 || tok >= c->special_base) break;
        uint32_t agree = 0, dissent = 0;
        for (uint32_t i = 0; i < nc; i++) {
            if (!alive[i]) continue;
            const uint32_t s = cand[i].pos + 1u + j;
            if (s >= n) { alive[i] = 0; continue; }
            if (c->hist[s] == tok) agree++; else { dissent++; alive[i] = 0; }
        }
        const uint64_t key = k3_copy_key(w, tok);
        const uint32_t cls = k3_copy_refuted(c, key) ? 3u : dissent * 3u > agree ? 2u : agree ? 1u : 0u;
        const uint32_t lj = bl + j > K3_COPY_LCAP ? K3_COPY_LCAP : bl + j;
        const uint32_t cell = cls * K3_COPY_NLB + k3_copy_lb(lj);
        c->p_tok[j] = tok; c->p_cell[j] = (uint8_t)cell; c->p_key[j] = key; c->p_n = j + 1u;
        P *= k3_copy_q(c, cell); A += P;
        const double gain = 1.0 + A - (ca + cb * (double)(j + 2u));   /* plain steps saved by a block of j + 1 drafts */
        if (gain > best_gain) { best_gain = gain; best_d = j + 1u; best_exp = A; }
        w[0] = w[1]; w[1] = w[2]; w[2] = tok;
    }
    if (c->p_n == 0u) return 0u;
    c->cand_on = true; c->p_src = bp; c->p_len = bl;
    if (best_gain > c->margin) { c->p_d = best_d; c->p_exp = best_exp; }
    if (c->trace) fprintf(stderr, "ds4: K3 copy at %u: %u tokens match at %u (%s), %u occurrences, first class %u (q %.2f); the rule drafts %u of %u (saving %.2f plain steps)\n",
                          e, bl, bp, bp + 1u < c->n_prompt ? "prompt" : "answer", nc, (unsigned)c->p_cell[0] / K3_COPY_NLB, k3_copy_q(c, c->p_cell[0]), c->p_d, c->p_n, best_gain);
    if (c->mode == 2) {
        if (c->p_d && !c->v_on) { c->v_on = true; c->v_n = c->p_d; c->v_at = 0; memcpy(c->v_tok, c->p_tok, sizeof(int) * c->p_d); }
        return 0u;
    }
    if (c->p_d == 0u) return 0u;
    memcpy(out, c->p_tok, sizeof(int) * c->p_d);
    return c->p_d;
}
/* is this step's copy sure enough to be drafted without asking the draft model? */
static inline bool k3_copy_sure(const k3_copy *c) { return c->p_d > 0u && c->p_exp >= c->first; }
/* the copy is drafted: the block's result scores it, not the next token */
static inline void k3_copy_drafted(k3_copy *c) { c->cand_on = false; if (c->p_src + 1u < c->n_prompt) c->st_prompt++; else c->st_answer++; }
/* a plain step took this long */
static inline void k3_copy_plain(k3_copy *c, double seconds) { if (c->active && seconds > 0.0) c->t1 = c->t1 > 0.0 ? 0.8 * c->t1 + 0.2 * seconds : seconds; }
/* step 161: where the record says a proposal of another source would be turned down (nd: nowhere) */
static uint32_t k3_copy_cut(const k3_copy *c, const int *drafts, uint32_t nd) {
    if (!c->active || c->n < K3_COPY_K) return nd;
    int w[3] = {c->hist[c->n - 3u], c->hist[c->n - 2u], c->hist[c->n - 1u]};
    for (uint32_t j = 0; j < nd; j++) {
        if (k3_copy_refuted(c, k3_copy_key(w, drafts[j]))) return j;
        w[0] = w[1]; w[1] = w[2]; w[2] = drafts[j];
    }
    return nd;
}
/* a block ran over drafts[0..nd): the first acc were the model's choice. The context already ends with the block's
   own next token. from_copy: the drafts were this step's copy; otherwise another source's (cut: where k3_copy_cut put the mark). */
static void k3_copy_result(k3_copy *c, const int *drafts, uint32_t nd, uint32_t acc, bool from_copy, uint32_t cut, double seconds) {
    if (!c->active) return;
    if (from_copy) {
        for (uint32_t j = 0; j < nd && j <= acc && j < c->p_n; j++) k3_copy_score(c, c->p_cell[j], j < acc);
        if (acc < nd && acc < c->p_n) { k3_copy_ref *r = k3_copy_ref_get(c, c->p_key[acc], true); if (r->losses < 65535u) r->losses++; c->st_ref++; }
        c->st_blocks++; c->st_drafted += nd; c->st_acc += acc; c->st_block_s += seconds;
        if (acc == nd) c->st_full++;
        if (acc == 0u) c->st_zero++;
        if (c->t1 > 0.0 && seconds > 0.0) {   /* the block's time in plain steps against its rows; older blocks fade */
            const double x = (double)(nd + 1u), y = seconds / c->t1, f = 0.985;
            c->r1 = c->r1 * f + 1.0; c->rx = c->rx * f + x; c->ry = c->ry * f + y; c->rxx = c->rxx * f + x * x; c->rxy = c->rxy * f + x * y;
        }
    } else {
        if (acc < nd && c->n >= 4u) {   /* the token the block turned down, after the three tokens before the block's own next token */
            k3_copy_ref *r = k3_copy_ref_get(c, k3_copy_key(c->hist + c->n - 4u, drafts[acc]), true);
            if (r->losses < 65535u) r->losses++;
            c->st_qloss++;
        }
        if (cut < nd) { c->st_qcut++; if (acc <= cut) c->st_qcut_right++; else c->st_qcut_lost += acc - cut; }
    }
}
static void k3_copy_report(k3_copy *c) {
    if (!c->active) return;
    if (c->v_on) k3_copy_shadow_end(c);
    if (c->st_steps == 0u) return;
    double a, b; k3_copy_cost(c, &a, &b);
    fprintf(stderr, "ds4: K3 copy%s: %u of %u steps had an earlier occurrence of the last %u tokens; first token right %u of %u scored; blocks %u (prompt %u, answer %u): "
            "drafted %u, accepted %u (%.1f %%), full %u, zero %u, %.2f s; turned down: %u recorded, %u first choices replaced; block cost %.2f + %.2f x rows plain steps, plain step %.3f s\n",
            c->mode == 2 ? " (shadow)" : "", c->st_found, c->st_steps, K3_COPY_K, c->st_first_ok, c->st_scored, c->st_blocks, c->st_prompt, c->st_answer, c->st_drafted, c->st_acc,
            c->st_drafted ? 100.0 * (double)c->st_acc / (double)c->st_drafted : 0.0, c->st_full, c->st_zero, c->st_block_s, c->st_ref, c->st_swapped, a, b, c->t1);
    if (c->mode == 2) fprintf(stderr, "ds4: K3 copy (shadow): the rule would have run %u blocks: drafted %u, accepted %u; about %.1f plain steps saved at that block cost\n",
                              c->sv_blocks, c->sv_drafted, c->sv_acc, c->sv_saved);
    if (c->st_qloss || c->st_qcut) fprintf(stderr, "ds4: K3 copy: the draft model's proposals: %u turned-down tokens recorded; the record marked %u proposals: %u cuts would have cost nothing, the others %u accepted tokens%s\n",
                                           c->st_qloss, c->st_qcut, c->st_qcut_right, c->st_qcut_lost, c->cut_qwen ? " (cuts applied)" : " (cuts not applied)");
    if (c->trace) {
        static const char *const cls[K3_COPY_NCLS] = {"one occurrence", "several agree", "they disagree", "turned down before"};
        for (uint32_t k = 0; k < K3_COPY_NCLS; k++) {
            fprintf(stderr, "ds4: K3 copy: right by match length 3 / 4 / 5-6 / 7-10 / 11-18 / 19+, %-18s:", cls[k]);
            for (uint32_t l = 0; l < K3_COPY_NLB; l++) fprintf(stderr, " %.2f (%.0f)", k3_copy_q(c, k * K3_COPY_NLB + l), c->q_t[k * K3_COPY_NLB + l]);
            fprintf(stderr, "\n");
        }
    }
}
#endif
