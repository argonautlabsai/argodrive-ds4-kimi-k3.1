/* copy_replay.c: ds4_k3_copy.h over a recorded token stream, without a GPU (19 Nov 2026).
   usage: copy_replay <stream> [a b]
   The stream has one token a line: "P <id>" for the prompt, "G <id>" for the answer, in order ("R ..." starts a new
   request; DS4_K3_COPY_DUMP=<file> makes the engine write such a file). The engine's loop is followed: at each step
   the candidate is proposed; when the rule drafts, a block is simulated, its accepted tokens being those equal to the
   recorded answer and its time a + b x rows plain steps (default 0.8 0.5); otherwise one plain step. The answer is
   the recorded one by construction; what is measured is how many forward passes and how much time the drafts save.
   Settings come from the environment as in the engine (DS4_K3_COPY_ROWS, DS4_K3_COPY_MARGIN, DS4_K3_COPY_COST, DS4_K3_COPY_TRACE). */
#include "ds4_k3_copy.h"

static k3_copy c;
static void run(const int *P, uint32_t nP, const int *G, uint32_t nG, double a, double b, int req) {
    if (nG < 2u) return;
    k3_copy_begin(&c, P, nP, 1 << 30, NULL, NULL, 0);
    uint32_t i = 0, blocks = 0, plain = 0, drafted = 0, accepted = 0;
    double t = 0.0;
    k3_copy_push(&c, G[i++]);   /* the first token comes with the prompt pass */
    while (i < nG) {
        for (;;) {
            if (nG - i < 2u) break;
            int prop[K3_COPY_MAXD];
            const uint32_t nd = k3_copy_propose(&c, prop, nG - i - 1u);
            if (nd == 0u) break;
            k3_copy_drafted(&c);
            uint32_t acc = 0;
            while (acc < nd && prop[acc] == G[i + acc]) acc++;   /* nd <= nG - i - 1: the block's own next token is in the record */
            for (uint32_t k = 0; k <= acc; k++) k3_copy_push(&c, G[i + k]);
            i += acc + 1u;
            const double bt = a + b * (double)(nd + 1u);
            t += bt; blocks++; drafted += nd; accepted += acc;
            k3_copy_result(&c, prop, nd, acc, true, nd, bt * (c.t1 > 0.0 ? c.t1 : 1.0));
        }
        if (i >= nG) break;
        k3_copy_plain(&c, 1.0);
        t += 1.0; plain++;
        k3_copy_push(&c, G[i++]);
    }
    printf("request %d: prompt %u, answer %u tokens: %u blocks (drafted %u, accepted %u, %.0f %%), %u plain steps; time %.1f plain steps instead of %u (%+.1f %%)\n",
           req, nP, nG, blocks, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0.0, plain, t, nG - 1u, 100.0 * ((double)(nG - 1u) / t - 1.0));
    k3_copy_report(&c);
    c.active = false;
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: copy_replay <stream> [a b]\n"); return 2; }
    const double a = argc > 3 ? atof(argv[2]) : K3_COPY_COST_A, b = argc > 3 ? atof(argv[3]) : K3_COPY_COST_B;
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    k3_copy_settings(&c);
    if (c.mode == 0) c.mode = 1;
    int *P = NULL, *G = NULL; uint32_t nP = 0, nG = 0, capP = 0, capG = 0; int req = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'R') { if (nG) run(P, nP, G, nG, a, b, req++); nP = nG = 0; continue; }
        if (line[0] != 'P' && line[0] != 'G') continue;
        const int id = atoi(line + 1);
        if (line[0] == 'P') { if (nP == capP) { capP = capP ? capP * 2u : 4096u; P = realloc(P, sizeof(int) * capP); } P[nP++] = id; }
        else { if (nG == capG) { capG = capG ? capG * 2u : 4096u; G = realloc(G, sizeof(int) * capG); } G[nG++] = id; }
    }
    fclose(f);
    if (nG) run(P, nP, G, nG, a, b, req);
    free(P); free(G);
    return 0;
}
