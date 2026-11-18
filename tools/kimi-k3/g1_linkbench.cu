// g1_linkbench: what two cards take in at once, and whether they can hand data to each other (18 Nov 2026).
//   links   pinned host memory -> each card alone, then both at once (N streams of 17.5 MB copies per card), and back
//   fill    expert files -> pinned host memory -> card(s): O_DIRECT reads by R reader threads, each file uploaded as it lands;
//           first card alone, second card alone, then files alternating between the two
//   xdev    a chain across the cards ordered only by cudaStreamWaitEvent, the host-bounce copy of a slab, and the peer copy,
//           each checked against a pattern
// usage: g1_linkbench links|fill|xdev [--dev0 A] [--dev1 B] [--streams N] [--rounds K] [--readers R] [--files M]
//                                      [--dir DIR] [--dir-b DIR2]
// --dev1 may equal --dev0 (one card: the two "cards" then share its link). Build: nvcc -O2 -o g1_linkbench g1_linkbench.cu
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

static const size_t CHUNK = 17547264;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct link_arm { int dev; int dir; int streams, rounds; double secs = 0, t0 = 0, t1 = 0; uint8_t *h = nullptr; };
static std::atomic<int> g_go{0};
static void link_thread(link_arm *a) {
    CK(cudaSetDevice(a->dev));
    uint8_t *d = nullptr;
    CK(cudaMalloc(&d, CHUNK * a->streams));
    std::vector<cudaStream_t> s(a->streams);
    for (auto &x : s) CK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    CK(cudaMemcpy(d, a->h, CHUNK * a->streams, cudaMemcpyHostToDevice));   /* touch everything once */
    CK(cudaDeviceSynchronize());
    g_go.fetch_add(1);
    while (g_go.load() >= 0) std::this_thread::yield();   /* released together */
    a->t0 = now();
    for (int k = 0; k < a->rounds; k++)
        for (int i = 0; i < a->streams; i++)
            CK(a->dir == 0 ? cudaMemcpyAsync(d + i * CHUNK, a->h + i * CHUNK, CHUNK, cudaMemcpyHostToDevice, s[i])
                           : cudaMemcpyAsync(a->h + i * CHUNK, d + i * CHUNK, CHUNK, cudaMemcpyDeviceToHost, s[i]));
    for (auto &x : s) CK(cudaStreamSynchronize(x));
    a->t1 = now(); a->secs = a->t1 - a->t0;
    for (auto &x : s) cudaStreamDestroy(x);
    cudaFree(d);
}
static void run_links(const std::vector<int> &devs, int dir, int streams, int rounds, std::vector<uint8_t *> &hbuf) {
    std::vector<link_arm> arms(devs.size());
    std::vector<std::thread> th;
    g_go = 0;
    for (size_t i = 0; i < devs.size(); i++) { arms[i].dev = devs[i]; arms[i].dir = dir; arms[i].streams = streams; arms[i].rounds = rounds; arms[i].h = hbuf[i]; th.emplace_back(link_thread, &arms[i]); }
    while (g_go.load() < (int)devs.size()) std::this_thread::yield();
    g_go = -1;
    for (auto &t : th) t.join();
    const double gb = (double)CHUNK * streams * rounds / 1e9;
    double first = arms[0].t0, last = arms[0].t1, sum = 0;
    for (auto &a : arms) { if (a.t0 < first) first = a.t0; if (a.t1 > last) last = a.t1; }
    printf("%s", dir == 0 ? "host -> card" : "card -> host");
    for (auto &a : arms) { printf("  | card %d: %.1f GB in %.2f s = %.1f GB/s", a.dev, gb, a.secs, gb / a.secs); sum += gb; }
    if (arms.size() > 1) printf("  | together %.1f GB in %.2f s = %.1f GB/s", sum, last - first, sum / (last - first));
    printf("\n");
}

struct fill_ctx {
    std::vector<std::string> files; std::atomic<size_t> next{0};
    int dev[2]; int policy;   /* 0: all to dev[0], 1: all to dev[1], 2: alternate */
    std::atomic<uint64_t> bytes[2]; std::atomic<uint64_t> read_ns{0}, up_ns{0};
    uint8_t *region[2] = {nullptr, nullptr}; int readers = 0;
};
static void fill_thread(fill_ctx *c, int id) {
    uint8_t *h = nullptr;
    cudaStream_t s[2] = {nullptr, nullptr};
    CK(cudaSetDevice(c->dev[0])); CK(cudaMallocHost((void **)&h, CHUNK + 4096)); CK(cudaStreamCreateWithFlags(&s[0], cudaStreamNonBlocking));
    CK(cudaSetDevice(c->dev[1])); CK(cudaStreamCreateWithFlags(&s[1], cudaStreamNonBlocking));
    uint8_t *buf = (uint8_t *)(((uintptr_t)h + 4095) & ~(uintptr_t)4095);
    for (;;) {
        const size_t i = c->next.fetch_add(1);
        if (i >= c->files.size()) break;
        const double t0 = now();
        const int fd = open(c->files[i].c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) { perror(c->files[i].c_str()); exit(1); }
        size_t got = 0;
        while (got < CHUNK) { const ssize_t r = pread(fd, buf + got, CHUNK - got, (off_t)got); if (r <= 0) { perror("pread"); exit(1); } got += (size_t)r; }
        close(fd);
        const double t1 = now();
        const int w = c->policy == 2 ? (int)(i & 1u) : c->policy;
        CK(cudaSetDevice(c->dev[w]));
        CK(cudaMemcpyAsync(c->region[w] + (size_t)id * CHUNK, buf, CHUNK, cudaMemcpyHostToDevice, s[w]));
        CK(cudaStreamSynchronize(s[w]));
        const double t2 = now();
        c->bytes[w] += CHUNK; c->read_ns += (uint64_t)((t1 - t0) * 1e9); c->up_ns += (uint64_t)((t2 - t1) * 1e9);
    }
    cudaFreeHost(h);
}
static void list_dir(const char *dir, std::vector<std::string> &out, size_t limit) {
    DIR *d = opendir(dir);
    if (!d) { perror(dir); exit(1); }
    std::vector<std::string> names;
    while (struct dirent *e = readdir(d)) { const char *n = e->d_name; const size_t l = strlen(n); if (l > 4 && !strcmp(n + l - 4, ".bin") && n[0] == 'L') names.push_back(n); }
    closedir(d);
    for (auto &n : names) {
        if (out.size() >= limit) break;
        const std::string p = std::string(dir) + "/" + n;
        struct stat st;
        if (stat(p.c_str(), &st) == 0 && (size_t)st.st_size == CHUNK) out.push_back(p);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: g1_linkbench links|fill|xdev [--dev0 A] [--dev1 B] [--streams N] [--rounds K] [--readers R] [--files M] [--dir D] [--dir-b D2]\n"); return 2; }
    const std::string mode = argv[1];
    int dev0 = 0, dev1 = 0, streams = 16, rounds = 48, readers = 64; size_t n_files = 3000;
    const char *dir = nullptr, *dir_b = nullptr;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--dev0") dev0 = atoi(argv[i + 1]); else if (k == "--dev1") dev1 = atoi(argv[i + 1]);
        else if (k == "--streams") streams = atoi(argv[i + 1]); else if (k == "--rounds") rounds = atoi(argv[i + 1]);
        else if (k == "--readers") readers = atoi(argv[i + 1]); else if (k == "--files") n_files = (size_t)atol(argv[i + 1]);
        else if (k == "--dir") dir = argv[i + 1]; else if (k == "--dir-b") dir_b = argv[i + 1];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    if (dev0 >= n_dev || dev1 >= n_dev) { fprintf(stderr, "%d CUDA device(s) visible; --dev0 %d --dev1 %d\n", n_dev, dev0, dev1); return 2; }
    printf("g1_linkbench %s: %d CUDA device(s) visible, first card %d, second card %d%s\n", mode.c_str(), n_dev, dev0, dev1, dev0 == dev1 ? " (the same card)" : "");
    if (mode == "links") {
        std::vector<uint8_t *> hb(2);
        CK(cudaSetDevice(dev0));
        for (auto &h : hb) { CK(cudaMallocHost((void **)&h, CHUNK * streams)); memset(h, 0x5a, CHUNK * streams); }
        for (int dir_ = 0; dir_ < 2; dir_++) {
            std::vector<int> a{dev0}, b{dev1}, both{dev0, dev1};
            run_links(a, dir_, streams, rounds, hb);
            if (dev1 != dev0) run_links(b, dir_, streams, rounds, hb);
            run_links(both, dir_, streams, rounds, hb);
        }
        return 0;
    }
    if (mode == "fill") {
        fill_ctx c;
        if (!dir) { fprintf(stderr, "fill needs --dir <directory of expert files>\n"); return 2; }
        list_dir(dir, c.files, n_files);
        if (dir_b) list_dir(dir_b, c.files, n_files + n_files / 4);
        if (c.files.empty()) { fprintf(stderr, "no expert files in %s\n", dir); return 1; }
        c.dev[0] = dev0; c.dev[1] = dev1; c.readers = readers;
        for (int w = 0; w < 2; w++) { CK(cudaSetDevice(c.dev[w])); CK(cudaMalloc(&c.region[w], CHUNK * (size_t)readers)); }
        printf("%zu files of %.1f MB (%.1f GB per arm), %d readers, O_DIRECT; each arm reads different files than the one before only if --files allows\n",
               c.files.size(), CHUNK / 1e6, c.files.size() * CHUNK / 1e9, readers);
        const size_t per_arm = c.files.size() / 3;
        const std::vector<std::string> all = c.files;
        for (int policy = 0; policy < 3; policy++) {
            if (policy == 1 && dev1 == dev0) continue;
            c.files.assign(all.begin() + policy * per_arm, all.begin() + (policy + 1) * per_arm);   /* never a file the page cache or the drive saw in this run */
            c.policy = policy; c.next = 0; c.bytes[0] = 0; c.bytes[1] = 0; c.read_ns = 0; c.up_ns = 0;
            std::vector<std::thread> th;
            const double t0 = now();
            for (int r = 0; r < readers; r++) th.emplace_back(fill_thread, &c, r);
            for (auto &t : th) t.join();
            const double el = now() - t0, b0 = c.bytes[0] / 1e9, b1 = c.bytes[1] / 1e9;
            printf("%-22s %.1f GB in %.2f s = %.1f GB/s  (first card %.1f GB = %.1f GB/s, second %.1f GB = %.1f GB/s; a reader spent %.0f %% reading, %.0f %% uploading)\n",
                   policy == 0 ? "first card alone:" : policy == 1 ? "second card alone:" : "alternating:", b0 + b1, el, (b0 + b1) / el, b0, b0 / el, b1, b1 / el,
                   100.0 * c.read_ns / (c.read_ns + c.up_ns + 1.0), 100.0 * c.up_ns / (c.read_ns + c.up_ns + 1.0));
        }
        return 0;
    }
    if (mode == "xdev") {
        const size_t n = (size_t)256 << 20;
        uint8_t *pat = nullptr, *hb = nullptr, *hc = nullptr, *a0 = nullptr, *b1 = nullptr;
        cudaStream_t s0, s1; cudaEvent_t ev0, ev1;
        CK(cudaSetDevice(dev0)); CK(cudaMallocHost((void **)&pat, n)); CK(cudaMallocHost((void **)&hb, n)); CK(cudaMallocHost((void **)&hc, n));
        CK(cudaMalloc(&a0, n)); CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking)); CK(cudaEventCreateWithFlags(&ev0, cudaEventDisableTiming));
        CK(cudaSetDevice(dev1)); CK(cudaMalloc(&b1, n)); CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking)); CK(cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming));
        int bad = 0; double secs = 0;
        for (int it = 0; it < 8; it++) {
            uint32_t x = 12345u + (uint32_t)it * 7919u;
            for (size_t i = 0; i < n; i += 4) { x = x * 1664525u + 1013904223u; memcpy(pat + i, &x, 4); }
            memset(hb, 0, n); memset(hc, 0, n);
            const double t0 = now();
            CK(cudaSetDevice(dev0));
            CK(cudaMemcpyAsync(a0, pat, n, cudaMemcpyHostToDevice, s0));
            CK(cudaMemcpyAsync(hb, a0, n, cudaMemcpyDeviceToHost, s0));
            CK(cudaEventRecord(ev0, s0));
            CK(cudaSetDevice(dev1));
            CK(cudaStreamWaitEvent(s1, ev0, 0));                      /* the only thing that orders the two cards */
            CK(cudaMemcpyAsync(b1, hb, n, cudaMemcpyHostToDevice, s1));
            CK(cudaMemcpyAsync(hc, b1, n, cudaMemcpyDeviceToHost, s1));
            CK(cudaEventRecord(ev1, s1));
            CK(cudaSetDevice(dev0));
            CK(cudaStreamWaitEvent(s0, ev1, 0));                      /* and back */
            CK(cudaMemcpyAsync(a0, hc, n, cudaMemcpyHostToDevice, s0));
            CK(cudaMemcpyAsync(hb, a0, n, cudaMemcpyDeviceToHost, s0));
            CK(cudaStreamSynchronize(s0));
            secs += now() - t0;
            if (memcmp(hb, pat, n) != 0 || memcmp(hc, pat, n) != 0) bad++;
        }
        printf("cross-card event chain (card %d -> host -> card %d -> host -> card %d, 256 MB, ordered by cudaStreamWaitEvent alone): %d of 8 rounds wrong, %.1f GB/s through the chain's six copies\n",
               dev0, dev1, dev0, bad, 8.0 * 6.0 * n / 1e9 / secs);
        {   /* a slab from the second card to the first through pinned memory */
            const int k = 256; const double t0 = now();
            for (int i = 0; i < k; i++) {
                CK(cudaSetDevice(dev1)); CK(cudaMemcpyAsync(hb, b1, CHUNK, cudaMemcpyDeviceToHost, s1)); CK(cudaEventRecord(ev1, s1));
                CK(cudaSetDevice(dev0)); CK(cudaStreamWaitEvent(s0, ev1, 0)); CK(cudaMemcpyAsync(a0, hb, CHUNK, cudaMemcpyHostToDevice, s0)); CK(cudaStreamSynchronize(s0));
            }
            const double el = now() - t0;
            printf("host bounce of one expert slab, second card -> first: %.2f ms each (%.1f GB/s end to end)\n", 1e3 * el / k, k * CHUNK / 1e9 / el);
        }
        if (dev0 != dev1) {
            int can01 = 0, can10 = 0;
            CK(cudaDeviceCanAccessPeer(&can01, dev0, dev1)); CK(cudaDeviceCanAccessPeer(&can10, dev1, dev0));
            printf("peer access reported: card %d -> %d %s, %d -> %d %s\n", dev0, dev1, can01 ? "yes" : "no", dev1, dev0, can10 ? "yes" : "no");
            if (can01 && can10) {
                CK(cudaSetDevice(dev0)); cudaError_t e = cudaDeviceEnablePeerAccess(dev1, 0); if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
                CK(cudaSetDevice(dev1)); e = cudaDeviceEnablePeerAccess(dev0, 0); if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
                (void)cudaGetLastError();
            }
            int pbad = 0; double psecs = 0;
            for (int it = 0; it < 8; it++) {   /* cudaMemcpyPeer works with or without peer access; what matters is that the bytes arrive */
                uint32_t x = 999u + (uint32_t)it * 104729u;
                for (size_t i = 0; i < n; i += 4) { x = x * 1664525u + 1013904223u; memcpy(pat + i, &x, 4); }
                CK(cudaSetDevice(dev0)); CK(cudaMemcpy(a0, pat, n, cudaMemcpyHostToDevice));
                const double t0 = now();
                CK(cudaMemcpyPeer(b1, dev1, a0, dev0, n));
                CK(cudaSetDevice(dev1)); CK(cudaDeviceSynchronize());
                psecs += now() - t0;
                memset(hc, 0, n);
                CK(cudaMemcpy(hc, b1, n, cudaMemcpyDeviceToHost));
                if (memcmp(hc, pat, n) != 0) pbad++;
            }
            printf("peer copy card %d -> %d, 256 MB: %d of 8 rounds wrong, %.1f GB/s\n", dev0, dev1, pbad, 8.0 * n / 1e9 / psecs);
        }
        return bad ? 1 : 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
