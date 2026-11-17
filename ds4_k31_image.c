/* Kimi K3.1 image input: decoding and Moonshot's preprocessing, pixel for pixel (see ds4_k31_image.h).
 * Build without fast math and without contraction (-ffp-contract=off): the float steps must round as numpy's do. */
#include "ds4_k31_image.h"

#include "third_party/iris/jpeg.h"   /* declarations only: ds4_image.c holds the implementations */
#include "third_party/iris/png.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K31I_PATCH 14
#define K31I_FACTOR 28           /* merge kernel 2 x patch 14 */
#define K31I_LIMIT 65536.0       /* in_patch_limit */
#define K31I_ONE_SIDE 512        /* patch_limit_on_one_side */
#define K31I_PRECISION 22        /* Pillow's PRECISION_BITS for 8-bit images */

static int k31i_error(char *error, size_t cap, const char *msg) {
    if (error && cap) snprintf(error, cap, "%s", msg);
    return 0;
}

int ds4_k31_image_decode(ds4_k31_image *out, const uint8_t *encoded, size_t len, char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!encoded || len < 8) return k31i_error(error, error_cap, "image is empty");
    const uint8_t *src = NULL;
    int w = 0, h = 0, c = 0;
    png_image *png = NULL;
    jpeg_image *jpg = NULL;
    if (memcmp(encoded, "\x89PNG\r\n\x1a\n", 8) == 0) {
        png = png_load_mem(encoded, len);
        if (!png) return k31i_error(error, error_cap, "invalid or unsupported PNG image");
        src = png->data; w = png->width; h = png->height; c = png->channels;
    } else if (encoded[0] == 0xff && encoded[1] == 0xd8) {
        jpg = jpeg_load_mem(encoded, len);   /* no EXIF rotation: PIL's Image.open does not apply it */
        if (!jpg) return k31i_error(error, error_cap, "invalid or unsupported JPEG image");
        src = jpg->data; w = jpg->width; h = jpg->height; c = jpg->channels;
    } else {
        return k31i_error(error, error_cap, "image must be JPEG or PNG");
    }
    int ok = w > 0 && h > 0 && (c == 1 || c == 2 || c == 3 || c == 4) && (size_t)w * (size_t)h <= ((size_t)64 << 20);
    if (ok) {
        out->rgba = malloc((size_t)w * h * 4u);
        ok = out->rgba != NULL;
    }
    if (ok) {
        for (size_t i = 0; i < (size_t)w * h; i++) {
            const uint8_t *p = src + i * (size_t)c;
            uint8_t *q = out->rgba + i * 4u;
            if (c <= 2) { q[0] = q[1] = q[2] = p[0]; q[3] = c == 2 ? p[1] : 255; }
            else { q[0] = p[0]; q[1] = p[1]; q[2] = p[2]; q[3] = c == 4 ? p[3] : 255; }
        }
        out->width = (uint32_t)w; out->height = (uint32_t)h; out->has_alpha = c == 2 || c == 4;
    }
    if (png) png_free(png);
    if (jpg) jpeg_free(jpg);
    if (!ok) { free(out->rgba); out->rgba = NULL; return k31i_error(error, error_cap, "image is too large or has an unsupported layout"); }
    return 1;
}

void ds4_k31_image_free(ds4_k31_image *image) {
    if (!image) return;
    free(image->rgba);
    memset(image, 0, sizeof(*image));
}

/* ---- Pillow's bicubic resample (libImaging/Resample.c, 8 bits per channel) ---- */
static double k31i_bicubic(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

typedef struct { int ksize; int *bounds; int32_t *kk; } k31i_coeffs;

static int k31i_precompute(int in_size, int out_size, k31i_coeffs *c) {
    const float in0 = 0.0f, in1 = (float)in_size;   /* the default box (0, 0, width, height), as floats */
    double scale = (double)(in1 - in0) / out_size, filterscale = scale;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = 2.0 * filterscale;
    const int ksize = (int)ceil(support) * 2 + 1;
    double *pre = malloc((size_t)out_size * ksize * sizeof(double));
    c->bounds = malloc((size_t)out_size * 2 * sizeof(int));
    c->kk = malloc((size_t)out_size * ksize * sizeof(int32_t));
    if (!pre || !c->bounds || !c->kk) { free(pre); return 0; }
    const double inv_filterscale = 1.0 / filterscale;
    for (int xx = 0; xx < out_size; xx++) {
        const double center = in0 + (xx + 0.5) * scale;
        double ww = 0.0;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double *k = &pre[(size_t)xx * ksize];
        int x;
        for (x = 0; x < xmax; x++) {
            const double w = k31i_bicubic((x + xmin - center + 0.5) * inv_filterscale);
            k[x] = w;
            ww += w;
        }
        if (ww != 0.0)
            for (x = 0; x < xmax; x++) k[x] /= ww;
        for (; x < ksize; x++) k[x] = 0;
        c->bounds[xx * 2 + 0] = xmin;
        c->bounds[xx * 2 + 1] = xmax;
    }
    for (size_t i = 0; i < (size_t)out_size * ksize; i++)
        c->kk[i] = pre[i] < 0 ? (int)(-0.5 + pre[i] * (1 << K31I_PRECISION)) : (int)(0.5 + pre[i] * (1 << K31I_PRECISION));
    free(pre);
    c->ksize = ksize;
    return 1;
}

static void k31i_coeffs_free(k31i_coeffs *c) { free(c->bounds); free(c->kk); c->bounds = NULL; c->kk = NULL; }

static uint8_t k31i_clip8(int v) {
    v >>= K31I_PRECISION;   /* arithmetic shift, as Pillow's lookup index */
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}

/* in [ih][iw][4] -> out [ih][ow][4], channels 0..nch-1 (the rest copied from the input's nearest column is never
   used: callers pass nch 3 or 4) */
static void k31i_horizontal(uint8_t *out, const uint8_t *in, int iw, int ih, int ow, int nch, const k31i_coeffs *c) {
    for (int yy = 0; yy < ih; yy++) {
        const uint8_t *line = in + (size_t)yy * iw * 4;
        for (int xx = 0; xx < ow; xx++) {
            const int xmin = c->bounds[xx * 2], xmax = c->bounds[xx * 2 + 1];
            const int32_t *k = &c->kk[(size_t)xx * c->ksize];
            int ss[4] = {1 << (K31I_PRECISION - 1), 1 << (K31I_PRECISION - 1), 1 << (K31I_PRECISION - 1), 1 << (K31I_PRECISION - 1)};
            for (int x = 0; x < xmax; x++) {
                const uint8_t *p = line + (size_t)(x + xmin) * 4;
                for (int ch = 0; ch < nch; ch++) ss[ch] += p[ch] * k[x];
            }
            uint8_t *o = out + ((size_t)yy * ow + xx) * 4;
            for (int ch = 0; ch < nch; ch++) o[ch] = k31i_clip8(ss[ch]);
            if (nch == 3) o[3] = 255;
        }
    }
}

static void k31i_vertical(uint8_t *out, const uint8_t *in, int w, int oh, int nch, const k31i_coeffs *c) {
    for (int yy = 0; yy < oh; yy++) {
        const int ymin = c->bounds[yy * 2], ymax = c->bounds[yy * 2 + 1];
        const int32_t *k = &c->kk[(size_t)yy * c->ksize];
        for (int xx = 0; xx < w; xx++) {
            int ss[4] = {1 << (K31I_PRECISION - 1), 1 << (K31I_PRECISION - 1), 1 << (K31I_PRECISION - 1), 1 << (K31I_PRECISION - 1)};
            for (int y = 0; y < ymax; y++) {
                const uint8_t *p = in + ((size_t)(y + ymin) * w + xx) * 4;
                for (int ch = 0; ch < nch; ch++) ss[ch] += p[ch] * k[y];
            }
            uint8_t *o = out + ((size_t)yy * w + xx) * 4;
            for (int ch = 0; ch < nch; ch++) o[ch] = k31i_clip8(ss[ch]);
            if (nch == 3) o[3] = 255;
        }
    }
}

/* Image.resize((ow, oh), BICUBIC) on [ih][iw][4]; nch 4 for RGBa (premultiplied), 3 for RGB. Pillow runs the
   horizontal pass first unless the height shrinks by more than twice what the width does. */
static uint8_t *k31i_resize(const uint8_t *in, int iw, int ih, int ow, int oh, int nch) {
    const int need_h = ow != iw, need_v = oh != ih;
    const int horizontal_first = !((ih - oh) > 0 && (ih - oh) > (iw - ow) * 2);
    uint8_t *cur = malloc((size_t)iw * ih * 4);
    if (!cur) return NULL;
    memcpy(cur, in, (size_t)iw * ih * 4);
    int cw = iw, ch = ih;
    for (int pass = 0; pass < 2; pass++) {
        const int do_h = horizontal_first ? pass == 0 : pass == 1;
        if (do_h && need_h) {
            k31i_coeffs c;
            uint8_t *next = malloc((size_t)ow * ch * 4);
            if (!next || !k31i_precompute(iw, ow, &c)) { free(next); free(cur); return NULL; }
            k31i_horizontal(next, cur, cw, ch, ow, nch, &c);
            k31i_coeffs_free(&c);
            free(cur); cur = next; cw = ow;
        } else if (!do_h && need_v) {
            k31i_coeffs c;
            uint8_t *next = malloc((size_t)cw * oh * 4);
            if (!next || !k31i_precompute(ih, oh, &c)) { free(next); free(cur); return NULL; }
            k31i_vertical(next, cur, cw, oh, nch, &c);
            k31i_coeffs_free(&c);
            free(cur); cur = next; ch = oh;
        }
    }
    return cur;
}

/* ---- the preprocessing ---- */
int ds4_k31_image_preprocess(ds4_k31_image_patches *out, const ds4_k31_image *image, char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!image || !image->rgba || !image->width || !image->height) return k31i_error(error, error_cap, "no image");
    const uint32_t w = image->width, h = image->height;
    /* navit_resize_image */
    const double s1 = sqrt(K31I_LIMIT / (fmax(1.0, (double)(w / K31I_PATCH)) * fmax(1.0, (double)(h / K31I_PATCH))));
    const double s2 = (double)(K31I_ONE_SIDE * K31I_PATCH) / (double)w, s3 = (double)(K31I_ONE_SIDE * K31I_PATCH) / (double)h;
    double scale = 1.0;
    if (s1 < scale) scale = s1;
    if (s2 < scale) scale = s2;
    if (s3 < scale) scale = s3;
    long nw = (long)((double)w * scale), nh = (long)((double)h * scale);
    if (nw < 1) nw = 1;
    if (nh < 1) nh = 1;
    if (nw > K31I_ONE_SIDE * K31I_PATCH) nw = K31I_ONE_SIDE * K31I_PATCH;
    if (nh > K31I_ONE_SIDE * K31I_PATCH) nh = K31I_ONE_SIDE * K31I_PATCH;
    const long pw = (K31I_FACTOR - nw % K31I_FACTOR) % K31I_FACTOR, ph = (K31I_FACTOR - nh % K31I_FACTOR) % K31I_FACTOR;
    const uint32_t W = (uint32_t)(nw + pw), H = (uint32_t)(nh + ph);
    /* Image.resize in the image's own mode: RGBA through premultiplied RGBa (Convert.c rgbA2rgba / rgba2rgbA) */
    uint8_t *px;
    if (image->has_alpha) {
        uint8_t *pre = malloc((size_t)w * h * 4);
        if (!pre) return k31i_error(error, error_cap, "out of memory");
        for (size_t i = 0; i < (size_t)w * h; i++) {
            const uint8_t *s = image->rgba + i * 4;
            uint8_t *d = pre + i * 4;
            const unsigned a = s[3];
            for (int ch = 0; ch < 3; ch++) {
                const unsigned tmp = s[ch] * a + 128;
                d[ch] = (uint8_t)(((tmp >> 8) + tmp) >> 8);
            }
            d[3] = (uint8_t)a;
        }
        px = k31i_resize(pre, (int)w, (int)h, (int)nw, (int)nh, 4);
        free(pre);
        if (!px) return k31i_error(error, error_cap, "out of memory");
        for (size_t i = 0; i < (size_t)nw * nh; i++) {
            uint8_t *d = px + i * 4;
            const unsigned a = d[3];
            if (a != 255 && a != 0)
                for (int ch = 0; ch < 3; ch++) {
                    const int v = (int)(255 * d[ch]) / (int)a;
                    d[ch] = (uint8_t)(v <= 0 ? 0 : v < 256 ? v : 255);
                }
        }
        /* fill_transparent_bg_with, chessboard after the resize: float32 blend, truncated as numpy's astype(uint8) */
        for (long y = 0; y < nh; y++)
            for (long x = 0; x < nw; x++) {
                uint8_t *d = px + ((size_t)y * nw + x) * 4;
                const float bg = ((y / 8 + x / 8) % 2 == 1) ? 180.0f : 255.0f;
                const float alpha = (float)d[3] / 255.0f;
                const float inv = 1.0f - alpha;
                for (int ch = 0; ch < 3; ch++) {
                    const float fg = alpha * (float)d[ch];
                    const float back = inv * bg;
                    d[ch] = (uint8_t)(fg + back);
                }
            }
    } else {
        px = k31i_resize(image->rgba, (int)w, (int)h, (int)nw, (int)nh, 3);
        if (!px) return k31i_error(error, error_cap, "out of memory");
    }
    /* zero padding, (v / 255 - 0.5) * 2 rounded as numpy does it, 14x14 patches (channel, row, column) */
    const uint32_t gw = W / K31I_PATCH, gh = H / K31I_PATCH;
    out->patches = malloc((size_t)gw * gh * 3u * K31I_PATCH * K31I_PATCH * sizeof(float));
    if (!out->patches) { free(px); return k31i_error(error, error_cap, "out of memory"); }
    float lut[256];
    for (int v = 0; v < 256; v++) {
        float f = (float)((double)v / 255.0);
        f = (float)((double)f - 0.5);
        lut[v] = (float)((double)f * 2.0);
    }
    for (uint32_t py = 0; py < gh; py++)
        for (uint32_t pxi = 0; pxi < gw; pxi++) {
            float *patch = out->patches + ((size_t)py * gw + pxi) * 588u;
            for (int ch = 0; ch < 3; ch++)
                for (int r = 0; r < K31I_PATCH; r++)
                    for (int q = 0; q < K31I_PATCH; q++) {
                        const uint32_t y = py * K31I_PATCH + (uint32_t)r, x = pxi * K31I_PATCH + (uint32_t)q;
                        const uint8_t v = (y < (uint32_t)nh && x < (uint32_t)nw) ? px[((size_t)y * nw + x) * 4 + ch] : 0;
                        patch[ch * 196 + r * K31I_PATCH + q] = lut[v];
                    }
        }
    free(px);
    out->width = w; out->height = h;
    out->resized_width = (uint32_t)nw; out->resized_height = (uint32_t)nh;
    out->grid_width = gw; out->grid_height = gh;
    out->patch_count = gw * gh;
    out->token_count = (gh / 2u) * (gw / 2u);
    return 1;
}

void ds4_k31_image_patches_free(ds4_k31_image_patches *patches) {
    if (!patches) return;
    free(patches->patches);
    memset(patches, 0, sizeof(*patches));
}
