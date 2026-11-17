/* ds4_k31_image.c against k31_vision_ref.py (17 Nov 2026): decode and preprocess an image file, write the patches as
 * raw float32 (compare byte for byte with the reference's PREFIX.patches.f32).
 *   cc -O2 -std=gnu11 -ffp-contract=off -o k31_image_test tools/kimi-k3/k31_image_test.c ds4_k31_image.c -lm
 *   k31_image_test IMAGE OUT.f32 */
#define JPEG_IMPLEMENTATION
#define PNG_IMPLEMENTATION
#include "../../third_party/iris/jpeg.h"
#include "../../third_party/iris/png.h"
#include "../../ds4_k31_image.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s IMAGE OUT.f32\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read failed\n"); return 1; }
    fclose(f);
    char err[256] = {0};
    ds4_k31_image im;
    ds4_k31_image_patches p;
    if (!ds4_k31_image_decode(&im, buf, (size_t)n, err, sizeof(err)) || !ds4_k31_image_preprocess(&p, &im, err, sizeof(err))) {
        fprintf(stderr, "%s: %s\n", argv[1], err);
        return 1;
    }
    f = fopen(argv[2], "wb");
    fwrite(p.patches, sizeof(float), (size_t)p.patch_count * 588u, f);
    fclose(f);
    printf("%s: %ux%u%s -> resized %ux%u, grid %ux%u, %u patches, %u image tokens\n", argv[1], p.width, p.height,
           im.has_alpha ? " (alpha)" : "", p.resized_width, p.resized_height, p.grid_height, p.grid_width, p.patch_count, p.token_count);
    ds4_k31_image_patches_free(&p);
    ds4_k31_image_free(&im);
    free(buf);
    return 0;
}
