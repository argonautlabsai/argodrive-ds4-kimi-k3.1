/* Kimi K3.1 image input (17 Nov 2026): decoding and Moonshot's KimiK3VisionProcessor preprocessing, pixel for pixel.
 *
 * The reference is the checkpoint's own processor (Moonshot's kimi_k3_vision_processing.py and media_utils.py with
 * Pillow): the image as decoded, without EXIF rotation; navit resize (at most 65,536 patches of 14x14 and 512 a side,
 * scaled down only) in the image's own mode: Pillow's bicubic resample (separable, 22-bit fixed point, premultiplied
 * alpha for RGBA and LA), nearest neighbour for palette images; transparency (alpha, a palette's tRNS, a gray image's
 * colour key; an RGB image's colour key is ignored, as media_utils does) blended over Moonshot's chessboard after the
 * resize (float32, truncated); zero padding to a multiple of 28; (v / 255 - 0.5) * 2; 14x14 patches as (channel, row,
 * column). Compiled without fast math (see the Makefile) so the floats match numpy's.
 *
 * Two switches: DS4_K31_PILLOW=12 resizes as Pillow 12 does (horizontal pass first, vertical first only for images
 * over 100 times taller than wide), the default as Pillow 13 (vertical first when the height shrinks by more than
 * twice what the width does). DS4_K31_IMAGE_COMPAT=vllm preprocesses as vLLM's fused Kimi processor: every image
 * converted to RGB on load (alpha dropped, no chessboard), Pillow 13. Plain RGB images (JPEG, RGB PNG) come out the
 * same in every mode unless the pass order matters.
 * tools/kimi-k3/k31_vision_ref.py is the reference; tools/kimi-k3/k31_image_test.c compares the two. */
#ifndef DS4_K31_IMAGE_H
#define DS4_K31_IMAGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t width, height;
    uint8_t *rgba;            /* width x height x 4 */
    int has_alpha;            /* an alpha channel: PNG gray+alpha, RGBA, or a palette with tRNS */
    int palette;              /* a PNG palette image: Pillow resizes it nearest neighbour */
    int has_key;              /* a gray PNG with a tRNS colour key: pixels of that value after the resize are transparent */
    uint16_t key;
} ds4_k31_image;

typedef struct {
    uint32_t width, height;                   /* as decoded: the prompt says "image WxH" */
    uint32_t resized_width, resized_height;   /* before the padding */
    uint32_t grid_width, grid_height;         /* patches */
    uint32_t patch_count, token_count;        /* token_count = (grid_height / 2) * (grid_width / 2) */
    float *patches;                           /* patch_count x 588 */
} ds4_k31_image_patches;

int  ds4_k31_image_decode(ds4_k31_image *out, const uint8_t *encoded, size_t len, char *error, size_t error_cap);
void ds4_k31_image_free(ds4_k31_image *image);
int  ds4_k31_image_preprocess(ds4_k31_image_patches *out, const ds4_k31_image *image, char *error, size_t error_cap);
void ds4_k31_image_patches_free(ds4_k31_image_patches *patches);

#ifdef __cplusplus
}
#endif
#endif
