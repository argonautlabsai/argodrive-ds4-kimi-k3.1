/* Kimi K3.1 image input (17 Nov 2026): decoding and Moonshot's KimiK3VisionProcessor preprocessing, pixel for pixel.
 *
 * The reference is the Python path the serving stacks run (Moonshot's media_utils.py with Pillow): the image as decoded,
 * without EXIF rotation; navit resize (at most 65,536 patches of 14x14 and 512 a side, scaled down only); Pillow's
 * bicubic resample (separable, 22-bit fixed point, premultiplied alpha for RGBA); transparent pixels blended over
 * Moonshot's chessboard after the resize (float32, truncated); zero padding to a multiple of 28; (v / 255 - 0.5) * 2;
 * 14x14 patches as (channel, row, column). Compiled without fast math (see the Makefile) so the floats match numpy's.
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
    int has_alpha;            /* the file carries an alpha channel (PNG gray+alpha or RGBA) */
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
