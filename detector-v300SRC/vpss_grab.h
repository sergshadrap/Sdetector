#ifndef VPSS_GRAB_H
#define VPSS_GRAB_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t phy_y;     /* physical address of the Y plane (NV21: Y then interleaved VU) */
    uint64_t phy_uv;    /* physical address of the VU plane */
    uint32_t width, height, stride;
} VGrabFrame;

/* size of a VPSS channel; returns 0 or an SDK error code */
int  vgrab_chn_size(int grp, int chn, int *w, int *h);

/* slot 0 or 1: sets the channel's user depth to 1 so frames can be taken */
int  vgrab_open(int slot, int grp, int chn);

/* waits (timeout_ms, 0 = do not wait) for a frame and holds it until vgrab_release(slot).
 * A frame still held in that slot is released first. Returns 0 or an SDK error code
 * (0xa007800e = nothing there / timeout). */
int  vgrab_get(int slot, VGrabFrame *f, int timeout_ms);

/* releases the frame held in the slot (does nothing if none) */
void vgrab_release(int slot);

/* releases held frames and restores the original depth of every opened channel */
void vgrab_close(void);

#ifdef __cplusplus
}
#endif
#endif