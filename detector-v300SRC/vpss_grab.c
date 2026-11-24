/* vpss_grab.c - takes frames from majestic VPSS channels (no JPEG, no copy).
 * Build as C with the Goke headers (-I./include); do not include it in a file that uses
 * the hi_*.h headers. */
#include <stdio.h>
#include <string.h>
#include "type.h"
#include "common.h"
#include "comm_video.h"
#include "comm_vpss.h"
#include "gk_api_vpss.h"
#include "vpss_grab.h"

#define NSLOT 2

static struct {
    int grp, chn, open, held, warned;
    GK_U32 depth0;
    VIDEO_FRAME_INFO_S f;
} g[NSLOT];

int vgrab_chn_size(int grp, int chn, int *w, int *h)
{
    VPSS_CHN_ATTR_S a;
    GK_S32 r = GK_API_VPSS_GetChnAttr(grp, chn, &a);
    if (r == 0) { *w = (int)a.u32Width; *h = (int)a.u32Height; }
    return (int)r;
}

int vgrab_open(int slot, int grp, int chn)
{
    VPSS_CHN_ATTR_S a;
    GK_S32 r;
    if (slot < 0 || slot >= NSLOT) return -1;
    g[slot].grp = grp;
    g[slot].chn = chn;
    r = GK_API_VPSS_GetChnAttr(grp, chn, &a);
    if (r != 0) return (int)r;
    g[slot].depth0 = a.u32Depth;
    if (a.u32Depth < 1) a.u32Depth = 1;
    r = GK_API_VPSS_SetChnAttr(grp, chn, &a);
    if (r == 0) g[slot].open = 1;
    return (int)r;
}

void vgrab_release(int slot)
{
    if (slot < 0 || slot >= NSLOT) return;
    if (g[slot].held) {
        GK_API_VPSS_ReleaseChnFrame(g[slot].grp, g[slot].chn, &g[slot].f);
        g[slot].held = 0;
    }
}

int vgrab_get(int slot, VGrabFrame *o, int timeout_ms)
{
    GK_S32 r;
    if (slot < 0 || slot >= NSLOT || !g[slot].open) return -1;
    vgrab_release(slot);
    r = GK_API_VPSS_GetChnFrame(g[slot].grp, g[slot].chn, &g[slot].f, timeout_ms);
    if (r != 0 && timeout_ms > 0) {
        /* Nothing arrived. One reason is that majestic re-created the channel, which resets
         * the user depth to 0 and makes every later get fail with "buffer empty". Look, and
         * put the depth back if that happened. */
        VPSS_CHN_ATTR_S a;
        if (GK_API_VPSS_GetChnAttr(g[slot].grp, g[slot].chn, &a) == 0) {
            if (!g[slot].warned) {
                fprintf(stderr, "vgrab: grp%d ch%d get failed 0x%x, channel depth is %u\n",
                        g[slot].grp, g[slot].chn, (unsigned)r, a.u32Depth);
                g[slot].warned = 1;
            }
            if (a.u32Depth < 1) {
                fprintf(stderr, "vgrab: depth of grp%d ch%d was reset to 0, setting it again\n",
                        g[slot].grp, g[slot].chn);
                a.u32Depth = 1;
                if (GK_API_VPSS_SetChnAttr(g[slot].grp, g[slot].chn, &a) == 0)
                    r = GK_API_VPSS_GetChnFrame(g[slot].grp, g[slot].chn, &g[slot].f, timeout_ms);
            }
        }
    }
    if (r != 0) return (int)r;
    g[slot].held = 1;
    o->phy_y  = g[slot].f.stVFrame.u64PhyAddr[0];
    o->phy_uv = g[slot].f.stVFrame.u64PhyAddr[1];
    o->width  = g[slot].f.stVFrame.u32Width;
    o->height = g[slot].f.stVFrame.u32Height;
    o->stride = g[slot].f.stVFrame.u32Stride[0];
    return 0;
}

void vgrab_close(void)
{
    for (int s = 0; s < NSLOT; s++) {
        VPSS_CHN_ATTR_S a;
        if (!g[s].open) continue;
        vgrab_release(s);
        if (GK_API_VPSS_GetChnAttr(g[s].grp, g[s].chn, &a) == 0) {
            a.u32Depth = g[s].depth0;
            GK_API_VPSS_SetChnAttr(g[s].grp, g[s].chn, &a);
        }
        g[s].open = 0;
    }
}