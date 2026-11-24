/*
 * sdetector.cpp - motion (IVE) -> hardware crop (IVE DMA) -> YOLO26n (ncnn) -> evidence pictures,
 *                 all in one process, for Goke/HiSilicon cameras running OpenIPC + Majestic (GK7205V300).
 *
 *   VPSS grp0 ch1 (640x480)  -> IVE SAD + CCL motion regions inside the secured area
 *   VPSS grp0 ch0 (full res) -> IVE DMA cuts a window around the target
 *   detector thread          -> YOLO on the (enlarged or reduced) window; K of the last N results
 *                               positive = intrusion -> JPEG with the boxes in --out + one EVENT line
 *
 * Also: night mode from Majestic (/metrics/night) with an optional second pass; a frame-rate guard
 * (video1.fps from /etc/majestic.yaml); a MemAvailable guard; --scene/--range presets; only one
 * instance may run (lock /tmp/sdetector.lock).
 *
 * Run it with majestic running. See usage() for the options.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <signal.h>
#include <pthread.h>
#include <vector>
#include <algorithm>

#include "hi_type.h"
#include "hi_common.h"
#include "hi_comm_ive.h"
#include "hi_ive.h"
#include "mpi_sys.h"
#include "mpi_ive.h"

#include <jpeglib.h>
#include "vpss_grab.h"
#include "yolo_infer.h"

/* shims for the vendor libraries, which were built for uClibc, on musl */
int memcpy_s(void *d, size_t dn, const void *s, size_t n) { memcpy(d, s, n < dn ? n : dn); return 0; }
int memset_s(void *d, size_t dn, int c, size_t n) { memset(d, c, n < dn ? n : dn); return 0; }
int memmove_s(void *d, size_t dn, const void *s, size_t n) { memmove(d, s, n < dn ? n : dn); return 0; }
int strncpy_s(char *d, size_t dn, const char *s, size_t n) { strncpy(d, s, n < dn ? n : dn); return 0; }
int snprintf_s(char *d, size_t dn, size_t n, const char *f, ...) {
    va_list a; va_start(a, f); int r = vsnprintf(d, dn, f, a); va_end(a); return r;
}
const unsigned short int *__ctype_b;
size_t _stdlib_mb_cur_max(void) { return 0; }
int __fputc_unlocked(int c, FILE *stream) { return fputc(c, stream); }
int __fgetc_unlocked(FILE *stream) { return fgetc(stream); }

#define BLOCK 4
#define MAXGAP 8                 /* the largest --motion-gap */

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int sig) { (void)sig; g_stop = 1; }

/* describe a VPSS frame's Y plane as an IVE gray image (no copy, physical address only) */
static void wrap_img(IVE_IMAGE_S *img, const VGrabFrame *f)
{
    memset(img, 0, sizeof(*img));
    img->enType = IVE_IMAGE_TYPE_U8C1;
    img->u32Width = f->width;
    img->u32Height = f->height;
    img->au32Stride[0] = f->stride;
    img->au64PhyAddr[0] = f->phy_y;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* ---------------------------------------------------------------- IVE helpers */

static HI_S32 ive_wait(IVE_HANDLE h)
{
    HI_BOOL fin = HI_FALSE;
    HI_S32 ret;
    double t0 = now_ms();
    do {
        ret = HI_MPI_IVE_Query(h, &fin, HI_TRUE);
        if (ret == HI_ERR_IVE_QUERY_TIMEOUT && now_ms() - t0 > 2000.0) {     /* never wait for ever */
            fprintf(stderr, "IVE: handle %d not finished after 2 s, giving up\n", (int)h);
            return HI_FAILURE;
        }
    } while (ret == HI_ERR_IVE_QUERY_TIMEOUT);
    return ret;
}

/* hardware copy of a VPSS Y plane into an IVE image (IVE DMA, no CPU) */
static HI_S32 dma_copy(const VGrabFrame *f, const IVE_IMAGE_S *dst)
{
    IVE_DATA_S s, d;
    IVE_DMA_CTRL_S c;
    IVE_HANDLE h;
    memset(&s, 0, sizeof(s));
    memset(&d, 0, sizeof(d));
    memset(&c, 0, sizeof(c));
    s.u64PhyAddr = f->phy_y;
    s.u32Stride = f->stride;
    s.u32Width = f->width;
    s.u32Height = f->height;
    d.u64PhyAddr = dst->au64PhyAddr[0];
    d.u64VirAddr = dst->au64VirAddr[0];
    d.u32Stride = dst->au32Stride[0];
    d.u32Width = f->width;
    d.u32Height = f->height;
    c.enMode = IVE_DMA_MODE_DIRECT_COPY;
    HI_S32 r = HI_MPI_IVE_DMA(&h, &s, &d, &c, HI_TRUE);
    if (r != HI_SUCCESS) { fprintf(stderr, "DMA ret=0x%x\n", r); return r; }
    return ive_wait(h);
}

/* Cut a tw x th window at (x,y) out of a full-resolution NV21 frame into dst (tight tw x th
 * NV21: Y plane then VU plane) with the IVE DMA. x must be a multiple of 16, y even. */
static HI_S32 crop_nv21(const VGrabFrame *f, int x, int y, int tw, int th, const IVE_MEM_INFO_S *dst)
{
    if (!f || !dst || tw <= 0 || th <= 0 ||
        (x & 15) != 0 || (y & 1) != 0 || (tw & 15) != 0 || (th & 1) != 0 ||
        x < 0 || y < 0 || x + tw > (int)f->width || y + th > (int)f->height) {
        fprintf(stderr, "crop_nv21: invalid window (%d,%d) %dx%d for %ux%u\n",
                x, y, tw, th, f ? f->width : 0, f ? f->height : 0);
        return HI_FAILURE;
    }

    IVE_DATA_S s, d;
    IVE_DMA_CTRL_S c;
    IVE_HANDLE h;
    HI_S32 r;
    memset(&c, 0, sizeof(c));
    c.enMode = IVE_DMA_MODE_DIRECT_COPY;

    memset(&s, 0, sizeof(s));
    memset(&d, 0, sizeof(d));
    s.u64PhyAddr = f->phy_y + (uint64_t)y * f->stride + x;
    s.u32Stride = f->stride;
    s.u32Width = tw;
    s.u32Height = th;
    d.u64PhyAddr = dst->u64PhyAddr;
    d.u64VirAddr = dst->u64VirAddr;
    d.u32Stride = tw;
    d.u32Width = tw;
    d.u32Height = th;
    r = HI_MPI_IVE_DMA(&h, &s, &d, &c, HI_TRUE);
    if (r != HI_SUCCESS) { fprintf(stderr, "crop Y DMA ret=0x%x\n", r); return r; }
    r = ive_wait(h);
    if (r != HI_SUCCESS) return r;

    s.u64PhyAddr = f->phy_uv + (uint64_t)(y / 2) * f->stride + x;
    s.u32Height = th / 2;
    d.u64PhyAddr = dst->u64PhyAddr + (uint64_t)tw * th;
    d.u64VirAddr = dst->u64VirAddr + (uint64_t)tw * th;
    d.u32Height = th / 2;
    r = HI_MPI_IVE_DMA(&h, &s, &d, &c, HI_TRUE);
    if (r != HI_SUCCESS) { fprintf(stderr, "crop VU DMA ret=0x%x\n", r); return r; }
    r = ive_wait(h);
    return r;
}


static HI_S32 alloc_img(IVE_IMAGE_S *img, HI_U32 w, HI_U32 h)
{
    memset(img, 0, sizeof(*img));
    img->enType = IVE_IMAGE_TYPE_U8C1;
    img->u32Width = w;
    img->u32Height = h;
    img->au32Stride[0] = (w + 15) & ~15;
    HI_S32 r = HI_MPI_SYS_MmzAlloc(&img->au64PhyAddr[0],
        (HI_VOID **)&img->au64VirAddr[0], NULL, HI_NULL,
        img->au32Stride[0] * h);
    if (r != HI_SUCCESS)
        fprintf(stderr, "alloc_img %ux%u failed: 0x%x\n", w, h, r);
    return r;
}

static void free_img(IVE_IMAGE_S *img)
{
    if (img->au64PhyAddr[0])
        HI_MPI_SYS_MmzFree(img->au64PhyAddr[0], (HI_VOID *)(HI_UL)img->au64VirAddr[0]);
}


struct Roi { int x0, y0, x1, y1; };
#define MAX_ROI 8

/* Reads majestic's motion area the same way mppnet does: every "- LxTxWxH" line in
 * /etc/majestic.yaml is a rectangle (left, top, width, height) in pixels of video0.size.
 * Returns the number of rectangles; ref_w and ref_h get video0.size (0 if not found). */
static int load_roi(Roi *out, int max, int *ref_w, int *ref_h)
{
    FILE *f = fopen("/etc/majestic.yaml", "r");
    char line[256];
    int in_video0 = 0, n = 0;
    *ref_w = *ref_h = 0;
    if (!f) return 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != ' ' && line[0] != '\t' && line[0] != '\n' && line[0] != '#' && line[0] != '\r') {
            in_video0 = (strncmp(line, "video0:", 7) == 0);
            continue;
        }
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (in_video0 && strncmp(p, "size:", 5) == 0) {
            int w, h;
            p += 5;
            while (*p == ' ') p++;
            if (sscanf(p, "%dx%d", &w, &h) == 2) { *ref_w = w; *ref_h = h; }
            continue;
        }
        if (*p == '-') {
            p++;
            while (*p == ' ') p++;
        }
        int rx, ry, rw, rh;
        if (sscanf(p, "%dx%dx%dx%d", &rx, &ry, &rw, &rh) == 4 && n < max) {
            out[n].x0 = rx;
            out[n].y0 = ry;
            out[n].x1 = rx + rw;
            out[n].y1 = ry + rh;
            n++;
        }
    }
    fclose(f);
    return n;
}

/* video1.fps from /etc/majestic.yaml (the second stream = the motion channel); 0 if not found */
static int load_video1_fps(void)
{
    FILE *f = fopen("/etc/majestic.yaml", "r");
    char line[256];
    int in_video1 = 0, fps = 0;
    if (!f) return 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != ' ' && line[0] != '\t' && line[0] != '\n' && line[0] != '#' && line[0] != '\r') {
            in_video1 = (strncmp(line, "video1:", 7) == 0);
            continue;
        }
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (in_video1 && strncmp(p, "fps:", 4) == 0) {
            int v = atoi(p + 4);
            if (v > 0) fps = v;
        }
    }
    fclose(f);
    return fps;
}

static int touches_roi(const Roi *r, int n, int x0, int y0, int x1, int y1)
{
    for (int i = 0; i < n; i++)
        if (x0 < r[i].x1 && x1 > r[i].x0 && y0 < r[i].y1 && y1 > r[i].y0)
            return 1;
    return 0;
}

/* Does a moving region count as "in the secured area"?
 *   mode 0 touch  - any part of the region overlaps the area
 *   mode 1 feet   - the bottom centre of the region is inside the area
 *   mode 2 inside - the whole region is inside the area (one block of slack, because the
 *                   regions are rounded outwards to 16 px blocks)                         */
#define ROI_SLACK 16
static int roi_accepts(const Roi *r, int n, int mode, int x0, int y0, int x1, int y1)
{
    if (mode == 0) return touches_roi(r, n, x0, y0, x1, y1);
    for (int i = 0; i < n; i++) {
        if (mode == 1) {
            int fx = (x0 + x1) / 2, fy = y1;
            if (fx >= r[i].x0 && fx < r[i].x1 && fy >= r[i].y0 && fy <= r[i].y1 + ROI_SLACK)
                return 1;
        } else {
            if (x0 >= r[i].x0 - ROI_SLACK && x1 <= r[i].x1 + ROI_SLACK &&
                y0 >= r[i].y0 - ROI_SLACK && y1 <= r[i].y1 + ROI_SLACK)
                return 1;
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- crop step */

struct Reg { int x0, y0, x1, y1; unsigned area; };

/* Merge the biggest region with every region that lies within `gap` pixels of the growing
 * box (repeat until nothing more joins). Far-away regions (reflections, light changes) are
 * left out. Returns the merged box and the summed area in blocks. */
static void pick_cluster(const Reg *r, int n, int gap,
                         int *bx0, int *by0, int *bx1, int *by1, unsigned *total)
{
    int used[IVE_MAX_REGION_NUM];
    memset(used, 0, sizeof(used));
    int seed = 0;
    for (int i = 1; i < n; i++)
        if (r[i].area > r[seed].area) seed = i;
    used[seed] = 1;
    *bx0 = r[seed].x0; *by0 = r[seed].y0; *bx1 = r[seed].x1; *by1 = r[seed].y1;
    *total = r[seed].area;
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            if (r[i].x0 < *bx1 + gap && r[i].x1 > *bx0 - gap &&
                r[i].y0 < *by1 + gap && r[i].y1 > *by0 - gap) {
                if (r[i].x0 < *bx0) *bx0 = r[i].x0;
                if (r[i].y0 < *by0) *by0 = r[i].y0;
                if (r[i].x1 > *bx1) *bx1 = r[i].x1;
                if (r[i].y1 > *by1) *by1 = r[i].y1;
                *total += r[i].area;
                used[i] = 1;
                changed = 1;
            }
        }
    }
}


static void save_rgb_jpeg(const char *fn, const unsigned char *rgb, int w, int h)
{
    FILE *f = fopen(fn, "wb");
    if (!f) return;
    struct jpeg_compress_struct c;
    struct jpeg_error_mgr em;
    c.err = jpeg_std_error(&em);
    jpeg_create_compress(&c);
    jpeg_stdio_dest(&c, f);
    c.image_width = (JDIMENSION)w;
    c.image_height = (JDIMENSION)h;
    c.input_components = 3;
    c.in_color_space = JCS_RGB;
    jpeg_set_defaults(&c);
    jpeg_set_quality(&c, 90, TRUE);
    jpeg_start_compress(&c, TRUE);
    while (c.next_scanline < c.image_height) {
        JSAMPROW rp = (JSAMPROW)(rgb + (size_t)c.next_scanline * w * 3);
        jpeg_write_scanlines(&c, &rp, 1);
    }
    jpeg_finish_compress(&c);
    jpeg_destroy_compress(&c);
    fclose(f);
}


/* ================================================================ helpers for the release */

static void logf(const char *fmt, ...)
{
    char ts[16];
    time_t t = time(NULL);
    struct tm tmv;
    localtime_r(&t, &tmv);
    strftime(ts, sizeof(ts), "%H:%M:%S", &tmv);
    printf("[%s] ", ts);
    va_list a;
    va_start(a, fmt);
    vprintf(fmt, a);
    va_end(a);
    printf("\n");

    /* stdout redirected to a file in RAM (/tmp): when it passes 256 KB start again from the top */
    struct stat lst;
    if (fstat(1, &lst) == 0 && S_ISREG(lst.st_mode) && lst.st_size > 256 * 1024) {
        fflush(stdout);
        if (ftruncate(1, 0) == 0) lseek(1, 0, SEEK_SET);
    }
}

static long kb_of(const char *file, const char *key)
{
    FILE *f = fopen(file, "r");
    if (!f) return -1;
    char line[256];
    size_t kl = strlen(key);
    long v = -1;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, key, kl) == 0) { v = atol(line + kl); break; }
    fclose(f);
    return v;
}

/* a window (sx,sy,sw,sh) of an NV21 image (w x h) to packed RGB, reduced by the integer
 * factor k with box averaging (k = 1: plain conversion). Output is (sw/k) x (sh/k).
 * NV21: V first, full-range BT.601. */
static void nv21_window_to_rgb(const uint8_t *nv, int w, int h, int sx, int sy, int sw, int sh,
                               int k, uint8_t *rgb)
{
    const uint8_t *Y = nv, *VU = nv + (size_t)w * h;
    int ow = sw / k, oh = sh / k;
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            int x0 = sx + ox * k, y0 = sy + oy * k;
            int yy;
            if (k == 1) {
                yy = Y[(size_t)y0 * w + x0];
            } else {
                int sum = 0;
                for (int yi = 0; yi < k; yi++) {
                    const uint8_t *row = Y + (size_t)(y0 + yi) * w + x0;
                    for (int xi = 0; xi < k; xi++) sum += row[xi];
                }
                yy = sum / (k * k);
            }
            int cx = (x0 + k / 2) & ~1, cy = (y0 + k / 2) / 2;
            const uint8_t *vu = VU + (size_t)cy * w;
            int v = vu[cx] - 128, u = vu[cx + 1] - 128;
            int r = yy + ((1436 * v) >> 10);
            int g = yy - ((352 * u + 731 * v) >> 10);
            int b = yy + ((1815 * u) >> 10);
            uint8_t *o = rgb + ((size_t)oy * ow + ox) * 3;
            o[0] = r < 0 ? 0 : r > 255 ? 255 : r;
            o[1] = g < 0 ? 0 : g > 255 ? 255 : g;
            o[2] = b < 0 ? 0 : b > 255 ? 255 : b;
        }
    }
}

static void draw_rect_rgb(unsigned char *img, int w, int h, int x1, int y1, int x2, int y2, int th)
{
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > w - 1) x2 = w - 1;
    if (y2 > h - 1) y2 = h - 1;
    if (x2 <= x1 || y2 <= y1) return;
    for (int t = 0; t < th; t++) {
        for (int x = x1; x <= x2; x++) {
            int ya = y1 + t, yb = y2 - t;
            if (ya <= y2) { unsigned char *p = img + ((size_t)ya * w + x) * 3; p[0] = 255; p[1] = 255; p[2] = 0; }
            if (yb >= y1) { unsigned char *p = img + ((size_t)yb * w + x) * 3; p[0] = 255; p[1] = 255; p[2] = 0; }
        }
        for (int y = y1; y <= y2; y++) {
            int xa = x1 + t, xb = x2 - t;
            if (xa <= x2) { unsigned char *p = img + ((size_t)y * w + xa) * 3; p[0] = 255; p[1] = 255; p[2] = 0; }
            if (xb >= x1) { unsigned char *p = img + ((size_t)y * w + xb) * 3; p[0] = 255; p[1] = 255; p[2] = 0; }
        }
    }
}

/* keeps only the newest `keep` evidence pictures (intr_*.jpg) of the directory */
static void prune_dir(const char *dir, int keep)
{
    struct Ent { time_t t; char name[96]; };
    std::vector<Ent> v;
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        size_t n = strlen(e->d_name);
        if (n < 9 || strncmp(e->d_name, "intr_", 5) != 0 || strcmp(e->d_name + n - 4, ".jpg") != 0) continue;
        char p[400];
        snprintf(p, sizeof(p), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(p, &st) != 0) continue;
        Ent en;
        en.t = st.st_mtime;
        snprintf(en.name, sizeof(en.name), "%s", e->d_name);
        v.push_back(en);
    }
    closedir(d);
    if ((int)v.size() <= keep) return;
    std::sort(v.begin(), v.end(), [](const Ent &a, const Ent &b) { return a.t < b.t; });
    for (size_t i = 0; i + (size_t)keep < v.size(); i++) {
        char p[400];
        snprintf(p, sizeof(p), "%s/%s", dir, v[i].name);
        unlink(p);
    }
}

/* ================================================================ detector thread */

/* One detection job: the evidence window (ew x eh, NV21, cut out of the full-resolution
 * frame at (ex,ey)) and, inside it, the tight window (tx,ty,tw,th) that YOLO looks at. */
struct Job {
    unsigned seq;
    double t_ms;
    int ex, ey, ew, eh;
    int tx, ty, tw, th;
    std::vector<uint8_t> nv;
};

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_stats_mu = PTHREAD_MUTEX_INITIALIZER;
static Job g_job;

/* a job starts only while MemAvailable is above this (kB, 0 = off): a job takes ~9 MB for a moment, and the
 * camera stalls when MemAvailable runs down to ~6 MB (OpenIPC/firmware issue 2828) */
static long g_min_avail_kb = 15000;
static int g_gray = 0;                    /* YOLO input in grey: 0 off, 1 always, 2 at night (Majestic /metrics/night) */
static int g_src_w = 0, g_src_h = 0;      /* size of the full-resolution channel */
static int g_fps_drop = 30;               /* % below video1.fps at which the camera counts as out of normal fps (0 = off) */
static volatile int g_fps_low = 0;        /* set by the motion loop: the sensor delivers far fewer frames than configured */

enum DetectorState { DET_IDLE = 0, DET_READY, DET_RUNNING };
static DetectorState g_det_state = DET_IDLE;

struct Cfg {
    float conf;
    int vote_k, vote_n;
    double save_gap_s;
    int keep;
    const char *out;
    int classes[80];
    int dbg_max;
    int ev_max_dim;
    int night_confirm;          /* at night an intrusion needs a second, confirming pass */
};
static Cfg g_cfg;

#define NIGHT_PORT 80            /* Majestic's HTTP port on localhost */
#define NIGHT_CONFIRM_MS 4000    /* the second pass must follow the first positive within this time */

/* Night mode from Majestic: GET /metrics/night answers with a line "night_enabled 0|1".
 * The same as: curl -s http://localhost/metrics/night | awk '$1=="night_enabled" {print $2}' */
static int night_query(int port)
{
    static const char req[] = "GET /metrics/night HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    char buf[2048];
    struct sockaddr_in a;
    struct timeval tv;
    int fd, n, total = 0, v = -1;

    fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    tv.tv_sec = 0;
    tv.tv_usec = 300000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((unsigned short)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) < 0 || write(fd, req, sizeof(req) - 1) < 0) {
        close(fd);
        return -1;
    }
    while (total < (int)sizeof(buf) - 1 && (n = (int)read(fd, buf + total, sizeof(buf) - 1 - total)) > 0)
        total += n;
    close(fd);
    buf[total] = 0;

    for (char *line = buf; line && *line; ) {      /* comment lines ("# HELP night_enabled ...") do not start with the name */
        if (!strncmp(line, "night_enabled", 13) && (line[13] == ' ' || line[13] == '\t')) {
            const char *p = line + 13;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '0' || *p == '1') v = *p - '0';
            break;
        }
        line = strchr(line, '\n');
        if (line) line++;
    }
    return v;
}

/* Asked at most every 5 s, only by the detector thread. */
static int night_now(void)
{
    static double t_last = -1e9;
    static int state = 0, announced = 0, warned = 0;
    double t = now_ms();

    if (t - t_last < 5000.0) return state;
    t_last = t;

    int v = night_query(NIGHT_PORT);
    if (v < 0) {
        if (!warned) { logf("night: no answer from http://localhost:%d/metrics/night (treated as day)", NIGHT_PORT); warned = 1; }
        return state;
    }
    if (v != state || !announced) {
        logf("night mode %s (Majestic metrics/night)", v ? "ON" : "OFF");
        announced = 1;
        state = v;
    }
    return state;
}

static unsigned long g_n_analysed = 0, g_n_pos = 0, g_n_events = 0;
static double g_sum_yolo = 0, g_max_yolo = 0;

static std::vector<uint8_t> g_rgb;
static std::vector<uint8_t> g_ev;

/* Brightness inside a box (box in pixels of the YOLO window, kt = window/input scale): mean Y and the share
 * of saturated pixels (>= 250), every 3rd pixel. For judging bright specks / IR glare taken for a person. */
static void box_brightness(const Job &job, int kt, float bx1, float by1, float bx2, float by2, int *mean, int *sat)
{
    *mean = *sat = 0;
    int x0 = job.tx + (int)bx1 * kt, y0 = job.ty + (int)by1 * kt, x1 = job.tx + (int)bx2 * kt, y1 = job.ty + (int)by2 * kt;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > job.ew) x1 = job.ew;
    if (y1 > job.eh) y1 = job.eh;
    long sum = 0;
    int cnt = 0, ns = 0;
    for (int y = y0; y < y1; y += 3) {
        const uint8_t *row = job.nv.data() + (size_t)y * job.ew;
        for (int x = x0; x < x1; x += 3) {
            int v = row[x];
            sum += v;
            cnt++;
            if (v >= 250) ns++;
        }
    }
    if (cnt > 0) { *mean = (int)(sum / cnt); *sat = ns * 100 / cnt; }
}

static void process_job(Job &job)
{
    int kt = job.tw / 192;
    { int b = job.th / 256; if (b < kt) kt = b; }
    if (kt < 1) kt = 1;
    int iw = job.tw / kt, ih = job.th / kt;

    int night = night_now();
    g_rgb.resize((size_t)iw * ih * 3);
    nv21_window_to_rgb(job.nv.data(), job.ew, job.eh, job.tx, job.ty,
                       iw * kt, ih * kt, kt, g_rgb.data());
    if (g_gray == 1 || (g_gray == 2 && night)) {      /* IR pictures are purple: give YOLO the brightness only */
        uint8_t *q = g_rgb.data();
        for (size_t i = 0, n = (size_t)iw * ih; i < n; i++, q += 3) {
            uint8_t l = (uint8_t)((q[0] * 77 + q[1] * 150 + q[2] * 29) >> 8);
            q[0] = q[1] = q[2] = l;
        }
    }

    YoloDet dets[20];
    float pbest = 0.f;
    double yms = 0.0;
    double infer_start = now_ms();
    int n = yolo_run(g_rgb.data(), iw, ih, dets, 20, &pbest, &yms);
    if (n < 0) { logf("#%u YOLO failed", job.seq); return; }

    pthread_mutex_lock(&g_stats_mu);
    g_n_analysed++;
    g_sum_yolo += yms;
    if (yms > g_max_yolo) g_max_yolo = yms;
    pthread_mutex_unlock(&g_stats_mu);

    int vk = g_cfg.vote_k, vn = g_cfg.vote_n;
    int second_pass = night && g_cfg.night_confirm && !g_fps_low;     /* not while the camera is out of its normal fps */
    if (second_pass) {      /* the second pass: the last analyses must ALL be positive */
        if (vk < 2) vk = 2;
        vn = vk;
    }

    int pos[20], npos = 0;
    float best = 0.f;
    char txt[512];
    int tl = 0;
    txt[0] = 0;
    for (int k = 0; k < n; k++) {
        int c = dets[k].cls;
        int is_pos = (c >= 0 && c < 80 && g_cfg.classes[c] && dets[k].conf >= g_cfg.conf);
        if (is_pos) {
            pos[npos++] = k;
            if (dets[k].conf > best) best = dets[k].conf;
        }
        int bm = 0, bs = 0;
        if (is_pos) box_brightness(job, kt, dets[k].x1, dets[k].y1, dets[k].x2, dets[k].y2, &bm, &bs);
        if (tl < (int)sizeof(txt) - 100) {
            tl += snprintf(txt + tl, sizeof(txt) - tl, " %s%s %.2f (%d,%d)-(%d,%d)", is_pos ? "*" : "", yolo_class_name(c), dets[k].conf,
                           job.ex + job.tx + (int)dets[k].x1 * kt, job.ey + job.ty + (int)dets[k].y1 * kt,
                           job.ex + job.tx + (int)dets[k].x2 * kt, job.ey + job.ty + (int)dets[k].y2 * kt);
            if (is_pos) tl += snprintf(txt + tl, sizeof(txt) - tl, " Y=%d sat=%d%%", bm, bs);
            tl += snprintf(txt + tl, sizeof(txt) - tl, ";");
        }
    }
    int positive = npos > 0;
    if (positive) {
        pthread_mutex_lock(&g_stats_mu);
        g_n_pos++;
        pthread_mutex_unlock(&g_stats_mu);
    }

    /* vote: at least vote_k positives among the last vote_n results (not older than 10 s) */
    static double vt[8];
    static int vp[8];
    static int vc = 0;
    if (vc == 8) { memmove(vt, vt + 1, 7 * sizeof(double)); memmove(vp, vp + 1, 7 * sizeof(int)); vc = 7; }
    vt[vc] = infer_start;
    vp[vc] = positive;
    vc++;
    int votes = 0, seen = 0;
    if (second_pass) {
        /* second pass: the latest results must ALL be positive, each following the previous one
         * within NIGHT_CONFIRM_MS (a flicker that is gone in a few seconds does not pass) */
        for (int i = vc - 1; i >= 0; i--) {
            if (!vp[i]) break;
            if (i < vc - 1 && vt[i + 1] - vt[i] > NIGHT_CONFIRM_MS) break;
            votes++;
        }
        seen = votes;
    } else {
        for (int i = vc - 1; i >= 0 && seen < vn; i--) {
            if (infer_start - vt[i] > 10000.0) break;
            seen++;
            votes += vp[i];
        }
    }

    logf("#%u win=(%d,%d) %dx%d%s yolo=%.0fms queue=%.0fms:%s%s", job.seq, job.ex + job.tx, job.ey + job.ty, job.tw, job.th,
         kt > 1 ? " (reduced)" : (job.tw < 192 ? " (enlarged)" : ""), yms, infer_start - job.t_ms, n ? txt : " nothing",
         positive ? (votes >= vk ? "  -> INTRUSION" : "  -> positive") : "");

    static int dbg_saved = 0;
    if (dbg_saved < g_cfg.dbg_max) {
        char fn[400];
        snprintf(fn, sizeof(fn), "%s/dbg_%04u.jpg", g_cfg.out, job.seq);
        save_rgb_jpeg(fn, g_rgb.data(), iw, ih);
        dbg_saved++;
    }

    /* A picture is saved (and mailed) at once when the best box lies inside the window. If it touches the
     * window edge (the person is cut off: at night only the bright part of the person moves for the motion
     * stage), the next job is waited for, for 5 s at most, then the cut picture is saved anyway. */
    static double last_save = -1e12;
    static double t_cut = -1;
    int defer = 0;
    if (positive && votes >= vk) {
        int bi = pos[0];
        for (int j = 1; j < npos; j++)
            if (dets[pos[j]].conf > dets[bi].conf) bi = pos[j];
        const YoloDet &b = dets[bi];
        int wx = job.ex + job.tx, wy = job.ey + job.ty, tol = 3;
        int cut = (b.x1 <= tol && wx > 0) || (b.y1 <= tol && wy > 0) ||
                  (b.x2 >= iw - tol && wx + job.tw < g_src_w) || (b.y2 >= ih - tol && wy + job.th < g_src_h);
        if (t_cut >= 0 && infer_start - t_cut > 15000.0) t_cut = -1;      /* left from an old episode */
        if (cut) {
            if (t_cut < 0) {
                t_cut = infer_start;
                logf("#%u the box touches the window edge: waiting up to 5 s for a complete one", job.seq);
            }
            if (infer_start - t_cut < 5000.0) defer = 1;
        }
    }
    if (positive && votes >= vk && !defer && infer_start - last_save >= g_cfg.save_gap_s * 1000.0) {
        t_cut = -1;
        int ke = 1;
        while (job.eh / ke > g_cfg.ev_max_dim) ke++;
        int ow = job.ew / ke, oh = job.eh / ke;
        g_ev.resize((size_t)ow * oh * 3);
        nv21_window_to_rgb(job.nv.data(), job.ew, job.eh, 0, 0, ow * ke, oh * ke, ke, g_ev.data());
        for (int j = 0; j < npos; j++) {
            const YoloDet &d = dets[pos[j]];
            draw_rect_rgb(g_ev.data(), ow, oh, (job.tx + (int)d.x1 * kt) / ke, (job.ty + (int)d.y1 * kt) / ke,
                          (job.tx + (int)d.x2 * kt) / ke, (job.ty + (int)d.y2 * kt) / ke, oh >= 480 ? 3 : 2);
        }
        time_t tt = time(NULL);
        struct tm tmv;
        localtime_r(&tt, &tmv);
        char fn[400];
        snprintf(fn, sizeof(fn), "%s/intr_%04d%02d%02d_%02d%02d%02d_%04u.jpg", g_cfg.out, tmv.tm_year + 1900, tmv.tm_mon + 1,
                 tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec, job.seq);
        save_rgb_jpeg(fn, g_ev.data(), ow, oh);
        prune_dir(g_cfg.out, g_cfg.keep);
        last_save = infer_start;
        pthread_mutex_lock(&g_stats_mu);
        g_n_events++;
        pthread_mutex_unlock(&g_stats_mu);
        logf("INTRUSION picture %s (%dx%d, %d object(s), best %.2f)", fn, ow, oh, npos, best);
        /* Machine-readable event, one line, stable format (for a supervising program):
         *   EVENT intrusion time=<unix seconds> n=<objects> best=<0..1> classes=<a,b> picture=<path>
         * classes: names of the positive objects, no duplicates, spaces replaced by '_'.
         * The path must not contain spaces (--out). Human log lines never start with "EVENT". */
        char cls[256];
        int cl = 0;
        cls[0] = 0;
        for (int j = 0; j < npos; j++) {
            const char *nm = yolo_class_name(dets[pos[j]].cls);
            int dup = 0;
            for (int m = 0; m < j; m++)
                if (dets[pos[m]].cls == dets[pos[j]].cls) dup = 1;
            if (dup || cl >= (int)sizeof(cls) - 40) continue;
            cl += snprintf(cls + cl, sizeof(cls) - cl, "%s%s", cl ? "," : "", nm);
        }
        for (char *q = cls; *q; q++)
            if (*q == ' ') *q = '_';
        printf("EVENT intrusion time=%ld n=%d best=%.2f classes=%s picture=%s\n", (long)tt, npos, best, cls, fn);
        fflush(stdout);
    }
}

static void *detector_thread(void *)
{
    Job job;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (g_det_state != DET_READY && !g_stop)
            pthread_cond_wait(&g_cv, &g_mu);

        if (g_stop) {
            pthread_mutex_unlock(&g_mu);
            break;
        }

        std::swap(job, g_job);
        g_det_state = DET_RUNNING;
        pthread_mutex_unlock(&g_mu);

        process_job(job);

        pthread_mutex_lock(&g_mu);
        g_det_state = DET_IDLE;
        pthread_cond_broadcast(&g_cv);
        pthread_mutex_unlock(&g_mu);
    }
    return NULL;
}

/* Choose the two windows for a target and cut the evidence window out of the next fresh
 * full-resolution frame with the IVE DMA.
 *   tight window (what YOLO sees): the target plus margin, 3:4, never smaller than min_h high
 *                                  (smaller targets are enlarged), never bigger than hmax
 *   evidence window: the tight window rounded up to a multiple of 64 high, at least ev_h high
 * Returns 0 on success. */
static int build_job(Job &jb, int cx0, int cy0, int cx1, int cy1, int fw, int fh, int margin,
                     int min_h, int hmax, int ev_h, const IVE_MEM_INFO_S *mem, unsigned seq, double *dma_ms)
{
    int bw = cx1 - cx0, bh = cy1 - cy0;
    int cxm = (cx0 + cx1) / 2, cym = (cy0 + cy1) / 2;
    double m = 1.0 + 2.0 * margin / 100.0;
    int need_h = (int)(bh * m), need_w = (int)(bw * m);
    int hy = need_h;
    if (hy * 3 < need_w * 4) hy = (need_w * 4 + 2) / 3;
    if (hy < min_h) hy = min_h;
    if (hy > hmax) hy = hmax;
    hy &= ~1;
    int wy = (hy * 3 / 4) & ~1;
    int he = (hy + 63) & ~63;
    if (he < ev_h) he = ev_h;
    if (he > hmax) he = hmax;
    int we = he * 3 / 4;

    int ex = cxm - we / 2;
    if (ex > fw - we) ex = fw - we;
    if (ex < 0) ex = 0;
    ex &= ~15;
    int ey = cym - he / 2;
    if (ey > fh - he) ey = fh - he;
    if (ey < 0) ey = 0;
    ey &= ~1;
    int tx = cxm - ex - wy / 2;
    if (tx > we - wy) tx = we - wy;
    if (tx < 0) tx = 0;
    tx &= ~1;
    int ty = cym - ey - hy / 2;
    if (ty > he - hy) ty = he - hy;
    if (ty < 0) ty = 0;
    ty &= ~1;

    double t0 = now_ms();
    VGrabFrame vf;
    if (vgrab_get(1, &vf, 0) == 0) vgrab_release(1);          /* drop an old queued frame */
    if (vgrab_get(1, &vf, 250) != 0) return -1;                /* the next fresh one */
    HI_S32 r = crop_nv21(&vf, ex, ey, we, he, mem);
    vgrab_release(1);
    if (r != HI_SUCCESS) return -2;
    size_t sz = (size_t)we * he * 3 / 2;
    HI_MPI_SYS_MmzFlushCache(mem->u64PhyAddr, (HI_VOID *)(HI_UL)mem->u64VirAddr, (HI_U32)sz);
    jb.nv.resize(sz);
    memcpy(jb.nv.data(), (const void *)(HI_UL)mem->u64VirAddr, sz);     /* cache synchronized above */
    jb.seq = seq;
    jb.t_ms = now_ms();
    jb.ex = ex; jb.ey = ey; jb.ew = we; jb.eh = he;
    jb.tx = tx; jb.ty = ty; jb.tw = wy; jb.th = hy;
    if (dma_ms) *dma_ms = jb.t_ms - t0;
    return 0;
}

/* ================================================================ main */

static void usage(void)
{
    fprintf(stderr,
        "usage: sdetector [options]    (run it with majestic running)\n"
        "  --model=PREFIX   YOLO model files PREFIX.param/.bin (default ./model/yolo26n-fp16_256_192)\n"
        "  --fp32           run ncnn without fp16\n"
        "  --out=DIR        evidence pictures go here (default /tmp/sdetector; /tmp is RAM!)\n"
        "  --keep=N         keep only the newest N pictures (default 10; /tmp is RAM, ~0.25 MB each)\n"
        "  --classes=LIST   COCO class ids that count as an intrusion, e.g. 0,2,16 (default 0 = person)\n"
        "  --conf=C         confidence needed for a positive result, 0.30 .. 0.99 (default 0.45)\n"
        "  --vote=K/N       intrusion = K positives among the last N results (default 2/3)\n"
        "  --scene=indoor|outdoor   preset: grey at night, conf 0.50 / 0.45, smallest window 450 / 128 px (outdoor untested)\n"
        "  --range=near|mid|far     smallest YOLO window 450 / 256 / 128 px (overrides the scene; explicit --min-h wins)\n"
        "  --gray=off|on|night   give YOLO a grey picture (always, or only at night); default off\n"
        "  --motion-gap=N   compare each frame with the one N frames back (default 1 = the previous one): a slow or\n"
        "                   low-contrast mover gives a bigger difference and a fuller region (1..8)\n"
        "  --fps-drop=PCT   the camera is out of normal fps when video1 delivers PCT %% less than video1.fps in\n"
        "                   /etc/majestic.yaml (default 30, 0 = off): the night second pass is skipped meanwhile\n"
        "  --min-avail=KB   start a YOLO job only while MemAvailable is at least KB (default 15000, 0 = off)\n"
        "  --night-confirm=on|off  at night (Majestic /metrics/night, read all the time) an intrusion needs a\n"
        "                   second pass right after the first positive; without it no alarm and no picture (default on)\n"
        "  --save-gap=S     seconds between two pictures of the same intrusion (default 5)\n"
        "  --sad-thr=N --min-cluster=N --gap=PX --margin=P --roi-mode=feet|inside|touch --no-roi\n"
        "  --margin=P       extra space added on EACH side of the moving target, in percent of its size (default 15, 0..100)\n"
        "  --min-h=PX       smallest window height YOLO gets (default 128; smaller targets are enlarged)\n"
        "  --ev-h=PX        evidence picture height at least (default 512; use 768 for more context)\n"
        "  --ev-max=PX      longest side of the saved picture (default 1024)\n"
        "  --debug-crops=N  also save the first N windows YOLO sees as dbg_*.jpg in --out\n"
        "  --status=S       memory/status line every S seconds (default 30, 0 = off)\n"
        "  --frames=N       stop after N motion frames (default 0 = run until killed)\n"
        "  --verbose        one line per frame, like ive_live\n");
}

int main(int argc, char **argv)
{
    int frames = 0, motion_gap = 1, sad_thr = 100, area_thr = 4, verbose = 0, use_roi = 1, roi_mode = 1;
    int margin = 15, gap = 96, min_cluster = 4, chn = 1;
    const int win_max = 1024;   /* fixed: the IVE DMA cannot cut more than ~1080 lines */
    int min_h = 128, ev_h = 512, status_s = 30, fp32 = 0;
    const char *model = "./model/yolo26n-fp16_256_192";
    memset(&g_cfg, 0, sizeof(g_cfg));
    g_cfg.conf = 0.45f;
    g_cfg.vote_k = 2;
    g_cfg.vote_n = 3;
    g_cfg.night_confirm = 1;
    g_cfg.save_gap_s = 5;
    g_cfg.keep = 10;
    g_cfg.out = "/tmp/sdetector";
    g_cfg.classes[0] = 1;
    g_cfg.ev_max_dim = 1024;

    /* presets first, so that every explicit option, wherever it stands, overrides them.
     * --scene: indoor (grey at night, conf 0.50, windows >= 450 px) or outdoor (grey at night, conf 0.45, windows >= 128 px);
     * --range: near / mid / far = smallest YOLO window 450 / 256 / 128 px. The outdoor values are a first guess, not tested. */
    for (int a = 1; a < argc; a++) {
        const char *s = argv[a];
        if (strncmp(s, "--scene=", 8) == 0) {
            if (!strcmp(s + 8, "indoor")) { g_gray = 2; g_cfg.conf = 0.50f; min_h = 450; }
            else if (!strcmp(s + 8, "outdoor")) { g_gray = 2; g_cfg.conf = 0.45f; min_h = 128; }
            else { fprintf(stderr, "--scene is indoor or outdoor\n"); usage(); return 1; }
            fprintf(stderr, "scene preset: %s\n", s + 8);
        }
    }
    for (int a = 1; a < argc; a++) {
        const char *s = argv[a];
        if (strncmp(s, "--range=", 8) == 0) {
            if (!strcmp(s + 8, "near")) min_h = 450;
            else if (!strcmp(s + 8, "mid")) min_h = 256;
            else if (!strcmp(s + 8, "far")) min_h = 128;
            else { fprintf(stderr, "--range is near, mid or far\n"); usage(); return 1; }
            fprintf(stderr, "range: %s (smallest YOLO window %d px)\n", s + 8, min_h);
        }
    }
    for (int a = 1; a < argc; a++) {
        const char *s = argv[a];
        if (strncmp(s, "--scene=", 8) == 0 || strncmp(s, "--range=", 8) == 0) continue;     /* applied above */
        if (strncmp(s, "--frames=", 9) == 0) frames = atoi(s + 9);
        else if (strncmp(s, "--model=", 8) == 0) model = s + 8;
        else if (strcmp(s, "--fp32") == 0) fp32 = 1;
        else if (strncmp(s, "--out=", 6) == 0) g_cfg.out = s + 6;
        else if (strncmp(s, "--keep=", 7) == 0) g_cfg.keep = atoi(s + 7);
        else if (strncmp(s, "--classes=", 10) == 0) {
            memset(g_cfg.classes, 0, sizeof(g_cfg.classes));
            const char *p = s + 10;
            while (*p) {
                int c = atoi(p);
                if (c >= 0 && c < 80) g_cfg.classes[c] = 1;
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        }
        else if (strncmp(s, "--conf=", 7) == 0) g_cfg.conf = (float)atof(s + 7);
        else if (strncmp(s, "--vote=", 7) == 0) {
            int k, n;
            if (sscanf(s + 7, "%d/%d", &k, &n) != 2 || k < 1 || n < k || n > 8) { fprintf(stderr, "--vote=K/N (N up to 8)\n"); return 1; }
            g_cfg.vote_k = k; g_cfg.vote_n = n;
        }
        else if (strncmp(s, "--save-gap=", 11) == 0) g_cfg.save_gap_s = atof(s + 11);
        else if (strncmp(s, "--night-confirm=", 16) == 0) {
            if (!strcmp(s + 16, "on")) g_cfg.night_confirm = 1;
            else if (!strcmp(s + 16, "off")) g_cfg.night_confirm = 0;
            else { fprintf(stderr, "--night-confirm=on|off\n"); return 1; }
        }
        else if (strncmp(s, "--sad-thr=", 10) == 0) sad_thr = atoi(s + 10);
        else if (strncmp(s, "--motion-gap=", 13) == 0) motion_gap = atoi(s + 13);
        else if (strncmp(s, "--gray=", 7) == 0) {
            const char *v = s + 7;
            if (!strcmp(v, "off")) g_gray = 0;
            else if (!strcmp(v, "on")) g_gray = 1;
            else if (!strcmp(v, "night")) g_gray = 2;
            else { usage(); return 1; }
        }
        else if (strncmp(s, "--area-thr=", 11) == 0) area_thr = atoi(s + 11);
        else if (strncmp(s, "--min-cluster=", 14) == 0) min_cluster = atoi(s + 14);
        else if (strncmp(s, "--gap=", 6) == 0) gap = atoi(s + 6);
        else if (strncmp(s, "--margin=", 9) == 0) margin = atoi(s + 9);
        else if (strncmp(s, "--min-h=", 8) == 0) min_h = atoi(s + 8);
        else if (strncmp(s, "--min-avail=", 12) == 0) g_min_avail_kb = atol(s + 12);
        else if (strncmp(s, "--fps-drop=", 11) == 0) g_fps_drop = atoi(s + 11);
        else if (strncmp(s, "--ev-h=", 7) == 0) ev_h = atoi(s + 7);
        else if (strncmp(s, "--ev-max=", 9) == 0) g_cfg.ev_max_dim = atoi(s + 9);
        else if (strncmp(s, "--debug-crops=", 14) == 0) g_cfg.dbg_max = atoi(s + 14);
        else if (strncmp(s, "--status=", 9) == 0) status_s = atoi(s + 9);
        else if (strncmp(s, "--chn=", 6) == 0) chn = atoi(s + 6);
        else if (strcmp(s, "--no-roi") == 0) use_roi = 0;
        else if (strncmp(s, "--roi-mode=", 11) == 0) {
            const char *m = s + 11;
            if (strcmp(m, "touch") == 0) roi_mode = 0;
            else if (strcmp(m, "feet") == 0) roi_mode = 1;
            else if (strcmp(m, "inside") == 0) roi_mode = 2;
            else { usage(); return 1; }
        }
        else if (strcmp(s, "--verbose") == 0) verbose = 1;
        else { fprintf(stderr, "unknown option %s\n", s); usage(); return 1; }
    }
    if (frames < 0 || motion_gap < 1 || motion_gap > MAXGAP || sad_thr < 0 || sad_thr > 65535 || area_thr < 0 || area_thr > 65535 ||
        min_cluster < 1 || gap < 0 || margin < 0 || margin > 100 ||
        min_h < 1 || ev_h < 1 || g_cfg.ev_max_dim < 64 ||
        g_cfg.keep < 1 || g_cfg.conf < 0.30f || g_cfg.conf > 0.99f || g_min_avail_kb < 0 || g_fps_drop < 0 || g_fps_drop > 90 || g_cfg.save_gap_s < 0 || status_s < 0 || chn < 0) {
        fprintf(stderr, "invalid numeric option\n");
        usage();
        return 1;
    }

    /* only one instance: a second one doubles the memory use (~33 MB each) and stalls the camera */
    int lock_fd = open("/tmp/sdetector.lock", O_RDWR | O_CREAT, 0644);
    if (lock_fd >= 0 && flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        fprintf(stderr, "another sdetector is already running (lock /tmp/sdetector.lock), exiting\n");
        return 1;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    mkdir(g_cfg.out, 0755);

    /* ---- detector model ---- */
    if (fp32) yolo_set_fp16(0);
    {
        char pp[300], bp[300];
        snprintf(pp, sizeof(pp), "%s.param", model);
        snprintf(bp, sizeof(bp), "%s.bin", model);
        double t0 = now_ms();
        if (yolo_init(pp, bp) != 0) return 1;
        fprintf(stderr, "model %s loaded in %.0f ms%s\n", model, now_ms() - t0, fp32 ? " (fp32)" : "");
    }

    /* ---- video system, channels ---- */
    HI_S32 ret = HI_MPI_SYS_Init();
    if (ret != HI_SUCCESS) { fprintf(stderr, "SYS_Init: 0x%x\n", ret); return 1; }
    int src_w = 0, src_h = 0;
    int vr = vgrab_chn_size(0, 0, &src_w, &src_h);
    g_src_w = src_w;
    g_src_h = src_h;
    if (vr != 0) {
        fprintf(stderr, "vgrab_chn_size(0,0): 0x%x - Majestic's video pipeline is not running (is Majestic up?)\n", vr);
        return 1;
    }
    vr = vgrab_open(0, 0, chn);
    if (vr != 0) { fprintf(stderr, "vgrab_open(0,%d): 0x%x\n", chn, vr); return 1; }
    atexit(vgrab_close);
    vr = vgrab_open(1, 0, 0);
    if (vr != 0) { fprintf(stderr, "vgrab_open(grp0 ch0): 0x%x\n", vr); return 1; }

    int hmax = win_max < src_h ? win_max : src_h;
    hmax &= ~63;
    while (hmax > 64 && hmax * 3 / 4 > src_w) hmax -= 64;
    if (ev_h > hmax) ev_h = hmax;
    ev_h = (ev_h + 63) & ~63;
    if (ev_h > hmax) ev_h = hmax;
    if (min_h < 64) min_h = 64;
    if (min_h > hmax) min_h = hmax;

    IVE_MEM_INFO_S crop_mem;
    memset(&crop_mem, 0, sizeof(crop_mem));
    crop_mem.u32Size = (HI_U32)((size_t)(hmax * 3 / 4) * hmax * 3 / 2);
    if (HI_MPI_SYS_MmzAlloc(&crop_mem.u64PhyAddr, (HI_VOID **)&crop_mem.u64VirAddr,
            NULL, HI_NULL, crop_mem.u32Size) != HI_SUCCESS) { fprintf(stderr, "no MMZ for the crop buffer\n"); return 1; }
    fprintf(stderr, "windows: YOLO window %d..%d px high (enlarged below %d px), evidence window >= %dx%d, MMZ crop buffer %u KB\n",
            min_h, hmax, 256, ev_h * 3 / 4, ev_h, crop_mem.u32Size / 1024);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    pthread_t th_det;
    {
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setstacksize(&at, 2 * 1024 * 1024);
        if (pthread_create(&th_det, &at, detector_thread, NULL) != 0) { fprintf(stderr, "cannot start the detector thread\n"); return 1; }
        pthread_attr_destroy(&at);
    }

    IVE_IMAGE_S img_cur, img_sad16, img_sad;
    IVE_IMAGE_S ring[MAXGAP];            /* the last motion_gap frames: SAD compares with the oldest one */
    int ring_pos = 0, ring_have = 0;
    IVE_IMAGE_S *cur = &img_cur, *prev = &ring[0];
    IVE_MEM_INFO_S blob_mem;
    memset(&img_cur, 0, sizeof(img_cur));
    memset(ring, 0, sizeof(ring));
    memset(&img_sad, 0, sizeof(img_sad));
    memset(&img_sad16, 0, sizeof(img_sad16));
    memset(&blob_mem, 0, sizeof(blob_mem));

    int ive_ready = 0, W = 0, H = 0, bw = 0, bh = 0, ccl_ok = 0;
    Roi roi_raw[MAX_ROI], roi_px[MAX_ROI];
    int n_roi = 0, roi_ref_w = 0, roi_ref_h = 0;
    if (use_roi) n_roi = load_roi(roi_raw, MAX_ROI, &roi_ref_w, &roi_ref_h);
    int full_w = src_w, full_h = src_h;

    unsigned long n_frames = 0, n_motion = 0, n_targets = 0, n_jobs = 0;
    unsigned long n_jobfail = 0, n_busy = 0, n_getfail = 0, n_lowmem = 0;
    double t_lowmem_log = 0;
    int nominal_fps = g_fps_drop > 0 ? load_video1_fps() : 0;
    double fps_now = 0, t_fps = now_ms();
    unsigned long fr_fps = 0;
    int fps_ok_win = 0;
    if (g_gray) fprintf(stderr, "YOLO input: grey %s (colour otherwise)\n", g_gray == 1 ? "always" : "while Majestic night mode is on");
    if (g_fps_drop > 0) {
        if (nominal_fps > 0)
            fprintf(stderr, "fps guard: video1 is set to %d fps, out of normal below %.1f fps\n", nominal_fps, nominal_fps * (100 - g_fps_drop) / 100.0);
        else
            fprintf(stderr, "fps guard: video1.fps not found in /etc/majestic.yaml - off\n");
    }
    unsigned seq = 0;
    double t_begin = now_ms(), t_status = t_begin, sum_ive = 0;

    Job jb;     /* declared once: after each swap it holds the detector's old buffer, so its capacity is reused */
    for (unsigned long i = 0; !g_stop && (frames == 0 || i < (unsigned long)frames); i++) {
        VGrabFrame vcur;
        int gr = vgrab_get(0, &vcur, 500);
        if (gr != 0) {
            if ((n_getfail++ % 100) == 0) fprintf(stderr, "motion channel get failed: 0x%x (%lu so far)\n", gr, n_getfail);
            continue;
        }
        double t0 = now_ms();
        int dw = (int)vcur.width, dh = (int)vcur.height, fw = src_w, fh = src_h;

        /* IVE images are padded to W multiple of 64, H multiple of 16 (CCL needs that) */
        int nW = (dw + 63) & ~63, nH = (dh + 15) & ~15;
        if (!ive_ready) {
            W = nW; H = nH;
            if (nW != dw || nH != dh) {
                fprintf(stderr, "channel %dx%d: width must be a multiple of 64 and height of 16\n", dw, dh);
                return 1;
            }
            if (n_roi > 0) {
                if (roi_ref_w <= 0 || roi_ref_h <= 0) {
                    fprintf(stderr, "secured area found but video0.size is not in /etc/majestic.yaml - ignoring it\n");
                    n_roi = 0;
                }
                for (int k = 0; k < n_roi; k++) {
                    roi_px[k].x0 = (int)((double)roi_raw[k].x0 * fw / roi_ref_w);
                    roi_px[k].y0 = (int)((double)roi_raw[k].y0 * fh / roi_ref_h);
                    roi_px[k].x1 = (int)((double)roi_raw[k].x1 * fw / roi_ref_w);
                    roi_px[k].y1 = (int)((double)roi_raw[k].y1 * fh / roi_ref_h);
                    fprintf(stderr, "secured area #%d (%s): (%d,%d)-(%d,%d) in the %dx%d picture\n", k + 1,
                            roi_mode == 0 ? "touch" : roi_mode == 1 ? "feet" : "inside",
                            roi_px[k].x0, roi_px[k].y0, roi_px[k].x1, roi_px[k].y1, fw, fh);
                }
            } else if (use_roi) {
                fprintf(stderr, "no secured area in /etc/majestic.yaml - using the whole picture\n");
            }
            bw = W / BLOCK; bh = H / BLOCK;
            ccl_ok = (bw >= 64 && bh >= 64 && bw <= 720 && bh <= 640);
            if (!ccl_ok) { fprintf(stderr, "SAD output %dx%d is outside the CCL limits\n", bw, bh); return 1; }
            for (int g = 0; g < motion_gap; g++)
                if (alloc_img(&ring[g], W, H) != HI_SUCCESS) return 1;
            if (alloc_img(&img_sad, bw, bh) != HI_SUCCESS) return 1;
            img_sad16.enType = IVE_IMAGE_TYPE_U16C1;
            img_sad16.u32Width = bw;
            img_sad16.u32Height = bh;
            img_sad16.au32Stride[0] = (bw * 2 + 15) & ~15;
            if (HI_MPI_SYS_MmzAlloc(&img_sad16.au64PhyAddr[0], (HI_VOID **)&img_sad16.au64VirAddr[0], NULL, HI_NULL,
                    img_sad16.au32Stride[0] * bh) != HI_SUCCESS) return 1;
            blob_mem.u32Size = sizeof(IVE_CCBLOB_S);
            if (HI_MPI_SYS_MmzAlloc(&blob_mem.u64PhyAddr, (HI_VOID **)&blob_mem.u64VirAddr, NULL, HI_NULL,
                    blob_mem.u32Size) != HI_SUCCESS) return 1;
            ive_ready = 1;
            dma_copy(&vcur, &ring[0]);                  /* the first frames only prime the ring */
            ring_pos = 1 % motion_gap;
            ring_have = 1;
            vgrab_release(0);
            fprintf(stderr, "running: motion %dx%d (1 block = %d px), stop with Ctrl-C or killall sdetector\n", dw, dh,
                    (int)(BLOCK * (double)fw / dw + 0.5));
            continue;
        }
        if (dw != W || dh != H) { fprintf(stderr, "frame size changed\n"); break; }
        wrap_img(cur, &vcur);

        if (ring_have < motion_gap) {                    /* still filling the ring (motion_gap > 1) */
            dma_copy(&vcur, &ring[ring_pos]);
            ring_pos = (ring_pos + 1) % motion_gap;
            ring_have++;
            vgrab_release(0);
            continue;
        }
        prev = &ring[ring_pos];                          /* the frame motion_gap steps ago */

        /* ---- IVE: SAD (4x4 blocks, threshold), then CCL ---- */
        IVE_HANDLE handle;
        IVE_SAD_CTRL_S sad_ctrl;
        memset(&sad_ctrl, 0, sizeof(sad_ctrl));
        sad_ctrl.enMode = IVE_SAD_MODE_MB_4X4;
        sad_ctrl.enOutCtrl = IVE_SAD_OUT_CTRL_THRESH;
        sad_ctrl.u16Thr = (HI_U16)sad_thr;
        sad_ctrl.u8MinVal = 0;
        sad_ctrl.u8MaxVal = 255;
        ret = HI_MPI_IVE_SAD(&handle, cur, prev, &img_sad16, &img_sad, &sad_ctrl, HI_TRUE);
        if (ret != HI_SUCCESS) { fprintf(stderr, "SAD ret=0x%x\n", ret); break; }
        if (ive_wait(handle) != HI_SUCCESS) break;
        dma_copy(&vcur, prev);                          /* current takes the place of the oldest, by hardware */
        ring_pos = (ring_pos + 1) % motion_gap;
        vgrab_release(0);

        IVE_CCL_CTRL_S ccl_ctrl;
        memset(&ccl_ctrl, 0, sizeof(ccl_ctrl));
        ccl_ctrl.enMode = IVE_CCL_MODE_4C;
        ccl_ctrl.u16InitAreaThr = (HI_U16)area_thr;
        ccl_ctrl.u16Step = 2;
        ret = HI_MPI_IVE_CCL(&handle, &img_sad, &blob_mem, &ccl_ctrl, HI_TRUE);
        if (ret != HI_SUCCESS) { fprintf(stderr, "CCL ret=0x%x\n", ret); break; }
        if (ive_wait(handle) != HI_SUCCESS) break;
        double t_ive = now_ms();
        sum_ive += t_ive - t0;
        n_frames++;

        if (nominal_fps > 0) {              /* every 5 s: how many frames did the sensor really deliver */
            double tn = now_ms();
            if (tn - t_fps >= 5000.0) {
                fps_now = (n_frames - fr_fps) * 1000.0 / (tn - t_fps);
                t_fps = tn;
                fr_fps = n_frames;
                double lo = nominal_fps * (100 - g_fps_drop) / 100.0;
                double hi = lo + (nominal_fps - lo) / 3.0;          /* back to normal a bit above the alarm level */
                if (!g_fps_low && fps_now < lo) {
                    g_fps_low = 1;
                    fps_ok_win = 0;
                    logf("WARNING: camera is out of normal fps: %.1f fps against %d set, intrusions may be missed; "
                         "the night second pass is off until it recovers", fps_now, nominal_fps);
                } else if (g_fps_low) {
                    fps_ok_win = (fps_now >= hi) ? fps_ok_win + 1 : 0;
                    if (fps_ok_win >= 2) {
                        g_fps_low = 0;
                        logf("camera fps is back to normal: %.1f fps against %d set, the night second pass is restored", fps_now, nominal_fps);
                    }
                }
            }
        }

        HI_MPI_SYS_MmzFlushCache(blob_mem.u64PhyAddr, (HI_VOID *)(HI_UL)blob_mem.u64VirAddr, blob_mem.u32Size);
        IVE_CCBLOB_S *blob = (IVE_CCBLOB_S *)(HI_UL)blob_mem.u64VirAddr;

        Reg regs[IVE_MAX_REGION_NUM];
        int nr = 0, n_outside = 0;
        for (int r = 0; r < IVE_MAX_REGION_NUM; r++) {
            if (blob->astRegion[r].u32Area == 0) continue;
            int x0 = (int)(blob->astRegion[r].u16Left * BLOCK * (double)fw / dw);
            int y0 = (int)(blob->astRegion[r].u16Top * BLOCK * (double)fh / dh);
            int x1 = (int)((blob->astRegion[r].u16Right + 1) * BLOCK * (double)fw / dw);
            int y1 = (int)((blob->astRegion[r].u16Bottom + 1) * BLOCK * (double)fh / dh);
            if (x1 > full_w) x1 = full_w;
            if (y1 > full_h) y1 = full_h;
            if (n_roi > 0 && !roi_accepts(roi_px, n_roi, roi_mode, x0, y0, x1, y1)) { n_outside++; continue; }
            regs[nr].x0 = x0; regs[nr].y0 = y0; regs[nr].x1 = x1; regs[nr].y1 = y1;
            regs[nr].area = (unsigned)blob->astRegion[r].u32Area;
            nr++;
        }

        if (nr > 0) {
            n_motion++;
            int cx0, cy0, cx1, cy1;
            unsigned ctotal = 0;
            pick_cluster(regs, nr, gap, &cx0, &cy0, &cx1, &cy1, &ctotal);
            if (verbose)
                printf("F%lu MOTION regions=%d (outside area: %d) target=(%d,%d)-(%d,%d) area=%u  ive=%.1f ms\n", i, nr, n_outside,
                       cx0, cy0, cx1, cy1, ctotal, t_ive - t0);
            if (ctotal >= (unsigned)min_cluster) {
                n_targets++;
                pthread_mutex_lock(&g_mu);
                int idle = (g_det_state == DET_IDLE);
                pthread_mutex_unlock(&g_mu);

                int lowmem = 0;
                if (idle && g_min_avail_kb > 0) {
                    long av = kb_of("/proc/meminfo", "MemAvailable:");
                    if (av >= 0 && av < g_min_avail_kb) {
                        lowmem = 1;
                        idle = 0;
                        n_lowmem++;
                        if (now_ms() - t_lowmem_log > 10000.0) {
                            t_lowmem_log = now_ms();
                            logf("LOW MEMORY: MemAvailable %ld kB < %ld kB, job skipped (%lu so far)", av, g_min_avail_kb, n_lowmem);
                        }
                    }
                }

                if (idle) {
                    double dma_ms = 0;
                    unsigned job_seq = seq + 1;
                    int br = build_job(jb, cx0, cy0, cx1, cy1, full_w, full_h, margin,
                                       min_h, hmax, ev_h, &crop_mem, job_seq, &dma_ms);
                    if (br == 0) {
                        pthread_mutex_lock(&g_mu);
                        if (g_det_state == DET_IDLE && !g_stop) {
                            seq = job_seq;
                            std::swap(g_job, jb);
                            g_det_state = DET_READY;
                            pthread_cond_signal(&g_cv);
                            pthread_mutex_unlock(&g_mu);
                            n_jobs++;
                            if (verbose) printf("  job #%u: window cut in %.1f ms\n", seq, dma_ms);
                        } else {
                            pthread_mutex_unlock(&g_mu);
                            n_busy++;
                        }
                    } else {
                        n_jobfail++;
                        if (verbose) printf("  crop failed (%d)\n", br);
                    }
                } else if (!lowmem) {
                    n_busy++;
                }
            }
        } else if (verbose) {
            printf("F%lu quiet (outside area: %d)  ive=%.1f ms\n", i, n_outside, t_ive - t0);
        }

        if (status_s > 0 && now_ms() - t_status >= status_s * 1000.0) {
            t_status = now_ms();
            unsigned long analysed, positive_count, event_count;
            double sum_yolo, max_yolo;
            pthread_mutex_lock(&g_stats_mu);
            analysed = g_n_analysed;
            positive_count = g_n_pos;
            event_count = g_n_events;
            sum_yolo = g_sum_yolo;
            max_yolo = g_max_yolo;
            pthread_mutex_unlock(&g_stats_mu);
            logf("STATUS up=%.0fs frames=%lu motion=%lu targets=%lu jobs=%lu (failed %lu, busy %lu) analysed=%lu positive=%lu intrusions=%lu "
                 "yolo avg=%.0f max=%.0f ms fps=%.1f%s | rss=%ld kB peak=%ld kB memfree=%ld kB cached=%ld kB",
                 (t_status - t_begin) / 1000.0, n_frames, n_motion, n_targets, n_jobs, n_jobfail, n_busy,
                 analysed, positive_count, event_count,
                 analysed ? sum_yolo / analysed : 0.0, max_yolo, fps_now, g_fps_low ? " LOW" : "",
                 kb_of("/proc/self/status", "VmRSS:"), kb_of("/proc/self/status", "VmHWM:"),
                 kb_of("/proc/meminfo", "MemFree:"), kb_of("/proc/meminfo", "Cached:"));
        }
    }

    /* ---- shutdown ---- */
    g_stop = 1;
    pthread_mutex_lock(&g_mu);
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
    pthread_join(th_det, NULL);
    logf("STOP frames=%lu motion=%lu targets=%lu jobs=%lu failed=%lu busy=%lu analysed=%lu positive=%lu intrusions=%lu | rss=%ld kB peak=%ld kB memfree=%ld kB",
         n_frames, n_motion, n_targets, n_jobs, n_jobfail, n_busy, g_n_analysed, g_n_pos, g_n_events,
         kb_of("/proc/self/status", "VmRSS:"), kb_of("/proc/self/status", "VmHWM:"), kb_of("/proc/meminfo", "MemFree:"));
    if (n_frames) fprintf(stderr, "average motion step %.2f ms\n", sum_ive / n_frames);

    if (ive_ready) {
        for (int g = 0; g < motion_gap; g++) free_img(&ring[g]);
        free_img(&img_sad);
        if (img_sad16.au64PhyAddr[0])
            HI_MPI_SYS_MmzFree(img_sad16.au64PhyAddr[0], (HI_VOID *)(HI_UL)img_sad16.au64VirAddr[0]);
        HI_MPI_SYS_MmzFree(blob_mem.u64PhyAddr, (HI_VOID *)(HI_UL)blob_mem.u64VirAddr);
    }
    vgrab_close();
    HI_MPI_SYS_MmzFree(crop_mem.u64PhyAddr, (HI_VOID *)(HI_UL)crop_mem.u64VirAddr);
    HI_MPI_SYS_Exit();
    return 0;
}