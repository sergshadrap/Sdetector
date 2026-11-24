/* yolo_infer.cpp - the YOLO26 detector as a small library:
 * takes an RGB buffer (no JPEG), runs YOLO26 through ncnn, decodes and applies NMS.
 * Compile it WITHOUT the Hisilicon headers (only ncnn's). */
#include "net.h"

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <vector>

#include "yolo_infer.h"

#ifndef INPUT_W
#define INPUT_W         192
#endif
#ifndef INPUT_H
#define INPUT_H         256
#endif

#define NUM_ANCHORS     ((INPUT_W / 8) * (INPUT_H / 8) + \
                         (INPUT_W / 16) * (INPUT_H / 16) + \
                         (INPUT_W / 32) * (INPUT_H / 32))
#define NUM_CLASSES     80
#define LIST_THRESH     0.30f
#define NMS_IOU         0.45f

static const char *coco_names[NUM_CLASSES] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

/* never destroyed on purpose (the process just exits) */
static ncnn::Net *g_net = NULL;
static ncnn::PoolAllocator *g_blob = NULL;
static ncnn::PoolAllocator *g_ws = NULL;

static int g_fp16 = 1;
void yolo_set_fp16(int on) { g_fp16 = on; }

const char *yolo_class_name(int cls)
{
    return (cls >= 0 && cls < NUM_CLASSES) ? coco_names[cls] : "?";
}

int yolo_init(const char *param_path, const char *bin_path)
{
    g_blob = new ncnn::PoolAllocator;
    g_ws = new ncnn::PoolAllocator;
    g_net = new ncnn::Net;

    g_net->opt.lightmode = true;
    g_net->opt.num_threads = 1;
    g_net->opt.use_vulkan_compute = false;
    g_net->opt.use_fp16_packed = g_fp16 != 0;
    g_net->opt.use_fp16_storage = g_fp16 != 0;
    g_net->opt.use_fp16_arithmetic = g_fp16 != 0;
    g_net->opt.use_packing_layout = true;
    /* winograd stays at the ncnn default (on): ~11 MB more RAM, ~100 ms faster. It froze the camera at 5 MP
     * with isp.blkCnt on automatic (6); needs isp.blkCnt 5 (see OpenIPC/firmware issue 2828) */
    g_net->opt.blob_allocator = g_blob;
    g_net->opt.workspace_allocator = g_ws;

    if (g_net->load_param(param_path) != 0 || g_net->load_model(bin_path) != 0) {
        fprintf(stderr, "yolo_init: cannot load %s / %s\n", param_path, bin_path);
        return -1;
    }
    return 0;
}

static float iou(const YoloDet &a, const YoloDet &b)
{
    float iw = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float ih = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw * ih;
    float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0 ? inter / uni : 0.f;
}

int yolo_run(const unsigned char *rgb, int w, int h,
             YoloDet *out, int max_out, float *best_person, double *infer_ms)
{
    if (!g_net) return -1;
    auto t0 = std::chrono::high_resolution_clock::now();

    ncnn::Mat input = (w == INPUT_W && h == INPUT_H)
        ? ncnn::Mat::from_pixels(rgb, ncnn::Mat::PIXEL_RGB, w, h)
        : ncnn::Mat::from_pixels_resize(rgb, ncnn::Mat::PIXEL_RGB, w, h, INPUT_W, INPUT_H);
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    input.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = g_net->create_extractor();
    ex.input("in0", input);
    ncnn::Mat o;
    if (ex.extract("out0", o) != 0) { fprintf(stderr, "yolo_run: extract() failed\n"); return -1; }
    if (o.dims != 2 || o.w != NUM_ANCHORS || o.h != (4 + NUM_CLASSES)) {
        fprintf(stderr, "yolo_run: unexpected out0 shape dims=%d w=%d h=%d (expected w=%d h=%d)\n",
                o.dims, o.w, o.h, NUM_ANCHORS, 4 + NUM_CLASSES);
        return -1;
    }

    const float *cx_row = o.row(0), *cy_row = o.row(1), *w_row = o.row(2), *h_row = o.row(3);
    const float *person_row = o.row(4);
    float sx = (float)w / INPUT_W, sy = (float)h / INPUT_H;
    float bp = 0.f;
    std::vector<YoloDet> cands;

    for (int a = 0; a < NUM_ANCHORS; a++) {
        if (person_row[a] > bp) bp = person_row[a];
        int a_cls = -1;
        float a_best = 0.f;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = o.row(4 + c)[a];
            if (score > a_best) { a_best = score; a_cls = c; }
        }
        if (a_best >= LIST_THRESH) {
            YoloDet d;
            d.cls = a_cls;
            d.conf = a_best;
            d.x1 = std::max(0.f, (cx_row[a] - w_row[a] / 2) * sx);
            d.y1 = std::max(0.f, (cy_row[a] - h_row[a] / 2) * sy);
            d.x2 = std::min((float)w, (cx_row[a] + w_row[a] / 2) * sx);
            d.y2 = std::min((float)h, (cy_row[a] + h_row[a] / 2) * sy);
            cands.push_back(d);
        }
    }
    std::sort(cands.begin(), cands.end(), [](const YoloDet &a, const YoloDet &b) { return a.conf > b.conf; });
    if (cands.size() > 100) cands.resize(100);

    int n = 0;
    for (size_t i = 0; i < cands.size() && n < max_out; i++) {
        bool drop = false;
        for (int k = 0; k < n; k++)
            if (out[k].cls == cands[i].cls && iou(out[k], cands[i]) > NMS_IOU) { drop = true; break; }
        if (!drop) out[n++] = cands[i];
    }

    if (best_person) *best_person = bp;
    g_blob->clear();
    g_ws->clear();
    if (infer_ms)
        *infer_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    return n;
}