#ifndef YOLO_INFER_H
#define YOLO_INFER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int cls;                  /* COCO class id, 0 = person */
    float conf;
    float x1, y1, x2, y2;     /* in pixels of the image given to yolo_run() */
} YoloDet;

/* call BEFORE yolo_init(): on = 1 (default) lets ncnn use fp16 storage/arithmetic, 0 forces
 * plain fp32 (the weights of an fp16 model are converted to fp32 while loading) */
void yolo_set_fp16(int on);

/* loads the ncnn model once; returns 0 on success */
int yolo_init(const char *param_path, const char *bin_path);

/* rgb: packed R,G,B, w x h (the crop). The model input is 192x256 (build with
 * -DINPUT_W/-DINPUT_H for another model); other sizes are resized.
 * Returns the number of objects after NMS (conf >= 0.30, at most max_out), or -1 on error.
 * best_person gets the best person confidence of the whole output (the caller decides what counts;
 * objects below 0.30 are never listed). */
int yolo_run(const unsigned char *rgb, int w, int h,
             YoloDet *out, int max_out, float *best_person, double *infer_ms);

const char *yolo_class_name(int cls);

#ifdef __cplusplus
}
#endif
#endif