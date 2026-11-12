#pragma once
#include <stdint.h>
#include "model_config.h"

namespace fomo {
constexpr int kMaxDetections = kGridWidth * kGridHeight;
struct Detection {
    int class_id;
    float x;
    float y;
    float score;
};

// Teaching decoder: same-class 4-connected components, weighted cell centers.
// Input is contiguous NHWC int8 [1, grid_h, grid_w, classes].
inline int decode(const int8_t* output, float scale, int zero_point,
                  Detection* detections, int capacity, float threshold = kThreshold) {
    if (!output || !detections || capacity <= 0 || scale <= 0) return 0;
    uint8_t labels[kMaxDetections] = {};
    bool seen[kMaxDetections] = {};
    int queue[kMaxDetections];
    for (int cell = 0; cell < kMaxDetections; ++cell) {
        int winner = 0;
        for (int c = 1; c < kClassCount; ++c) {
            if (output[cell * kClassCount + c] > output[cell * kClassCount + winner]) {
                winner = c;
            }
        }
        const float score = (output[cell * kClassCount + winner] - zero_point) * scale;
        if (winner != 0 && score >= threshold) labels[cell] = winner;
    }
    int count = 0;
    for (int start = 0; start < kMaxDetections; ++start) {
        const int cid = labels[start];
        if (!cid || seen[start]) continue;
        int head = 0, tail = 0;
        queue[tail++] = start;
        seen[start] = true;  // Mark when enqueuing: each cell enters at most once.
        float sx = 0, sy = 0, weight = 0, peak = 0;
        while (head < tail) {
            const int cell = queue[head++];
            const int x = cell % kGridWidth, y = cell / kGridWidth;
            const float score = (output[cell * kClassCount + cid] - zero_point) * scale;
            sx += (x + 0.5f) * kInputWidth / kGridWidth * score;
            sy += (y + 0.5f) * kInputHeight / kGridHeight * score;
            weight += score;
            if (score > peak) peak = score;
            const int neighbors[] = {y > 0 ? cell - kGridWidth : -1,
                y + 1 < kGridHeight ? cell + kGridWidth : -1,
                x > 0 ? cell - 1 : -1, x + 1 < kGridWidth ? cell + 1 : -1};
            for (int next : neighbors) {
                if (next >= 0 && !seen[next] && labels[next] == cid) {
                    seen[next] = true;
                    queue[tail++] = next;
                }
            }
        }
        if (count < capacity) {
            detections[count++] = {cid, sx / weight, sy / weight, peak};
        }
    }
    return count;
}
}  // namespace fomo
