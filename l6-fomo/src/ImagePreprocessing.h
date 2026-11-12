#pragma once
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "model_config.h"

namespace fomo {
inline int8_t quantizePixel(uint8_t pixel, float scale, int zero_point) {
    const int value = static_cast<int>(floorf((pixel / 255.0f) / scale + zero_point + 0.5f));
    return static_cast<int8_t>(value < -128 ? -128 : (value > 127 ? 127 : value));
}

// Camera sends the high RGB565 byte first. Expand R5/G6/B5 with bit replication.
// Require exact input geometry so centroids remain in camera pixel coordinates.
inline bool prepareRgb565(const uint8_t* frame, size_t length, int width, int height,
                          int8_t* input, float scale, int zero_point) {
    if (!frame || !input || scale <= 0 || width != kInputWidth || height != kInputHeight ||
        length < static_cast<size_t>(width * height * 2)) return false;
    for (int pixel = 0; pixel < width * height; ++pixel) {
        const uint16_t color = (static_cast<uint16_t>(frame[2 * pixel]) << 8) | frame[2 * pixel + 1];
        const uint8_t r = (color >> 11) & 0x1f;
        const uint8_t g = (color >> 5) & 0x3f;
        const uint8_t b = color & 0x1f;
        input[3 * pixel] = quantizePixel((r << 3) | (r >> 2), scale, zero_point);
        input[3 * pixel + 1] = quantizePixel((g << 2) | (g >> 4), scale, zero_point);
        input[3 * pixel + 2] = quantizePixel((b << 3) | (b >> 2), scale, zero_point);
    }
    return true;
}
}  // namespace fomo
