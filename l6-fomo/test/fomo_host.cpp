// Host executable exercises exactly the decoder and preprocessor used on device.
#include "FomoDecoder.h"
#include "ImagePreprocessing.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    std::ifstream file(argv[2], std::ios::binary);
    if (!file) return 2;
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
    if (!strcmp(argv[1], "decode")) {
        if (bytes.size() != fomo::kMaxDetections * fomo::kClassCount) return 3;
        fomo::Detection detections[fomo::kMaxDetections];
        const int count = fomo::decode(reinterpret_cast<const int8_t*>(bytes.data()),
            fomo::kOutputScale, fomo::kOutputZeroPoint, detections, fomo::kMaxDetections);
        printf("[");
        for (int i = 0; i < count; ++i) {
            const auto& d = detections[i];
            printf("%s{\"class_id\":%d,\"x\":%.8f,\"y\":%.8f,\"score\":%.8f}",
                   i ? "," : "", d.class_id, d.x, d.y, d.score);
        }
        printf("]\n");
        return 0;
    }
    if (!strcmp(argv[1], "prepare")) {
        int8_t input[fomo::kInputWidth * fomo::kInputHeight * 3];
        if (!fomo::prepareRgb565(bytes.data(), bytes.size(), fomo::kInputWidth,
                fomo::kInputHeight, input, fomo::kInputScale, fomo::kInputZeroPoint)) return 4;
        return fwrite(input, 1, sizeof(input), stdout) == sizeof(input) ? 0 : 5;
    }
    return 6;
}
