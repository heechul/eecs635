#include "NeuralNetwork.h"
#include "model.h"
#include "model_config.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>

namespace {
// Measured arena use for this model: 270460 bytes. Leave ~16 KiB headroom.
// Recheck this budget when changing the model or TFLite Micro kernels.
constexpr size_t kTensorArenaSize = 280 * 1024;
constexpr uint32_t kArenaCaps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
bool validTensor(const TfLiteTensor* tensor, int height, int width, int channels,
                 float scale, int zero_point) {
    return tensor && tensor->dims && tensor->dims->size == 4 &&
        tensor->dims->data[0] == 1 && tensor->dims->data[1] == height &&
        tensor->dims->data[2] == width && tensor->dims->data[3] == channels &&
        tensor->type == kTfLiteInt8 && tensor->bytes == height * width * channels &&
        tensor->params.scale > 0 && fabsf(tensor->params.scale - scale) < 1e-7f &&
        tensor->params.zero_point == zero_point;
}
}

bool NeuralNetwork::begin() {
    if (ready_) return true;
    // Initialized once by setup(); the arena/interpreter live for the whole boot.
    const tflite::Model* model = tflite::GetModel(gmodel);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Unsupported TFLite schema");
        return false;
    }
    // Quantized MobileNetV2: BN/ReLU6 fold into convolutions during conversion.
    static tflite::MicroMutableOpResolver<4> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddAdd() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) {
        Serial.println("Operator registration failed");
        return false;
    }
    const size_t free_internal = heap_caps_get_free_size(kArenaCaps);
    const size_t largest_internal = heap_caps_get_largest_free_block(kArenaCaps);
    Serial.printf("Internal SRAM: free=%u bytes, largest block=%u bytes\n",
        static_cast<unsigned>(free_internal), static_cast<unsigned>(largest_internal));
    const char* arena_memory = "internal SRAM";
    // Prefer internal SRAM with 16-byte alignment for TFLite Micro/ESP-NN.
    arena_ = static_cast<uint8_t*>(heap_caps_aligned_alloc(
        16, kTensorArenaSize, kArenaCaps));
    if (!arena_) {
        Serial.printf("Internal SRAM arena allocation failed: need %u contiguous bytes "
                      "(largest block before allocation: %u bytes)\n",
            static_cast<unsigned>(kTensorArenaSize), static_cast<unsigned>(largest_internal));

        // Fallback to SPIRAM if internal SRAM allocation fails.
        arena_memory = "PSRAM";
        arena_ = static_cast<uint8_t*>(heap_caps_aligned_alloc(
            16, kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (!arena_) {
            Serial.println("SPIRAM arena allocation failed");
            return false;
        }
    }
    static tflite::MicroInterpreter interpreter(model, resolver, arena_, kTensorArenaSize);
    interpreter_ = &interpreter;
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        Serial.printf("AllocateTensors failed (%s arena: %u bytes)\n",
            arena_memory, static_cast<unsigned>(kTensorArenaSize));
        return false;
    }
    input_ = interpreter_->input(0);
    output_ = interpreter_->output(0);
    if (!validTensor(input_, fomo::kInputHeight, fomo::kInputWidth, fomo::kChannels,
                     fomo::kInputScale, fomo::kInputZeroPoint) ||
        !validTensor(output_, fomo::kGridHeight, fomo::kGridWidth, fomo::kClassCount,
                     fomo::kOutputScale, fomo::kOutputZeroPoint)) {
        Serial.println("FOMO tensor contract mismatch; export all model files together");
        return false;
    }
    Serial.printf("FOMO model: %u bytes, tensor arena (%s): %u / %u bytes\n",
        gmodel_len, arena_memory, static_cast<unsigned>(interpreter_->arena_used_bytes()),
        static_cast<unsigned>(kTensorArenaSize));
    ready_ = true;
    return true;
}

TfLiteStatus NeuralNetwork::predict() {
    return ready_ ? interpreter_->Invoke() : kTfLiteError;
}
