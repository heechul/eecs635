// FOMO cube centroid detection on Seeed XIAO ESP32S3 Sense.
// Based on the person-detection lab (C) 2023 Heechul Yun; MIT License.
#include <Arduino.h>
#include <esp_timer.h>
#include <string.h>
#include "NeuralNetwork.h"
#include "FomoDecoder.h"
#include "ImagePreprocessing.h"

#ifndef FOMO_STATIC_IMAGE
#define FOMO_STATIC_IMAGE 1
#endif
#if FOMO_STATIC_IMAGE
#include "img.h"
#else
#include "esp_camera.h"
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"
#endif

namespace {
constexpr int kLedPin = 21;  // XIAO user LED is active low.
NeuralNetwork network;
bool running = false;
fomo::Detection detections[fomo::kMaxDetections];

#if !FOMO_STATIC_IMAGE
bool startCamera() {
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.frame_size = FRAMESIZE_96X96;
    config.pixel_format = PIXFORMAT_RGB565;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    const esp_err_t result = esp_camera_init(&config);
    if (result != ESP_OK) {
        Serial.printf("Camera initialization failed: 0x%x\n", result);
        return false;
    }
    return true;
}
#endif
}

void setup() {
    Serial.begin(115200);
    pinMode(kLedPin, OUTPUT);
    digitalWrite(kLedPin, HIGH);
    const unsigned long started = millis();
    while (!Serial && millis() - started < 5000) delay(10);
    Serial.printf("FOMO cubes (%s input)\n", FOMO_STATIC_IMAGE ? "static" : "camera");
    if (!network.begin()) return;
#if !FOMO_STATIC_IMAGE
    if (!startCamera()) return;
#else
    static_assert(sizeof(img_data) == fomo::kInputWidth * fomo::kInputHeight * fomo::kChannels,
                  "Regenerate static image with the notebook");
#endif
    running = true;
}

void loop() {
    if (!running) { delay(1000); return; }
    TfLiteTensor* input = network.getInput();
    int64_t start = esp_timer_get_time();
#if FOMO_STATIC_IMAGE
    memcpy(input->data.int8, img_data, sizeof(img_data));
#else
    camera_fb_t* frame = esp_camera_fb_get();
    if (!frame) {
        Serial.println("Camera capture failed");
        digitalWrite(kLedPin, HIGH);
        delay(100);
        return;
    }
    const bool prepared = frame->format == PIXFORMAT_RGB565 &&
        fomo::prepareRgb565(frame->buf, frame->len, frame->width, frame->height,
                           input->data.int8, input->params.scale, input->params.zero_point);
    esp_camera_fb_return(frame);  // Return on every path after successful acquisition.
    if (!prepared) {
        Serial.println("Expected a 96x96 RGB565 camera frame");
        digitalWrite(kLedPin, HIGH);
        delay(100);
        return;
    }
#endif
    const int64_t prep_us = esp_timer_get_time() - start;
    start = esp_timer_get_time();
    if (network.predict() != kTfLiteOk) {
        Serial.println("Inference failed; stopping");
        digitalWrite(kLedPin, HIGH);
        running = false;
        return;
    }
    const int64_t infer_us = esp_timer_get_time() - start;
    TfLiteTensor* output = network.getOutput();
    const int count = fomo::decode(output->data.int8, output->params.scale,
        output->params.zero_point, detections, fomo::kMaxDetections);
    int counts[fomo::kClassCount] = {};
    for (int i = 0; i < count; ++i) {
        const auto& d = detections[i];
        ++counts[d.class_id];
        Serial.printf("%s confidence=%.3f x=%.1f y=%.1f\n",
                      fomo::kLabels[d.class_id], d.score, d.x, d.y);
    }
    Serial.printf("Objects=%d", count);
    for (int cid = 1; cid < fomo::kClassCount; ++cid) {
        Serial.printf(" %s=%d", fomo::kLabels[cid], counts[cid]);
    }
    Serial.printf(" | preprocess=%.2f ms inference=%.2f ms\n", prep_us / 1000., infer_us / 1000.);
    digitalWrite(kLedPin, count ? LOW : HIGH);
    delay(250);
}
