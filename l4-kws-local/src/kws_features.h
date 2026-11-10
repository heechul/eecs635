#pragma once
#include <cstdint>

namespace kws {
constexpr int kSampleRate = 16000;
constexpr int kSamples = 16000;
constexpr int kWindow = 480;
constexpr int kHop = 320;
constexpr int kFft = 512;
constexpr int kBands = 32;
constexpr int kFrames = 49;
constexpr int kFeatures = kFrames * kBands;

// One instance, used by one task. Scratch storage avoids large task stacks.
class Frontend {
 public:
  Frontend();
  void Compute(const int16_t* pcm, float* features);
 private:
  float window_[kWindow];
  // Each FFT bin belongs to at most two triangular filters. Store only their
  // nonzero weights, precomputed once rather than doing divisions per frame.
  uint16_t band_start_[kBands];
  uint16_t band_offset_[kBands + 1];
  float mel_weights_[2 * (kFft / 2 + 1)];
  float fft_step_real_[9], fft_step_imag_[9];
  float real_[kFft], imag_[kFft];
  float power_[kFft / 2 + 1];
  void Fft();
};
int8_t Quantize(float value, float scale, int zero_point);
}  // namespace kws
