#include "kws_features.h"
#include <cmath>

namespace kws {
namespace {
constexpr float kPi = 3.14159265358979323846f;
float Mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
}
Frontend::Frontend() {
  for (int i = 0; i < kWindow; ++i)
    window_[i] = 0.5f - 0.5f * cosf(2 * kPi * i / kWindow);
  const float lo = Mel(80), hi = Mel(7600);
  float edges_[kBands + 2];
  for (int i = 0; i < kBands + 2; ++i)
    edges_[i] = 700 * (powf(10, (lo + (hi-lo)*i/(kBands+1))/2595) - 1);
  uint16_t offset = 0;
  for (int band = 0; band < kBands; ++band) {
    band_offset_[band] = offset;
    band_start_[band] = 0;
    bool first = true;
    for (int bin = 0; bin <= kFft/2; ++bin) {
      float hz = bin * (float(kSampleRate)/kFft);
      float weight = fmaxf(0, fminf((hz-edges_[band])/(edges_[band+1]-edges_[band]),
                                  (edges_[band+2]-hz)/(edges_[band+2]-edges_[band+1])));
      if (weight > 0) {
        if (first) { band_start_[band] = bin; first = false; }
        mel_weights_[offset++] = weight;
      }
    }
  }
  band_offset_[kBands] = offset;
  for (int len = 2, stage = 0; len <= kFft; len <<= 1, ++stage) {
    fft_step_real_[stage] = cosf(-2*kPi/len);
    fft_step_imag_[stage] = sinf(-2*kPi/len);
  }
}
void Frontend::Fft() {
  for (int i = 1, j = 0; i < kFft; ++i) {
    int bit = kFft >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = real_[i]; real_[i] = real_[j]; real_[j] = t;
    }
  }
  for (int len = 2, stage = 0; len <= kFft; len <<= 1, ++stage) {
    const float wr = fft_step_real_[stage], wi = fft_step_imag_[stage];
    for (int base = 0; base < kFft; base += len) {
      float r = 1, im = 0;
      for (int j = 0; j < len/2; ++j) {
        int a = base+j, b = a+len/2;
        float br = real_[b]*r - imag_[b]*im;
        float bi = real_[b]*im + imag_[b]*r;
        real_[b] = real_[a]-br; imag_[b] = imag_[a]-bi;
        real_[a] += br; imag_[a] += bi;
        float next = r*wr - im*wi;
        im = r*wi + im*wr; r = next;
      }
    }
  }
}
void Frontend::Compute(const int16_t* pcm, float* features) {
  for (int frame = 0; frame < kFrames; ++frame) {
    for (int i = 0; i < kFft; ++i) {
      real_[i] = i < kWindow ? pcm[frame*kHop+i]/32768.0f*window_[i] : 0;
      imag_[i] = 0;
    }
    Fft();
    for (int i = 0; i <= kFft/2; ++i)
      power_[i] = real_[i]*real_[i] + imag_[i]*imag_[i];
    for (int band = 0; band < kBands; ++band) {
      float energy = 0;
      int bin = band_start_[band];
      for (int i = band_offset_[band]; i < band_offset_[band+1]; ++i, ++bin)
        energy += power_[bin]*mel_weights_[i];
      float value = (logf(fmaxf(energy, 1e-7f)) + 16.0f)/24.0f;
      features[frame*kBands+band] = fminf(1, fmaxf(0, value));
    }
  }
}
int8_t Quantize(float value, float scale, int zero_point) {
  int q = int(roundf(value/scale)) + zero_point;
  return int8_t(q < -128 ? -128 : q > 127 ? 127 : q);
}
}  // namespace kws
