// Standalone KWS lab: XIAO ESP32S3 Sense, Arduino ESP32 2.0.17.
#include <Arduino.h>
#include <atomic>
#include <cstdarg>
#include <esp_system.h>
#include "kws_features.h"

namespace {
std::atomic<const char*> stage{"boot"};
std::atomic<const char*> last_error{"none"};

void Log(const char* tag, const char* format, ...) {
  char message[224];
  va_list args;
  va_start(args, format);
  vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  Serial.printf("[%8lu] [%s] %s\n", static_cast<unsigned long>(millis()), tag, message);
}

void StatusTask(void*) {
  // Keep reporting even if setup fails or inference/capture stalls. Late serial
  // connections can see the current state without needing to catch boot output.
  for (;;) {
    Log("STATUS", "stage=%s; last_error=%s; free_heap=%u; uptime=%lu s",
        stage.load(), last_error.load(), ESP.getFreeHeap(),
        static_cast<unsigned long>(millis()/1000));
    vTaskDelay(pdMS_TO_TICKS(3000));
  }
}

void BeginConsole() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  // USB Serial/JTAG needs an initial write to establish its connected state.
  // Printing while waiting also avoids dropping the only startup message.
  uint32_t start = millis();
  do {
    Log("BOOT", "Standalone KWS; waiting for USB serial monitor (115200 baud)");
    delay(250);
  } while (!Serial && millis() - start < 5000);
  Log("BOOT", "build=%s %s; reset_reason=%d; CPU=%u MHz; free_heap=%u",
      __DATE__, __TIME__, int(esp_reset_reason()), ESP.getCpuFreqMHz(), ESP.getFreeHeap());
  if (xTaskCreate(StatusTask, "KwsStatus", 3072, nullptr, 1, nullptr) != pdPASS)
    Log("ERROR", "Could not start periodic status task");
}
}  // namespace

#if __has_include("kws_model_data.h")
#include "kws_model_data.h"
#include "kws_test_data.h"
#include <driver/i2s.h>
#include <cmath>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr int kLed = 21;  // Active low.
constexpr size_t kArenaBytes = 100 * 1024;
constexpr int kMicGain = 8;  // Match the working Edge Impulse capture path.
constexpr float kThreshold = 0.50f;
constexpr float kMargin = 0.20f;
alignas(16) uint8_t arena[kArenaBytes];
int16_t pcm[kws::kSamples];
float features[kws::kFeatures];
kws::Frontend frontend;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor *input = nullptr, *output = nullptr;
bool ready = false;

bool Fail(const char* message) {
  last_error.store(message);
  Log("ERROR", "%s (stage=%s)", message, stage.load());
  digitalWrite(kLed, HIGH);
  return false;
}

void PrepareInput(const int16_t* audio) {
  stage.store("DSP: log-mel features");
  uint32_t start = micros();
  frontend.Compute(audio, features);
  for (int i = 0; i < kws::kFeatures; ++i)
    input->data.int8[i] = kws::Quantize(features[i], input->params.scale,
                                      input->params.zero_point);
  Log("DSP", "%d features in %lu ms", kws::kFeatures,
      static_cast<unsigned long>((micros()-start)/1000));
}
bool InvokeModel() {
  stage.store("NN: Invoke");
  uint32_t start = micros();
  TfLiteStatus status = interpreter->Invoke();
  Log("NN", "Invoke status=%d, %lu ms", int(status),
      static_cast<unsigned long>((micros()-start)/1000));
  return status == kTfLiteOk;
}
bool Predict(const int16_t* audio, uint32_t* dsp_us, uint32_t* nn_us) {
  uint32_t start = micros();
  PrepareInput(audio);
  *dsp_us = micros() - start;
  start = micros();
  bool ok = InvokeModel();
  *nn_us = micros() - start;
  return ok;
}
float Score(int i) {
  return (int(output->data.int8[i]) - output->params.zero_point) * output->params.scale;
}
bool InitModel() {
  stage.store("model: schema and operators");
  Log("MODEL", "Loading %u bytes; arena capacity=%u bytes", unsigned(sizeof(kKwsModel)),
      unsigned(sizeof(arena)));
  const tflite::Model* model = tflite::GetModel(kKwsModel);
  if (model->version() != TFLITE_SCHEMA_VERSION) return Fail("Model schema mismatch.");
  static tflite::MicroMutableOpResolver<5> resolver;
  if (resolver.AddConv2D() != kTfLiteOk || resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddReshape() != kTfLiteOk || resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) return Fail("Operator registration failed.");
  static tflite::MicroInterpreter instance(model, resolver, arena, sizeof(arena));
  interpreter = &instance;
  stage.store("model: AllocateTensors");
  Log("MODEL", "Allocating tensors...");
  if (interpreter->AllocateTensors() != kTfLiteOk) return Fail("Arena allocation failed.");
  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || input->dims->size != 4 ||
      input->dims->data[0] != 1 || input->dims->data[1] != kws::kFrames ||
      input->dims->data[2] != kws::kBands || input->dims->data[3] != 1 ||
      output->type != kTfLiteInt8 || output->dims->size != 2 ||
      output->dims->data[0] != 1 || output->dims->data[1] != kKwsLabelCount ||
      input->params.scale <= 0 || output->params.scale <= 0)
    return Fail("Tensor contract mismatch. Re-export from train-kws.ipynb.");
  Log("MODEL", "Input int8 [1,%d,%d,1] scale=%.8f zero=%d; output scale=%.8f zero=%d",
      kws::kFrames, kws::kBands, input->params.scale, input->params.zero_point,
      output->params.scale, output->params.zero_point);
  stage.store("model: parity self-test");
  Log("TEST", "Running exported PCM fixture through frontend and model...");
  PrepareInput(kKwsTestPcm);
  // Check input BEFORE Invoke: the arena planner may reuse its storage for
  // intermediate activations once the first operator has consumed it.
  int max_difference = 0;
  for (int i = 0; i < kws::kFeatures; ++i) {
    int d = abs(int(input->data.int8[i]) - int(kKwsTestInput[i]));
    if (d > max_difference) max_difference = d;
  }
  Log("TEST", "Frontend input max error=%d int8 counts (limit=1)", max_difference);
  if (!InvokeModel()) return Fail("Self-test Invoke failed.");
  float max_score_difference = 0;
  for (int i = 0; i < kKwsLabelCount; ++i)
    max_score_difference = fmaxf(max_score_difference, fabsf(Score(i) - kKwsTestScores[i]));
  Serial.printf("Self-test: input max error=%d counts, score max error=%.5f\n",
                max_difference, max_score_difference);
  if (max_difference > 1 || max_score_difference > 0.03f)
    return Fail("Self-test FAILED. Check frontend/model/runtime versions.");
  Serial.printf("Self-test PASS; arena %u/%u bytes; model %u bytes\n",
                unsigned(interpreter->arena_used_bytes()), unsigned(sizeof(arena)),
                unsigned(sizeof(kKwsModel)));
  return true;
}

bool InitMicrophone() {
  stage.store("microphone: I2S setup");
  Log("AUDIO", "Initializing PDM: clock=GPIO42, data=GPIO41, %d Hz, 16-bit mono",
      kws::kSampleRate);
  i2s_config_t config = {};
  config.mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
  config.sample_rate = kws::kSampleRate;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 4;
  config.dma_buf_len = 512;
  esp_err_t error = i2s_driver_install(I2S_NUM_0, &config, 0, nullptr);
  if (error != ESP_OK) {
    Log("AUDIO", "i2s_driver_install: %s (0x%x)", esp_err_to_name(error), unsigned(error));
    return Fail("I2S driver install failed.");
  }
  i2s_pin_config_t pins = {};
  pins.bck_io_num = I2S_PIN_NO_CHANGE;
  pins.ws_io_num = 42;  // PDM clock, not a conventional I2S word-select signal.
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = 41;
  error = i2s_set_pin(I2S_NUM_0, &pins);
  if (error != ESP_OK) {
    Log("AUDIO", "i2s_set_pin: %s (0x%x)", esp_err_to_name(error), unsigned(error));
    i2s_driver_uninstall(I2S_NUM_0);
    return Fail("I2S pin setup failed.");
  }
  Log("AUDIO", "Microphone ready");
  return true;
}

bool Record() {
  stage.store("audio: drain queued DMA");
  // Discard DMA audio queued during the previous inference. Bounded drain;
  // then capture an immutable 1 s block, avoiding a producer/consumer race.
  int16_t discard[512];
  size_t bytes = 0;
  for (int i = 0; i < 8; ++i) {
    esp_err_t error = i2s_read(I2S_NUM_0, discard, sizeof(discard), &bytes, 0);
    if (error == ESP_ERR_TIMEOUT) break;  // Empty nonblocking drain is normal.
    if (error != ESP_OK) {
      Log("AUDIO", "DMA drain: %s (0x%x)", esp_err_to_name(error), unsigned(error));
      return false;
    }
    if (bytes == 0) break;
  }
  size_t received = 0;
  stage.store("audio: recording 1 second");
  Log("AUDIO", "Capturing %d samples...", kws::kSamples);
  uint32_t start = millis();
  while (received < sizeof(pcm)) {
    const size_t request = min(size_t(1024), sizeof(pcm)-received);
    esp_err_t error = i2s_read(I2S_NUM_0, reinterpret_cast<uint8_t*>(pcm) + received,
                              request, &bytes, pdMS_TO_TICKS(250));
    if (error != ESP_OK || bytes == 0 || (bytes & 1) || bytes > request || millis()-start > 2500) {
      Log("AUDIO", "Read failed: %s; bytes=%u; captured=%u/%u; elapsed=%lu ms",
          esp_err_to_name(error), unsigned(bytes), unsigned(received), unsigned(sizeof(pcm)),
          static_cast<unsigned long>(millis()-start));
      return false;
    }
    const size_t first_sample = received / sizeof(int16_t);
    const size_t sample_count = bytes / sizeof(int16_t);
    for (size_t i = 0; i < sample_count; ++i) {
      const int32_t amplified = int32_t(pcm[first_sample + i]) * kMicGain;
      pcm[first_sample + i] = static_cast<int16_t>(
          amplified < INT16_MIN ? INT16_MIN :
          amplified > INT16_MAX ? INT16_MAX : amplified);
    }
    received += bytes;
  }
  Log("AUDIO", "Captured %u bytes in %lu ms", unsigned(received),
      static_cast<unsigned long>(millis()-start));
  return true;
}
}  // namespace

void setup() {
  pinMode(kLed, OUTPUT);
  digitalWrite(kLed, HIGH);
  BeginConsole();
  Log("BOOT", "Starting model and microphone initialization");
  ready = InitModel() && InitMicrophone();
  if (ready) {
    stage.store("ready");
    Log("READY", "Speak yes/no near the microphone; LED lights for accepted yes");
  } else {
    stage.store("halted: initialization failed");
    Log("ERROR", "Initialization failed; periodic STATUS messages will repeat the error");
  }
}
void loop() {
  if (!ready) { delay(1000); return; }
  if (!Record()) { Fail("Audio read failed; retrying."); delay(1000); return; }
  last_error.store("none");
  int peak = 0;
  double squares = 0, sum = 0;
  for (int16_t sample : pcm) {
    peak = max(peak, abs(int(sample)));
    sum += sample;
    squares += double(sample)*sample;
  }
  uint32_t dsp_us = 0, nn_us = 0;
  if (!Predict(pcm, &dsp_us, &nn_us)) {
    ready = Fail("Invoke failed.");
    stage.store("halted: inference failed");
    return;
  }
  int best = 0;
  for (int i = 1; i < kKwsLabelCount; ++i) if (Score(i) > Score(best)) best = i;
  float second = 0;
  for (int i = 0; i < kKwsLabelCount; ++i) if (i != best) second = fmaxf(second, Score(i));
  bool accepted = Score(best) >= kThreshold && Score(best) - second >= kMargin;
  digitalWrite(kLed, accepted && strcmp(kKwsLabels[best], "yes") == 0 ? LOW : HIGH);
  const double mean = sum/kws::kSamples;
  const double ac_rms = sqrt(fmax(0.0, squares/kws::kSamples - mean*mean));
  Serial.printf("Predictions (DSP: %lu ms., Classification: %lu ms., Anomaly: 0 ms.):\n",
                static_cast<unsigned long>(dsp_us/1000),
                static_cast<unsigned long>(nn_us/1000));
  for (int i = 0; i < kKwsLabelCount; ++i)
    if (i == best) {
      Serial.printf("[*] %s: %.6f\n", kKwsLabels[i], Score(i));
    } else {
      Serial.printf("    %s: %.6f\n", kKwsLabels[i], Score(i));
    }
  Serial.printf("Result: %s (best=%s %.3f, margin=%.3f)\n",
                accepted ? kKwsLabels[best] : "uncertain", kKwsLabels[best],
                Score(best), Score(best)-second);
  Serial.printf("Audio: RMS=%.4f, AC_RMS=%.4f, DC=%.4f, peak=%d\n\n",
                sqrt(squares/kws::kSamples)/32768, ac_rms/32768, mean/32768, peak);
  if (peak == 0) Log("WARN", "All microphone samples are zero; check Sense board connection");
  else if (peak >= 32760) Log("WARN", "Audio is clipping; reduce sound level or increase distance");
  stage.store("ready");
}
#else
// A clean checkout can build before training, but never runs an untrained model.
void setup() {
  BeginConsole();
  stage.store("halted: no exported model");
  last_error.store("Run train-kws.ipynb Export, then rebuild/upload");
}
void loop() {
  Serial.println("No KWS model. Run train-kws.ipynb through Export, then rebuild/upload.");
  delay(3000);
}
#endif
