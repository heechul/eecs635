#pragma once
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

class NeuralNetwork {
public:
    bool begin();
    bool ready() const { return ready_; }
    TfLiteTensor* getInput() { return input_; }
    TfLiteTensor* getOutput() { return output_; }
    TfLiteStatus predict();

private:
    bool ready_ = false;
    uint8_t* arena_ = nullptr;
    tflite::MicroInterpreter* interpreter_ = nullptr;
    TfLiteTensor* input_ = nullptr;
    TfLiteTensor* output_ = nullptr;
};
