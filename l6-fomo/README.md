# Lab 6: FOMO cube detection

Train a FOMO centroid detector with a truncated MobileNetV2 backbone and deploy it on a **Seeed XIAO ESP32S3 Sense**. The notebook and firmware use RGB 96×96 input, a 12×12 heatmap, and five classes: background, blue, green, red, yellow.

Start with [train-fomo.ipynb](train-fomo.ipynb). It generates synthetic conveyor scenes, trains a fully convolutional model, measures held-out object detection performance, exports full int8 TFLite, and regenerates the firmware assets. Dataset generation works offline; no Kaggle or Edge Impulse account is required. The Python helpers are in [fomo_lab.py](fomo_lab.py).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
jupyter lab train-fomo.ipynb
```

Run every cell in order from this directory. For Colab setup, see notebook section 1. Training duration and numerical results depend on the machine. The checked-in model and executed notebook provide a working synthetic baseline.

Training also saves `model/fomo.weights.h5` and `model/training-history.csv` alongside the deployable TFLite models.

## Firmware

Install PlatformIO, connect the board, and first run the static-image check:

```sh
pio run -e seeed_xiao_esp32s3_static
pio run -e seeed_xiao_esp32s3_static -t upload
pio device monitor -b 115200
```

Compare printed detections with [model/static-expected.json](model/static-expected.json) and [model/static-input.png](model/static-input.png). This runs the generated test input through TFLite Micro without using the camera. Minor differences between desktop and ESP-NN kernels are possible; compare classes/counts and positions within one grid cell.

For camera input:

```sh
pio run -e seeed_xiao_esp32s3
pio run -e seeed_xiao_esp32s3 -t upload
pio device monitor -b 115200
```

Use overhead views of separated colored cubes on a dark background, roughly 10–16 pixels wide in the 96×96 camera frame. Firmware expands the high-byte-first RGB565 camera data into RGB, normalizes to `[0,1]`, then quantizes using the input tensor's scale and zero point. Coordinates use the full 96×96 frame, origin at top left. Serial reports per-frame class counts and centroids; the active-low LED lights when a detection is present. This does not track unique objects across frames.

The integer model uses Conv2D, DepthwiseConv2D, Add and Softmax operators; batch normalization and activations are folded/fused during conversion. The firmware first requests a 280 KiB tensor arena in internal SRAM with 16-byte alignment, leaving about 16 KiB above the measured 270,460-byte use. It allocates the arena before camera initialization and prints free internal memory, the largest free block, and actual arena use at startup. The allocation requires one contiguous internal block; if unavailable, the firmware falls back to a 16-byte-aligned PSRAM arena. The startup message reports the actual memory used. If both allocations fail, inference stops with a diagnostic. The camera framebuffer uses PSRAM. Recheck the arena budget after model or kernel changes, and compare serial inference timings on the board to measure the SRAM speedup. Shape, int8 dtype, schema and quantization are checked before inference. Export `model.cc`, `model_config.h` and `img.h` together; `model.cc` is compiled once via declarations in `model.h`.

## Detection and limits

The model uses EI's FOMO backbone/head architecture: MobileNetV2 α=0.35 cut at `block_6_expand_relu` (12×12×96 features), then a 1×1 Conv(32, ReLU), 1×1 Conv(5), and per-cell softmax. MobileNetV2 adds inverted residual blocks, depthwise convolutions and residual connections. We train all retained layers from scratch (`weights=None`) on synthetic data, with batch-normalization momentum 0.9 for moving-statistic adaptation. Inputs remain normalized to `[0,1]`. This does not load EI's pretrained checkpoint; the loss and decoder are local teaching implementations, and no Edge Impulse SDK is required. See [Edge Impulse's FOMO documentation](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo) for the original method and [Cubes on conveyor belt](https://studio.edgeimpulse.com/public/497627/latest) for an optional real dataset.

Each grid cell predicts one class. The decoder thresholds the winning foreground probability at 0.60 and groups same-class 4-connected cells. Each component yields a confidence-weighted cell-center location and peak confidence. These locations do not estimate object dimensions. Same-cell centroids cannot be separated, and touching same-class components merge. Synthetic scenes keep centers at least 24 pixels apart.

The notebook reports float/int8 precision, recall and F1 using one-to-one same-class matching within 8 input pixels, plus per-frame class count error. See [model/metrics.json](model/metrics.json) for measured synthetic results. Separate seeds generate training, validation and test sets; only training images calibrate quantization. Synthetic performance does not establish real-camera accuracy. Collect representative annotated camera images and retrain before relying on live detections; notebook section 10 describes the dataset extension exercise.

## Host verification

```sh
.venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
```

The tests check centroid encoding, int8 tensor/export consistency, detection matching, and Python/C++ preprocessing and decoder agreement. A C++ compiler (`c++`) is required. Build both PlatformIO environments as well. Physical camera behavior, on-board latency and TFLite Micro output must be checked on the connected board using the static and live procedures above.
