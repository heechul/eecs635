# Lab 4: Standalone keyword spotting

Train a keyword spotting (KWS) model locally and run it on a **Seeed Studio XIAO ESP32S3 Sense**. The four classes from the original lab are **yes**, **no**, **unknown** (other words), and **noise** (background sound or silence). Training, feature extraction, quantization and firmware export are under your control; no Edge Impulse account, service or SDK is needed.

You will collect/inspect audio, train a small CNN, measure float versus int8 accuracy, and deploy a model with matching preprocessing on the microcontroller. This version uses log-mel energies instead of the original lab's MFCCs. It keeps the same four-class task while making every preprocessing step explicit.

## What you need

- XIAO ESP32S3 **Sense**, with its microphone expansion board attached, and a USB data cable. The base board alone has no microphone.
- VS Code with PlatformIO, or the PlatformIO CLI (`pio`).
- Python **3.12**, a local Jupyter kernel, and a machine capable of running TensorFlow. CPU training is supported; GPU setup is optional.
- Internet for the initial packages, firmware dependencies, and public dataset. No network is needed for training after these are cached, or for board inference.
- About **1 GB of free disk space**. Mini Speech Commands is approximately a 182 MB ZIP with 8,000 recordings across eight words; extraction and artifacts need additional space. Training time depends on your machine and sample cap.

## 1. Set up Python

Open a terminal **in `l4-kws`**:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name kws-local --display-name "Python 3.12 (KWS)"
jupyter lab train-kws.ipynb
```

On Windows, activate with `.venv\Scripts\activate` (or `.venv\Scripts\Activate.ps1` in PowerShell). You can also open the notebook in VS Code and select `.venv` as its kernel. Use `requirements.txt` for this lab; the existing `requirements-local.txt` is not required.

TensorFlow 2.19.1 and legacy Keras 2.19.0 are the supported baseline. The notebook sets `TF_USE_LEGACY_KERAS=1` before importing TensorFlow; restart the kernel if you imported TensorFlow earlier. Python 3.14 is not the baseline. If your OS has no compatible TensorFlow wheel, use a supported Python 3.12 environment on Linux, macOS Apple Silicon, or Windows rather than changing the package versions individually.

## 2. Prepare data and train

Open [train-kws.ipynb](train-kws.ipynb), review the configuration, then run its cells in order.

1. **Download/split:** the notebook downloads and extracts **mini Speech Commands** to `data/mini_speech_commands_v1`. To reuse an extracted copy, change `DATA_DIR`. The loader rejects extra word folders or more than 1,000 WAVs per word, so an old full dataset cannot silently be used as mini. Existing data in `data/mini_speech_commands` is preserved. It keeps yes/no and maps down/go/left/right/stop/up to unknown. Mini has no official split lists, so a seeded SHA-1 hash of each speaker ID assigns approximately 80% of speakers to training, 10% to validation and 10% to test. Every recording from a speaker stays in one split.
2. **Inspect features:** listen to examples and view their log-mel images. The default cap is 1,000 training clips per class and up to 250 validation and test clips per class; actual counts are lower because each yes/no directory has only 1,000 recordings before splitting. All four classes are balanced to the smallest available speech class in each split. Unknown is sampled from the six other word directories.
3. **Train:** two Conv2D/MaxPool blocks, dropout, and a four-class softmax. Start with 25 epochs; early stopping restores the weights with the best validation loss. Training uses random shifts, gain changes, and generated noise mixing sampled once when the features are built.
4. **Quantize/evaluate:** representative **training** features calibrate full int8 conversion. Compare float/int8 model sizes, test accuracies, per-class recall and confusion matrices. Do not use test recordings to tune architecture or decision thresholds. No target accuracy is assumed.
5. **Export:** the last export cell writes both firmware headers directly into `src`. An optional cell classifies your own WAV locally.

**Noise class:** mini Speech Commands does not include background recordings. The default lab generates independent white noise, brown noise, mains hum and silence for each split, preserving the four-class model without requiring the full dataset. Synthetic noise performance does not measure rejection of real room sounds. Add your own noise WAVs under `data/extra/<split>/noise/` and test on real recordings. Optionally put at least three distinct background WAVs in `DATA_DIR/_background_noise_`; the loader reserves whole source recordings for separate splits before cropping and uses those instead of generated noise.

The notebook saves the Keras model, float/int8 TFLite models, split manifest, parity fixture, and `metadata.json` in `artifacts/`. Metadata includes settings, label order, model SHA256, quantization parameters and evaluation results. Generated data/models are ignored by Git. Keep your notebook outputs and metadata for the lab report. Generated headers are also ignored; regenerate them by running the notebook before deployment.

The download cell reloads the helper module on every run, displays the exact URL, and verifies the mini ZIP's SHA-256. It requests only `https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip`. After updating an already-open notebook, reload it from disk and restart the kernel before Run All. An older kernel can retain a previous downloader function even after its Python file changes. A previously misplaced full dataset is preserved here as `data/speech_commands_full_v0.02`; its old TAR cache is not used by the mini downloader.

### Optional: add your own recordings

Use one-second **mono signed 16-bit PCM WAVs at 16 kHz**. The loader rejects incompatible formats instead of silently resampling them. Use this layout:

```text
data/extra/
  train/yes/alice_nohash_0.wav
  train/no/alice_nohash_1.wav
  validation/yes/bob_nohash_0.wav
  test/yes/carol_nohash_0.wav
```

All split directories may contain `yes`, `no`, `unknown` and `noise` folders. They supplement the public dataset. Use a stable speaker ID before `_nohash_` and keep each person in a single split across all labels; the loader checks this before subsampling. Keep custom noise recording sessions separate too. Extra samples enter the capped pools, so not every extra recording will necessarily be selected. Rebuild the feature arrays and retrain after adding data.

Reserve recordings from new people and environments for your final assessment. Laptop and board microphones differ; adding actual board recordings to training can help. A useful first experiment is to compare desktop WAV classification with live board results at several distances.

## 3. Build and upload

From the same project directory:

```sh
pio run
pio run -t upload
pio device monitor -b 115200
```

Or use PlatformIO's Build, Upload, and Serial Monitor buttons in VS Code. If multiple serial devices are connected, select the board with `pio device list` and add `--upload-port <port>` / `--port <port>` to the appropriate command. Close the serial monitor before uploading. If necessary, hold BOOT while connecting USB to enter download mode, then reset after upload.

The project pins PlatformIO Espressif32 6.13.0 (Arduino ESP32 2.0.17) and uses the local TensorFlow Lite Micro runtime in `lib/ESP_TF`, including its ESP-NN kernels. Keep that library directory with the project. Its interpreter API uses four constructor arguments and does not use the older `MicroErrorReporter` class. The initial build downloads the PlatformIO platform/toolchain if needed. Do not copy an Edge Impulse export into `src`; its SDK is no longer part of this project. If migrating a previously populated checkout, remove your old generated `edge-impulse-sdk`, `model-parameters` and `tflite-model` directories first.

A clean checkout builds a diagnostic program that prints `No KWS model...`. **It does not contain a pretrained model.** Run the notebook's Export cell and rebuild/upload to enable inference. The generated files must stay together:

| File | Purpose |
| --- | --- |
| `src/kws_model_data.h` | Aligned int8 model bytes and label order |
| `src/kws_test_data.h` | Held-out PCM clip, expected input tensor and scores |
| `src/kws_features.h/.cpp` | Portable preprocessing, shared contract with Python |
| `src/main.cpp` | Microphone capture, self-test, inference and LED |
| `tools/kws_training.py` | Data splitting, Python frontend and export helpers |

## 4. Verify and test on the board

Open the monitor and press RESET to see startup output. The board first checks model schema and tensor shape/type, then runs the exported PCM fixture through the **C++ frontend and TFLite Micro**. Expect a `Self-test PASS` line with the tensor arena/model sizes. A feature discrepancy greater than one int8 count or a score discrepancy greater than 0.03 stops inference. This compares desktop and embedded processing; it does not measure recognition accuracy.

The feature comparison happens **before** `Invoke()`: TFLite Micro may reuse the input tensor's arena storage for intermediate activations during inference. Comparing that storage after inference produces a false self-test failure.

Diagnostic lines include uptime and a tag: `BOOT`, `MODEL`, `TEST`, `AUDIO`, `DSP`, `NN`, `READY`, or `ERROR`. A separate task prints `STATUS` every three seconds with the current stage, last error and free heap, including after initialization fails. You can therefore open the monitor after boot and still see why inference stopped. Audio reads have a deadline and report driver errors and the captured byte count rather than waiting indefinitely.

After the self-test, speak a keyword clearly near the microphone. Each serial result uses the Edge Impulse-style `Predictions` header with DSP/classification timings and one indented score per label, followed by the selected class (or `uncertain`) and audio diagnostics. A result is accepted when its best score is at least **0.75** and at least **0.20** above the runner-up. The active-low built-in LED on GPIO21 lights for an accepted **yes** and is off for all other outcomes. Labels are matched by name, not a fragile class index.

The log also shows the raw winning class, winning margin, AC RMS (signal variation after excluding the mean) and DC level. A high total RMS with low AC RMS can primarily be microphone bias rather than speech. `uncertain` is a rejection by the confidence/margin rule, not an additional trained class; inspect the raw scores before changing thresholds. Low held-out accuracy needs a better model/data pipeline, not just a lower threshold. Match DC removal or gain normalization in training and firmware if adding either preprocessing step, and regenerate the parity fixture.

The frontend precomputes sparse mel-filter weights and FFT stage constants once. This avoids recalculating roughly 403,000 mostly-zero filter weights per one-second window. On the tested 240 MHz XIAO ESP32S3, this reduced DSP from 479 ms to about 36 ms while retaining Python/firmware parity; the existing model's inference took about 10 ms. This optimization does not change the trained features or require retraining.

Capture uses GPIO42 as PDM clock and GPIO41 as PDM data, at 16 kHz/16 bits. Samples are amplified 8x, matching the working Edge Impulse capture path, with int16 saturation to prevent wraparound. Audio buffers remain unchanged while preprocessing/inference runs. This introductory implementation captures **non-overlapping one-second blocks**, discards queued audio after inference, and has gaps during processing. It is not a continuous streaming detector: words that straddle windows can be missed, and latency includes the capture period. Say each word repeatedly with pauses; streaming/overlapping inference is an extension exercise.

Try at least 20 yes and 20 no utterances, other words, silence, and background noise. Record missed keywords and false yes/no triggers. Repeat at several distances. Choose thresholds using validation/room trials, then record final held-out results. Change `kThreshold`/`kMargin` in `src/main.cpp` and the corresponding notebook values together.

### Troubleshooting

- **Blank monitor:** use the board's USB serial port at 115200 baud (check `pio device list`; on macOS it is usually `/dev/cu.usbmodem...`). For example, run `pio device monitor --port <port> -b 115200`, then press RESET. The project explicitly enables USB Serial/JTAG CDC with DTR asserted and RTS deasserted in the monitor. Close other serial applications before uploading or monitoring. A running diagnostic build prints a `STATUS` line every three seconds, even after model setup fails; check its stage and last error. An editor terminal alone is not a serial monitor.
- **Last stage repeats without progress:** `model: AllocateTensors`, `DSP: log-mel features`, or `NN: Invoke` identifies the operation that stalled. Keep the last `MODEL`, `TEST` and `STATUS` lines when reporting the issue. The build uses `-O2` rather than fast-math optimization to preserve frontend parity.
- **No model message:** confirm the notebook generated both headers in this checkout's `src`, then rebuild and upload.
- **`micro_error_reporter.h` missing:** the firmware and runtime APIs do not match. With the local `lib/ESP_TF` runtime, omit that include and the `MicroErrorReporter` object, and construct `MicroInterpreter(model, resolver, arena, sizeof(arena))`. Retraining or re-exporting the model is not needed for this API fix.
- **Self-test failed:** re-export both headers from the same notebook run; verify the frontend settings and pinned runtime. Do not simply disable the check. A new model operator requires a matching firmware resolver registration.
- **Arena allocation failed:** the default is 100 KiB in internal RAM. Reduce the network first, or adjust `kArenaBytes` after checking available RAM. Compile-time size reports do not prove runtime arena allocation will succeed.
- **No audio / nearly zero RMS:** check that the Sense expansion board is seated and its microphone connections are intact. This code uses the Arduino 2.x/ESP-IDF 4.x I2S driver; Arduino 3.x examples use different APIs.
- **Clipping / high peaks near 32767:** lower the sound level or increase distance. Capture applies a saturating 8x gain to match the working Edge Impulse path; quiet input and domain differences may still require new recordings and retraining.
- **Good desktop accuracy, poor board recognition:** inspect live audio levels, use the optional own-WAV test, test different speakers/rooms, and add representative board recordings. The parity fixture checks numerical consistency, not microphone quality or dataset coverage.

## 5. Submit your lab report

Include the dataset/split counts, separation of speakers and noise sources, model summary, training curves, float/int8 sizes and test accuracies, confusion matrix, per-class recall, board self-test/arena output, DSP+NN latency, and live trial results. Explain quantization's effect and describe one controlled experiment changing data, architecture, or thresholds. Discuss why held-out WAV accuracy may differ from real-room behavior and how block boundaries affect detection.

## Development checks

With the Python environment active and a C++11 compiler installed:

```sh
python tools/check_frontend.py
python tools/check_dataset_download.py
python tools/smoke_notebook.py /tmp/kws-smoke
```

The frontend check compares the actual firmware frontend against Python on silence, tones, low/full-scale random signals, DC and an impulse, including int8 rounding/clamping. The download check runs offline and verifies that the notebook bypasses a stale cached helper, requests the mini ZIP URL and checksum, and rejects full-dataset folders. The notebook smoke test executes **every notebook cell** on a small generated WAV dataset, including splitting, one training epoch, conversion, evaluation and header export. It keeps outputs in the requested scratch directory; synthetic accuracy is not a recognition benchmark. Training on the actual mini dataset and physical microphone/LED tests remain necessary for lab results. On Windows, replace `/tmp/kws-smoke` with a suitable scratch path.

## References and attribution

- [Original lab instructions](https://docs.google.com/document/d/1OadHig8dEW04-TyRh5z_GMDcjAB8ApQualodpi2gn5M/edit): EECS 690/700 EmbeddedML Lab 4, KWS. This replaces its Edge Impulse training/export steps.
- [Mini Speech Commands](https://www.tensorflow.org/tutorials/audio/simple_audio), the eight-word subset used here, derived from [Speech Commands](https://arxiv.org/abs/1804.03209), Pete Warden, 2018; dataset distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve the dataset's included license and attribution. Background source attributions are included with the dataset.
- [TensorFlow simple audio recognition tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio).
- [Full integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant).
- [Seeed XIAO ESP32S3 Sense microphone documentation](https://wiki.seeedstudio.com/xiao_esp32s3_sense_mic/).
- The original lab credited Marcelo Rovai's [TinyML Made Easy: Keyword Spotting](https://www.hackster.io/mjrobot/tinyml-made-easy-keyword-spotting-kws-5fa6e7).
