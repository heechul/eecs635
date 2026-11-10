"""Compare actual portable C++ frontend against Python; requires a C++11 compiler."""
from pathlib import Path
import os
import subprocess
import tempfile
import numpy as np
from kws_training import frontend, SAMPLES, SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory(prefix='kws-frontend-') as tmp:
        tmp = Path(tmp)
        (tmp/'runner.cpp').write_text(r'''
#include "kws_features.h"
#include <cstdio>
int main() {
  int16_t pcm[kws::kSamples];
  float output[kws::kFeatures];
  if (std::fread(pcm, sizeof(int16_t), kws::kSamples, stdin) != kws::kSamples) return 1;
  kws::Frontend frontend;
  frontend.Compute(pcm, output);
  if (kws::Quantize(0.0f, 0.01f, -128) != -128 ||
      kws::Quantize(1.0f, 0.01f, -128) != -28 ||
      kws::Quantize(10.0f, 0.01f, 0) != 127 ||
      kws::Quantize(-10.0f, 0.01f, 0) != -128) return 2;
  return std::fwrite(output, sizeof(float), kws::kFeatures, stdout) == kws::kFeatures ? 0 : 3;
}
''')
        exe = tmp/'frontend'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++11', '-O2', '-I', str(ROOT/'src'),
                        str(tmp/'runner.cpp'), str(ROOT/'src/kws_features.cpp'), '-o', str(exe)], check=True)
        rng = np.random.default_rng(4)
        t = np.arange(SAMPLES)/SAMPLE_RATE
        cases = {
            'silence': np.zeros(SAMPLES, np.int16),
            'sine': (12000*np.sin(2*np.pi*440*t)).astype(np.int16),
            'quiet': rng.integers(-20, 21, SAMPLES, dtype=np.int16),
            'random': rng.integers(-32768, 32768, SAMPLES, dtype=np.int16),
            'dc': np.full(SAMPLES, -32768, np.int16),
            'impulse': np.zeros(SAMPLES, np.int16),
        }
        cases['impulse'][8000] = 32767
        for name, pcm in cases.items():
            actual = subprocess.run([str(exe)], input=pcm.tobytes(), capture_output=True, check=True)
            actual = np.frombuffer(actual.stdout, np.float32).reshape(49, 32, 1)
            expected = frontend(pcm)
            error = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=0)
            # Match the model's typical [0,1] int8 scale and zero point too.
            q = lambda x: np.clip(np.floor(x*255+0.5)-128, -128, 127).astype(np.int16)
            assert np.max(np.abs(q(actual)-q(expected))) <= 1
            print(f'{name}: PASS, max feature error {error:.8f}')

if __name__ == '__main__':
    main()
