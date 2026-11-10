"""Data, frontend and export helpers for train-kws.ipynb (no Edge Impulse)."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import wave
import numpy as np

LABELS = ('yes', 'no', 'unknown', 'noise')
MINI_WORDS = ('yes', 'no', 'down', 'go', 'left', 'up', 'right', 'stop')
MINI_ARCHIVE = 'mini_speech_commands.zip'
MINI_DATASET_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/' + MINI_ARCHIVE
MINI_SHA256 = '49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0'
SAMPLE_RATE, SAMPLES, WINDOW, HOP, FFT, BANDS, FRAMES = 16000, 16000, 480, 320, 512, 32, 49
FRONTEND = dict(sample_rate=SAMPLE_RATE, samples=SAMPLES, window=WINDOW, hop=HOP,
                fft=FFT, bands=BANDS, frames=FRAMES, lower_hz=80, upper_hz=7600,
                window_type='periodic Hann', spectrum='unnormalized power',
                feature='clip((ln(max(mel_energy, 1e-7)) + 16) / 24, 0, 1)')


def read_pcm(path):
    """Require the same signed 16-bit mono PCM as the board; never silently resample."""
    with wave.open(str(path), 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) != (1, 2, SAMPLE_RATE, 'NONE'):
            raise ValueError(f'{path}: expected mono 16-bit PCM WAV at 16000 Hz')
        return np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').copy()


def one_second(pcm):
    # Speech Commands recordings are <=1 s; own longer clips are center cropped.
    start = max(0, (len(pcm) - SAMPLES)//2)
    result = np.zeros(SAMPLES, np.int16)
    count = min(len(pcm), SAMPLES)
    result[:count] = pcm[start:start+count]
    return result


def frontend(pcm):
    """Exact contract implemented in src/kws_features.cpp. No MFCC/DCT or DSP in TFLite."""
    pcm = np.asarray(pcm, dtype=np.int16)
    if pcm.shape != (SAMPLES,):
        raise ValueError(f'Expected {SAMPLES} PCM samples, got {pcm.shape}')
    frames = np.lib.stride_tricks.sliding_window_view(pcm.astype(np.float32)/32768, WINDOW)[::HOP]
    window = (0.5 - 0.5*np.cos(2*np.pi*np.arange(WINDOW)/WINDOW)).astype(np.float32)
    power = np.abs(np.fft.rfft(frames*window, n=FFT))**2
    mel = lambda hz: 2595*np.log10(1 + hz/700)
    edges = 700*(10**(np.linspace(mel(80), mel(7600), BANDS+2)/2595) - 1)
    hz = np.arange(FFT//2+1)*SAMPLE_RATE/FFT
    weights = np.maximum(0, np.minimum((hz[:, None]-edges[:-2])/(edges[1:-1]-edges[:-2]),
                                      (edges[2:]-hz[:, None])/(edges[2:]-edges[1:-1])))
    result = np.clip((np.log(np.maximum(power @ weights, 1e-7))+16)/24, 0, 1)
    return result.astype(np.float32)[..., None]


def validate_mini_dataset(root):
    """Reject a full Speech Commands tree accidentally placed in the mini folder."""
    root = Path(root)
    counts = {p.name: len(list(p.glob('*.wav'))) for p in root.iterdir()
              if p.is_dir() and not p.name.startswith(('_', '.'))} if root.exists() else {}
    unexpected = {word: count for word, count in counts.items() if count and word not in MINI_WORDS}
    oversized = {word: count for word, count in counts.items() if count > 1000}
    if unexpected or oversized:
        raise ValueError(
            f'{root} is not mini Speech Commands: unexpected words={list(unexpected)}, '
            f'folders with >1000 clips={oversized}. Use a new empty DATA_DIR and let '
            'the notebook download mini_speech_commands.zip; keep existing data separately.')
    return counts


def download_dataset(root):
    """Download the ~182 MB mini Speech Commands ZIP; no full dataset required."""
    import tensorflow as tf
    root = Path(root).resolve()
    counts = validate_mini_dataset(root)
    if all(counts.get(word) == 1000 for word in MINI_WORDS):
        print(f'Using verified mini dataset: {root} (8 words, 1000 WAVs each)')
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = tf.keras.utils.get_file(
        MINI_ARCHIVE, origin=MINI_DATASET_URL,
        file_hash=MINI_SHA256, hash_algorithm='sha256',
        cache_dir=str(root.parent), cache_subdir='downloads')
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            prefix = 'mini_speech_commands/'
            if not member.filename.startswith(prefix):
                continue  # Skip archive metadata such as __MACOSX.
            relative = member.filename[len(prefix):]
            target = (root/relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Unsafe path in dataset archive')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    counts = validate_mini_dataset(root)
    if not all(counts.get(word) == 1000 for word in MINI_WORDS):
        raise ValueError('Incomplete mini Speech Commands archive')
    return root


def speaker_split(speaker, seed=42):
    """Stable approximately 80/10/10 split by speaker, shared across all words."""
    bucket = int(hashlib.sha1(f'{seed}:{speaker}'.encode()).hexdigest(), 16) % 100
    return 'test' if bucket < 10 else 'validation' if bucket < 20 else 'train'


def split_records(root, cap=1000, seed=42, extra=None):
    """Speaker-disjoint mini dataset split; no official split lists are present.

    cap limits training examples/class; val/test caps are cap//4. Generated
    noise fills the fourth class unless custom noise recordings are supplied.
    """
    root = Path(root)
    validate_mini_dataset(root)
    records = {s: {label: [] for label in LABELS} for s in ('train', 'validation', 'test')}
    for path in sorted(root.glob('*/*.wav')):
        if path.parent.name.startswith('_'):
            continue
        if '_nohash_' not in path.stem:
            raise ValueError(f'{path}: expected <speaker>_nohash_<take>.wav')
        split = speaker_split(path.stem.split('_nohash_')[0], seed)
        label = path.parent.name if path.parent.name in ('yes', 'no') else 'unknown'
        records[split][label].append(path)
    # Audit BEFORE subsampling: never split recordings from one speaker across sets.
    speakers = {s: {p.stem.split('_nohash_')[0] for label in LABELS for p in records[s][label]}
                for s in records}
    if extra is not None and Path(extra).exists():
        for s in records:
            for label in LABELS:
                additions = sorted((Path(extra)/s/label).glob('*.wav'))
                if any('_nohash_' not in p.stem for p in additions):
                    raise ValueError('Own WAV names must use <speaker>_nohash_<take>.wav')
                records[s][label].extend(additions)
                speakers[s].update(p.stem.split('_nohash_')[0] for p in additions)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if speakers[a] & speakers[b]:
            raise ValueError(f'Speaker leakage between {a} and {b}')
    noise = sorted((root/'_background_noise_').glob('*.wav'))
    if noise and len(noise) < 3:
        raise ValueError('Optional _background_noise_ needs >=3 distinct source recordings')
    # Reserve whole noise recordings, not neighboring crops of the same source.
    sources = {'train': noise[2:], 'validation': noise[1:2], 'test': noise[:1]}
    rng = np.random.default_rng(seed)
    for s, classes in records.items():
        limit = cap if s == 'train' else max(1, cap//4)
        count = min(limit, len(classes['yes']), len(classes['no']), len(classes['unknown']))
        if count == 0:
            raise ValueError(f'{s}: missing speech classes')
        for label in ('yes', 'no', 'unknown'):
            paths = classes[label]
            classes[label] = [paths[i] for i in rng.permutation(len(paths))[:count]]
        # None requests independently generated noise (or an optional background crop).
        own_noise = classes['noise']
        classes['noise'] = own_noise[:count] + [None]*max(0, count-len(own_noise))
    return records, sources


def make_arrays(records, sources, seed=42):
    """Materialize balanced features. Training augmentation is sampled once per run."""
    rng = np.random.default_rng(seed)
    arrays, manifest = {}, {}
    for split, classes in records.items():
        backgrounds = [(p, read_pcm(p)) for p in sources[split]]
        if any(len(pcm) < SAMPLES for _, pcm in backgrounds):
            raise ValueError('Background recordings must be at least one second')
        def noise_crop():
            if not backgrounds:
                kind = rng.choice(('white', 'brown', 'hum'))
                signal = rng.standard_normal(SAMPLES)
                if kind == 'brown':
                    signal = np.cumsum(signal)
                    signal -= signal.mean()
                elif kind == 'hum':
                    frequency = rng.choice((50, 60))*rng.uniform(0.98, 1.02)
                    phase = rng.uniform(0, 2*np.pi)
                    signal = 0.1*signal + np.sin(2*np.pi*frequency*np.arange(SAMPLES)/SAMPLE_RATE + phase)
                signal *= rng.uniform(0.01, 0.1)/max(np.sqrt(np.mean(signal**2)), 1e-8)
                return signal.astype(np.float32), f'synthetic:{kind}', 0
            path, pcm = backgrounds[rng.integers(len(backgrounds))]
            offset = int(rng.integers(len(pcm)-SAMPLES+1))
            return pcm[offset:offset+SAMPLES].astype(np.float32)/32768, str(path), offset
        xs, ys, audio, entries = [], [], [], []
        for label_index, label in enumerate(LABELS):
            for path in classes[label]:
                if path is None:
                    signal, source, offset = noise_crop()
                    # Vary noise level and include digital silence.
                    signal *= rng.uniform(0.02, 0.3)
                    if len(xs) % 10 == 0:
                        signal[:] = 0
                        source = "synthetic:silence"
                else:
                    signal = one_second(read_pcm(path)).astype(np.float32)/32768
                    source, offset = str(path), 0
                if split == 'train':
                    # Zero-filled shift (never circular wrap) and random recording level.
                    shift = int(rng.integers(-1600, 1601))
                    shifted = np.zeros_like(signal)
                    if shift >= 0:
                        shifted[shift:] = signal[:SAMPLES-shift]
                    else:
                        shifted[:shift] = signal[-shift:]
                    signal = shifted*rng.uniform(0.5, 1.5)
                    if label != 'noise':
                        bg, _, _ = noise_crop()
                        signal += bg*rng.uniform(0, 0.15)
                pcm = np.clip(np.rint(signal*32768), -32768, 32767).astype(np.int16)
                xs.append(frontend(pcm)); ys.append(label_index); audio.append(pcm)
                entries.append(dict(label=label, source=source, offset=offset))
        order = rng.permutation(len(xs))
        arrays[split] = (np.asarray(xs)[order], np.asarray(ys, np.int64)[order], np.asarray(audio)[order])
        manifest[split] = [entries[i] for i in order]
        print(split, arrays[split][0].shape, dict(zip(LABELS, np.bincount(arrays[split][1], minlength=4))))
    return arrays, manifest


def quantize(x, detail):
    scale, zero = detail['quantization']
    if scale <= 0 or detail['dtype'] != np.int8:
        raise ValueError('Expected a quantized int8 tensor')
    # C++ roundf rounds halfway away from zero. Features are nonnegative.
    return np.clip(np.floor(x/scale+0.5)+zero, -128, 127).astype(np.int8)


def run_tflite(model, x):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_content=model, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if list(inp['shape']) != [1, FRAMES, BANDS, 1] or list(out['shape']) != [1, len(LABELS)]:
        raise ValueError('Tensor shape contract changed')
    scores = []
    for row in x:
        tensor = quantize(row[None], inp) if inp['dtype'] == np.int8 else row[None]
        interpreter.set_tensor(inp['index'], tensor)
        interpreter.invoke()
        value = interpreter.get_tensor(out['index'])[0]
        if out['dtype'] == np.int8:
            scale, zero = out['quantization']
            value = (value.astype(np.float32)-zero)*scale
        scores.append(value)
    return np.asarray(scores), inp, out


def export_headers(model, labels, pcm, destination):
    """Export the model, label order and a PCM->int8->score golden test together."""
    from tensorflow.lite.python import schema_py_generated as schema
    graph = schema.Model.GetRootAsModel(model, 0)
    allowed = {schema.BuiltinOperator.CONV_2D, schema.BuiltinOperator.MAX_POOL_2D,
               schema.BuiltinOperator.RESHAPE, schema.BuiltinOperator.FULLY_CONNECTED,
               schema.BuiltinOperator.SOFTMAX}
    for i in range(graph.OperatorCodesLength()):
        code = graph.OperatorCodes(i)
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        if builtin not in allowed:
            raise ValueError(f'Unsupported firmware operator: {builtin}; update resolver before exporting')
    scores, inp, out = run_tflite(model, frontend(pcm)[None])
    if inp['dtype'] != np.int8 or out['dtype'] != np.int8:
        raise ValueError('Firmware requires int8 input AND output')
    def array(ctype, name, values, width=16, alignment=''):
        values = list(values)
        rows = [', '.join(str(v) for v in values[i:i+width]) for i in range(0, len(values), width)]
        return f'{alignment}const {ctype} {name}[] = {{\n  ' + ',\n  '.join(rows) + '\n};\n'
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(model).hexdigest()
    text = f'// Generated by train-kws.ipynb. Model SHA256: {digest}\n#pragma once\n#include <cstdint>\n'
    text += array('unsigned char', 'kKwsModel', (f'0x{v:02x}' for v in model), alignment='alignas(16) ')
    text += f'constexpr int kKwsLabelCount = {len(labels)};\n'
    text += 'const char* const kKwsLabels[] = {' + ', '.join(json.dumps(s) for s in labels) + '};\n'
    test = '// Generated PCM -> feature tensor -> scores parity fixture.\n#pragma once\n#include <cstdint>\n'
    test += array('int16_t', 'kKwsTestPcm', pcm)
    test += array('int8_t', 'kKwsTestInput', quantize(frontend(pcm), inp).ravel())
    test += array('float', 'kKwsTestScores', (f'{float(s):.9f}f' for s in scores[0]))
    (destination/'kws_model_data.h').write_text(text)
    (destination/'kws_test_data.h').write_text(test)
    return digest
