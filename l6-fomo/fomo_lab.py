"""Reusable helpers for train-fomo.ipynb; no network or dataset credentials needed."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

INPUT_W = INPUT_H = 96
STRIDE = 8
GRID_W = GRID_H = 12
CLASS_NAMES = ('background', 'blue', 'green', 'red', 'yellow')
COLORS = np.array([(40, 90, 220), (45, 190, 65), (220, 45, 40), (230, 205, 35)])
THRESHOLD = 0.60
BACKBONE_ALPHA = 0.35
BACKBONE_CUT_POINT = 'block_6_expand_relu'


def rgb565_roundtrip(rgb):
    """Match firmware's bit-replicated RGB565 expansion before normalization."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    r, g, b = rgb[..., 0] >> 3, rgb[..., 1] >> 2, rgb[..., 2] >> 3
    return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4),
                     (b << 3) | (b >> 2)), axis=-1)


def make_dataset(count, seed):
    """Independent scenes: 0-5 shaded cubes, dark belt, random position/light/noise.

    boxes are (class_id, x, y, width, height) in 96x96 input coordinates.
    A 24-pixel center separation prevents overlapping objects/adjacent targets.
    """
    rng = np.random.default_rng(seed)
    images, annotations = [], []
    yy, xx = np.mgrid[:INPUT_H, :INPUT_W]
    for _ in range(count):
        level = rng.uniform(30, 80)
        texture = rng.normal(0, 3, (INPUT_H, INPUT_W))
        texture += 4 * np.sin((yy + rng.uniform(0, 10)) / 4)
        background = np.clip(level + texture, 0, 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(background[..., None], 3, axis=-1))
        draw = ImageDraw.Draw(image)
        boxes, centers = [], []
        for _ in range(int(rng.integers(0, 6))):
            for attempt in range(200):
                side = int(rng.integers(10, 17))
                x = int(rng.integers(2, INPUT_W - side - 2))
                y = int(rng.integers(2, INPUT_H - side - 2))
                cx, cy = x + side / 2, y + side / 2
                if all(np.hypot(cx - px, cy - py) >= 24 for px, py in centers):
                    break
            else:
                continue
            cid = int(rng.integers(1, len(CLASS_NAMES)))
            color = np.clip(COLORS[cid - 1] * rng.uniform(.70, 1.12)
                            + rng.normal(0, 5, 3), 0, 255)
            draw.rectangle((x + 2, y + 3, x + side + 1, y + side + 2), fill=(20, 20, 20))
            draw.rectangle((x, y, x + side - 1, y + side - 1), fill=tuple(color.astype(int)))
            # Top and right faces of an overhead cube; annotation covers the cube.
            draw.polygon([(x, y), (x + side - 1, y), (x + side - 4, y + 3),
                          (x + 3, y + 3)], fill=tuple(np.clip(color * 1.17, 0, 255).astype(int)))
            draw.polygon([(x + side - 1, y), (x + side - 1, y + side - 1),
                          (x + side - 4, y + side - 4), (x + side - 4, y + 3)],
                         fill=tuple((color * .70).astype(int)))
            boxes.append((cid, x, y, side, side))
            centers.append((cx, cy))
        pixels = np.asarray(image).astype(np.float32) + rng.normal(0, 1.5, (96, 96, 3))
        images.append(rgb565_roundtrip(np.clip(pixels, 0, 255).astype(np.uint8)))
        annotations.append(boxes)
    return np.asarray(images), annotations


def centroid_targets(annotations):
    """One foreground cell per box centroid; refuse unrepresentable collisions."""
    targets = np.zeros((len(annotations), GRID_H, GRID_W), np.int32)
    for i, boxes in enumerate(annotations):
        for cid, x, y, w, h in boxes:
            gx, gy = int((x + w / 2) // STRIDE), int((y + h / 2) // STRIDE)
            if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
                raise ValueError('Centroid outside input image')
            if targets[i, gy, gx]:
                raise ValueError('Two centroids occupy one grid cell; increase resolution')
            targets[i, gy, gx] = cid
    return targets


def build_model(tf):
    """EI's FOMO trunk/head architecture, with randomly initialized weights.

    Keep the stride-8 expansion before block 6's depthwise downsampling.
    Training/decoder code is local; this does not load EI's pretrained checkpoint.
    """
    layers = tf.keras.layers
    inputs = layers.Input((INPUT_H, INPUT_W, 3), name='rgb_0_to_1')
    mobilenet = tf.keras.applications.MobileNetV2(
        input_tensor=inputs, alpha=BACKBONE_ALPHA, include_top=False, weights=None)
    # Faster moving-statistic adaptation for this small, from-scratch dataset.
    for layer in mobilenet.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.momentum = 0.9
    x = mobilenet.get_layer(BACKBONE_CUT_POINT).output
    x = layers.Conv2D(32, 1, activation='relu', name='head')(x)
    logits = layers.Conv2D(len(CLASS_NAMES), 1, name='logits')(x)
    outputs = layers.Softmax(axis=-1, name='cell_probabilities')(logits)
    return tf.keras.Model(inputs, outputs, name='fomo_mobilenetv2_cubes')


def decode_heatmap(probabilities, threshold=None):
    """Argmax, threshold, then 4-connected components of the SAME class.

    Coordinates are confidence-weighted cell centers; score is peak probability.
    This policy is also implemented in src/FomoDecoder.h (not EI SDK merging).
    """
    if threshold is None:
        threshold = THRESHOLD
    p = np.asarray(probabilities)
    assert p.shape == (GRID_H, GRID_W, len(CLASS_NAMES))
    winners = p.argmax(axis=-1)
    confidence = p.max(axis=-1)
    active = np.where((winners != 0) & (confidence >= threshold), winners, 0)
    seen = np.zeros((GRID_H, GRID_W), bool)
    detections = []
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            cid = int(active[gy, gx])
            if not cid or seen[gy, gx]:
                continue
            queue = [(gy, gx)]
            seen[gy, gx] = True
            sx = sy = total = peak = 0.
            for y, x in queue:
                score = float(confidence[y, x])
                sx += (x + .5) * STRIDE * score
                sy += (y + .5) * STRIDE * score
                total += score
                peak = max(peak, score)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= ny < GRID_H and 0 <= nx < GRID_W and not seen[ny, nx]
                            and active[ny, nx] == cid):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            detections.append(dict(class_id=cid, x=sx / total, y=sy / total, score=peak))
    return detections


def detection_metrics(probabilities, annotations, threshold=None, tolerance=8.):
    """One-to-one same-class matching within 8 input pixels, highest score first."""
    tp = fp = fn = exact = 0
    errors = []
    per_class = {name: dict(tp=0, fp=0, fn=0) for name in CLASS_NAMES[1:]}
    for heatmap, boxes in zip(probabilities, annotations):
        detections = decode_heatmap(heatmap, threshold)
        unmatched = set(range(len(boxes)))
        for det in sorted(detections, key=lambda d: -d['score']):
            candidates = [(np.hypot(det['x'] - (x + w / 2), det['y'] - (y + h / 2)), j)
                          for j, (cid, x, y, w, h) in enumerate(boxes)
                          if j in unmatched and cid == det['class_id']]
            distance, match = min(candidates, default=(float('inf'), -1))
            name = CLASS_NAMES[det['class_id']]
            if distance <= tolerance:
                tp += 1
                per_class[name]['tp'] += 1
                unmatched.remove(match)
            else:
                fp += 1
                per_class[name]['fp'] += 1
        fn += len(unmatched)
        for j in unmatched:
            per_class[CLASS_NAMES[boxes[j][0]]]['fn'] += 1
        actual = np.bincount([b[0] for b in boxes], minlength=len(CLASS_NAMES))[1:]
        predicted = np.bincount([d['class_id'] for d in detections], minlength=len(CLASS_NAMES))[1:]
        exact += int(np.array_equal(actual, predicted))
        errors.append(int(np.abs(actual - predicted).sum()))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2 * precision * recall / max(precision + recall, 1e-9),
                exact_class_counts=exact / len(annotations),
                mean_absolute_class_count_error=float(np.mean(errors)), per_class=per_class)


def quantize_input(images, detail):
    scale, zero = detail['quantization']
    return np.clip(np.floor(np.asarray(images, np.float32) / 255. / scale + zero + .5),
                   -128, 127).astype(np.int8)


def tflite_predict(tf, model_bytes, images):
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    assert inp['dtype'] == np.int8 and out['dtype'] == np.int8
    assert tuple(inp['shape']) == (1, INPUT_H, INPUT_W, 3)
    assert tuple(out['shape']) == (1, GRID_H, GRID_W, len(CLASS_NAMES))
    results = []
    for image in images:
        interpreter.set_tensor(inp['index'], quantize_input(image[None], inp))
        interpreter.invoke()
        results.append((interpreter.get_tensor(out['index'])[0].astype(np.float32)
                        - out['quantization'][1]) * out['quantization'][0])
    return np.asarray(results), inp, out


def export_firmware(model_bytes, input_detail, output_detail, example, example_heatmap,
                    root='.'):
    """Generate C++ bytes, contract and a known int8 test frame from one export."""
    root = Path(root)
    src = root / 'src'
    src.mkdir(exist_ok=True)
    def array_text(data):
        flat = np.asarray(data).ravel()
        return '\n'.join('    ' + ', '.join(map(str, flat[i:i+16])) + ','
                         for i in range(0, len(flat), 16))
    (src / 'model.cc').write_text(
        '// Generated by train-fomo.ipynb. Do not edit.\n#include "model.h"\n'
        'alignas(16) const unsigned char gmodel[] = {\n' +
        '\n'.join('    ' + ', '.join(f'0x{v:02x}' for v in model_bytes[i:i+12]) + ','
                  for i in range(0, len(model_bytes), 12)) +
        '\n};\nconst unsigned int gmodel_len = sizeof(gmodel);\n')
    labels = ', '.join(json.dumps(n) for n in CLASS_NAMES)
    (src / 'model_config.h').write_text(f'''// Generated by train-fomo.ipynb. Do not edit.
#pragma once
namespace fomo {{
constexpr int kInputWidth = {INPUT_W};
constexpr int kInputHeight = {INPUT_H};
constexpr int kChannels = 3;
constexpr int kGridWidth = {GRID_W};
constexpr int kGridHeight = {GRID_H};
constexpr int kClassCount = {len(CLASS_NAMES)};
constexpr float kThreshold = {THRESHOLD}f;
constexpr float kInputScale = {input_detail['quantization'][0]:.12g}f;
constexpr int kInputZeroPoint = {input_detail['quantization'][1]};
constexpr float kOutputScale = {output_detail['quantization'][0]:.12g}f;
constexpr int kOutputZeroPoint = {output_detail['quantization'][1]};
constexpr const char* kLabels[] = {{{labels}}};
}}  // namespace fomo
''')
    (src / 'img.h').write_text('// Generated int8 input for FOMO_STATIC_IMAGE=1.\n'
        '#pragma once\n#include <stdint.h>\nconst int8_t img_data[] = {\n' +
        array_text(quantize_input(example, input_detail)) + '\n};\n')
    expected = decode_heatmap(example_heatmap)
    (root / 'model' / 'static-expected.json').write_text(json.dumps(expected, indent=2) + '\n')
    np.savez(root / 'model' / 'static-parity.npz', input=quantize_input(example, input_detail),
             heatmap=example_heatmap)
    return expected
