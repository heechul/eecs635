"""Contract and cross-language checks; run after generating notebook artifacts."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fomo_lab as lab


class FomoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.driver = str(Path(cls.tmp.name) / 'fomo_host')
        subprocess.run(['c++', '-std=c++11', '-Wall', '-Wextra', '-Werror',
                        '-I', str(ROOT / 'src'), str(ROOT / 'test/fomo_host.cpp'),
                        '-o', cls.driver], check=True)
        cls.scale, cls.zero = 1 / 256, -128  # int8 Softmax output contract.

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def host(self, mode, data):
        path = Path(self.tmp.name) / 'input.bin'
        path.write_bytes(data)
        return subprocess.check_output([self.driver, mode, str(path)])

    def assert_decoder_parity(self, quantized):
        probabilities = (quantized.astype(np.float32) - self.zero) * self.scale
        expected = lab.decode_heatmap(probabilities)
        actual = json.loads(self.host('decode', quantized.tobytes()))
        self.assertEqual([d['class_id'] for d in expected], [d['class_id'] for d in actual])
        for left, right in zip(expected, actual):
            np.testing.assert_allclose([left[k] for k in ('x', 'y', 'score')],
                                       [right[k] for k in ('x', 'y', 'score')], atol=2e-5)
        return actual

    def test_centroid_targets_and_collisions(self):
        boxes = [[(1, 0, 0, 8, 8), (4, 88, 88, 8, 8)], []]
        targets = lab.centroid_targets(boxes)
        self.assertEqual(targets[0, 0, 0], 1)
        self.assertEqual(targets[0, 11, 11], 4)
        self.assertEqual(np.count_nonzero(targets), 2)
        with self.assertRaises(ValueError):
            lab.centroid_targets([[(1, 1, 1, 4, 4), (2, 2, 2, 4, 4)]])

    def test_generator_reproducibility_and_empty_frames(self):
        a, boxes_a = lab.make_dataset(40, 12)
        b, boxes_b = lab.make_dataset(40, 12)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(boxes_a, boxes_b)
        self.assertTrue(any(not box for box in boxes_a))
        targets = lab.centroid_targets(boxes_a)
        self.assertEqual(np.count_nonzero(targets), sum(map(len, boxes_a)))
        c, _ = lab.make_dataset(40, 13)
        self.assertFalse(np.array_equal(a, c))

    def test_cpp_decoder_boundaries_classes_and_weighted_merge(self):
        q = np.full((12, 12, 5), -128, np.int8)
        q[..., 0] = 127
        self.assertEqual(self.assert_decoder_parity(q), [])
        # Adjacent equal classes merge; adjacent different classes do not.
        # Diagonal equal classes do not merge. Row ends never wrap.
        for y, x, cid, score in [(0, 0, 1, 127), (0, 1, 1, 50), (0, 2, 2, 127),
                                  (1, 3, 2, 127), (3, 11, 3, 127), (4, 0, 3, 127),
                                  (11, 11, 4, 25)]:
            q[y, x, 0] = -128
            q[y, x, cid] = score
        detections = self.assert_decoder_parity(q)
        self.assertEqual(len(detections), 5)  # 25 gives 0.5977, below threshold.
        self.assertGreater(detections[0]['x'], 4.)
        self.assertLess(detections[0]['x'], 8.)
        rng = np.random.default_rng(6)
        self.assert_decoder_parity(rng.integers(-128, 128, (12, 12, 5), dtype=np.int8))
        # Maximum number of disconnected detections fills capacity without overflow.
        for y in range(12):
            for x in range(12):
                q[y, x] = -128
                q[y, x, 1 + (x + y) % 2] = 127
        self.assertEqual(len(self.assert_decoder_parity(q)), 144)

    def test_cpp_rgb565_preprocessing(self):
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        rgb[0, :5] = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
        words = ((rgb[..., 0].astype(np.uint16) >> 3) << 11 |
                 (rgb[..., 1].astype(np.uint16) >> 2) << 5 |
                 (rgb[..., 2].astype(np.uint16) >> 3))
        frame = words.astype('>u2').tobytes()
        expected = lab.quantize_input(lab.rgb565_roundtrip(rgb),
                                      {'quantization': (1 / 255, -128)})
        actual = np.frombuffer(self.host('prepare', frame), np.int8).reshape(96, 96, 3)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(subprocess.CalledProcessError):
            self.host('prepare', frame[:30])

    def test_detection_matching_does_not_reuse_ground_truth(self):
        p = np.zeros((1, 12, 12, 5), np.float32)
        p[..., 0] = 1
        for x in (0, 2):
            p[0, 0, x, 0] = 0
            p[0, 0, x, 1] = 1
        metrics = lab.detection_metrics(p, [[(1, 8, 0, 8, 8)]])
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (1, 1, 0))
        self.assertEqual(metrics['precision'], .5)

    def test_exported_tflite_and_static_fixture(self):
        import tensorflow as tf
        model = lab.build_model(tf)
        self.assertEqual(model.get_layer(lab.BACKBONE_CUT_POINT).output_shape,
                         (None, 12, 12, 96))
        self.assertEqual(model.output_shape, (None, 12, 12, 5))
        self.assertTrue(any(isinstance(layer, tf.keras.layers.DepthwiseConv2D)
                            for layer in model.layers))
        self.assertNotIn('block_6_depthwise', [layer.name for layer in model.layers])
        model_bytes = (ROOT / 'model/fomo-int8.tflite').read_bytes()
        source = (ROOT / 'src/model.cc').read_text()
        self.assertEqual(bytes(int(v, 16) for v in re.findall(r'0x([0-9a-f]{2})', source)), model_bytes)
        from PIL import Image
        image = np.asarray(Image.open(ROOT / 'model/static-input.png'))
        probabilities, inp, out = lab.tflite_predict(tf, model_bytes, image[None])
        self.assertAlmostEqual(inp['quantization'][0], 1 / 255, places=8)
        self.assertEqual(inp['quantization'][1], -128)
        self.assertEqual(out['quantization'], (self.scale, self.zero))
        check = tf.lite.Interpreter(model_content=model_bytes)
        check.allocate_tensors()
        self.assertTrue(all(t['dtype'] != np.float32 for t in check.get_tensor_details()))
        ops = {op['op_name'] for op in check._get_ops_details()} - {'DELEGATE'}
        self.assertEqual(ops, {'CONV_2D', 'DEPTHWISE_CONV_2D', 'ADD', 'SOFTMAX'})
        with np.load(ROOT / 'model/static-parity.npz') as archive:
            fixture = {key: archive[key] for key in archive.files}
        np.testing.assert_array_equal(fixture['input'], lab.quantize_input(image, inp))
        header = (ROOT / 'src/img.h').read_text().split('{', 1)[1].split('}', 1)[0]
        np.testing.assert_array_equal(np.fromstring(header.strip().rstrip(','), sep=',', dtype=np.int16),
                                      fixture['input'].ravel())
        np.testing.assert_allclose(probabilities[0], fixture['heatmap'], atol=0)
        q = np.rint(probabilities[0] / self.scale + self.zero).astype(np.int8)
        actual = self.assert_decoder_parity(q)
        expected = json.loads((ROOT / 'model/static-expected.json').read_text())
        self.assertEqual([d['class_id'] for d in actual], [d['class_id'] for d in expected])


if __name__ == '__main__':
    unittest.main()
