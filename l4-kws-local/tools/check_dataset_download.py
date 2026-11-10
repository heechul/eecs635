"""Regression check: the notebook reloads stale helpers and requests only mini ZIP.

Runs offline with a tiny ZIP fixture and a mocked TensorFlow downloader.
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import importlib
import json
import sys
import tempfile
import zipfile
import kws_training as kws

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='kws-download-check-') as scratch:
        scratch = Path(scratch).resolve()
        archive = scratch/'mini.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
            for word in kws.MINI_WORDS:
                for i in range(1000):
                    zipped.writestr(f'mini_speech_commands/{word}/{i:08x}_nohash_0.wav', b'')
        downloader = Mock(return_value=str(archive))
        fake_tf = SimpleNamespace(keras=SimpleNamespace(utils=SimpleNamespace(get_file=downloader)))
        stale = Mock(side_effect=AssertionError('Stale downloader was called'))
        original_module = kws
        original_module.download_dataset = stale
        notebook = json.loads((ROOT/'train-kws.ipynb').read_text())
        cell = next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and 'records, noise_sources =' in ''.join(c['source']))
        namespace = dict(Path=Path, DATA_DIR=scratch/'data', AUTO_DOWNLOAD=True,
                         CAP_PER_CLASS=1000, SEED=42, EXTRA_DIR=scratch/'extra',
                         download_dataset=stale, kws=original_module)
        try:
            with patch.dict(sys.modules, {'tensorflow': fake_tf}), redirect_stdout(StringIO()):
                exec(compile(cell, 'notebook-download-cell', 'exec'), namespace)
            stale.assert_not_called()
            downloader.assert_called_once_with(
                kws.MINI_ARCHIVE, origin=kws.MINI_DATASET_URL, file_hash=kws.MINI_SHA256,
                hash_algorithm='sha256', cache_dir=str(scratch), cache_subdir='downloads')
            assert namespace['kws'].__file__ == str(ROOT/'tools/kws_training.py')
            assert sum(kws.validate_mini_dataset(scratch/'data').values()) == 8000
            (scratch/'data'/'backward').mkdir()
            (scratch/'data'/'backward'/'someone_nohash_0.wav').touch()
            try:
                kws.validate_mini_dataset(scratch/'data')
            except ValueError:
                pass
            else:
                raise AssertionError('Full-dataset-only word was accepted')
        finally:
            importlib.reload(kws)
    print('PASS: stale helper bypassed; mini ZIP URL + checksum requested; 8,000 clips validated; full dataset rejected')


if __name__ == '__main__':
    main()
