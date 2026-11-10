"""Execute every notebook cell on synthetic WAVs; NOT a model accuracy benchmark.

Usage: python tools/smoke_notebook.py /tmp/kws-smoke
All data, artifacts and model headers stay in the requested scratch directory.
"""
from pathlib import Path
import json
import os
import sys
import wave
import numpy as np
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]

def main():
    target = Path(sys.argv[1]).resolve()
    target.mkdir(parents=True, exist_ok=True)
    data = target/'data'
    rng = np.random.default_rng(7)
    from kws_training import speaker_split
    speakers = {split: [] for split in ("train", "validation", "test")}
    i = 0
    while any(len(ids) < 16 for ids in speakers.values()):
        speaker = f"smoke{i}"
        split = speaker_split(speaker)
        if len(speakers[split]) < 16:
            speakers[split].append(speaker)
        i += 1
    def wav(path, signal):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(np.clip(signal, -32768, 32767).astype('<i2').tobytes())
    t = np.arange(16000)/16000
    for word, freq in [('yes', 500), ('no', 1000), ('up', 2000), ('stop', 3000)]:
        for split, count in [('train', 16), ('validation', 4), ('test', 4)]:
            for i in range(count):
                path = data/word/f'{speakers[split][i]}_nohash_0.wav'
                wav(path, 4000*np.sin(2*np.pi*freq*t) + rng.normal(0, 100, 16000))
    spec = target/'kernels'/'kws-smoke'
    spec.mkdir(parents=True, exist_ok=True)
    (spec/'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='KWS smoke', language='python')))
    km = KernelManager(kernel_name='kws-smoke',
                       kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec.parent)]))
    env = dict(os.environ, KWS_DATA_DIR=str(data), KWS_CAP='16', KWS_EPOCHS='1', KWS_AUTO_DOWNLOAD='0',
               KWS_ARTIFACT_DIR=str(target/'artifacts'), KWS_EXPORT_DIR=str(target/'export'),
               MPLBACKEND='Agg', IPYTHONDIR=str(target/'ipython'))
    nb = nbformat.read(ROOT/'train-kws.ipynb', as_version=4)
    client = NotebookClient(nb, km=km, timeout=300, resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute(env=env)
    finally:
        nbformat.write(nb, target/'executed.ipynb')
    for cell in nb.cells:
        if cell.cell_type == 'code':
            for output in cell.get('outputs', []):
                if output.output_type == 'stream' and output.name == 'stdout':
                    print(output.text[-1500:])
    print('PASS: all notebook cells executed. Synthetic accuracy is not meaningful.')
    print('Scratch exports:', target/'export')

if __name__ == '__main__':
    main()
