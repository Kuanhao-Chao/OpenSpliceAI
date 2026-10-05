"""Check installed entrypoints, annotations and real inference away from source."""
import argparse
from importlib.resources import files
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import openspliceai
from openspliceai.header import __version__
from openspliceai.checkpoints import calibrated_checkpoint, atomic_torch_save
from openspliceai.predict.predict import load_pytorch_models
from openspliceai.train_base.openspliceai import SpliceAI
from openspliceai.variant.utils import load_pytorch_models as load_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-prefix', type=Path, required=True)
    args = parser.parse_args()
    assert Path(openspliceai.__file__).resolve().is_relative_to(args.expected_prefix.resolve())
    assert __version__ == '0.1.0.dev0'
    for name in ('grch37','grch38'):
        resource = files('openspliceai.variant')/'annotations'/f'{name}.txt'
        assert resource.is_file() and len(resource.read_text().splitlines()) > 1000
    subprocess.run([str(args.expected_prefix/'bin/openspliceai'), 'variant','--help'], check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    torch.manual_seed(1)
    model = SpliceAI(32,np.array([11]*4),np.array([1]*4)).eval()
    inputs = torch.zeros(2,4,101)
    inputs[:,0] = 1
    with torch.no_grad():
        expected_raw = model(inputs)
    with tempfile.TemporaryDirectory(prefix='osai-installed-smoke-') as directory:
        raw, calibrated = Path(directory)/'raw.pt',Path(directory)/'calibrated.pt'
        atomic_torch_save(model.state_dict(),raw)
        temperature = torch.tensor([.5,1.,2.])
        atomic_torch_save(calibrated_checkpoint(model,temperature,80),calibrated)
        model.apply_softmax = False
        with torch.no_grad():
            expected_calibrated = torch.softmax(model(inputs)/temperature.view(1,3,1),dim=1)
        for path, expected in ((raw,expected_raw),(calibrated,expected_calibrated)):
            models,_ = load_pytorch_models(str(path),torch.device('cpu'),21,80)
            scoring = load_variant(str(path),80)
            with torch.no_grad():
                torch.testing.assert_close(models[0](inputs),expected,rtol=0,atol=0)
                torch.testing.assert_close(scoring[0](inputs),expected,rtol=0,atol=0)
    print(f'Installed {__version__}: CLI, annotations, raw/calibrated prediction and scoring passed ({sys.prefix})')


if __name__ == '__main__':
    main()
