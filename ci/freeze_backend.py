"""Freeze backend test inputs and verify every copied file before a Slurm run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify(root):
    manifest = json.loads((root/'source-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = root/relative
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f'Frozen backend input changed: {relative}')
    print(f"Verified {len(manifest['files'])} frozen inputs")


def freeze(root, destination, weights_root):
    if destination.exists():
        raise ValueError('Snapshot destination must not already exist')
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
        '--', 'openspliceai', 'tests', 'validation', 'ci', 'setup.py', 'pyproject.toml',
        'pytest.ini', '.coveragerc', 'LICENSE'], cwd=root, text=True).splitlines()
    paths = [name for name in paths if not {'scripts','test','__pycache__'}.intersection(Path(name).parts)]
    inputs = {name:root/name for name in paths}
    for pattern in ('models/openspliceai-honeybee/80nt/model_80nt_rs*.pt',
                    'models/spliceai/SpliceAI_models_release/spliceai[1-5].h5'):
        weights = sorted(weights_root.glob(pattern))
        if len(weights) != 5:
            raise ValueError(f'Required five backend reference weights unavailable: {pattern}')
        inputs.update({str(path.relative_to(weights_root)):path for path in weights})
    destination.mkdir(parents=True)
    hashes = {}
    for name, source in sorted(inputs.items()):
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        hashes[name] = digest(target)
        if hashes[name] != digest(source):
            raise ValueError(f'Input changed during snapshot: {name}')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    metadata = {'git_revision':revision,'files':hashes,
                'scope':'Backend source, tests, runners and ten reference weights; outputs are outside the manifest.'}
    (destination/'source-manifest.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    verify(destination)
    print('Manifest SHA-256: '+digest(destination/'source-manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--weights-root',type=Path)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify(args.root)
    elif args.destination:
        freeze(args.root,args.destination,args.weights_root or args.root)
    else:
        parser.error('--destination or --verify is required')


if __name__ == '__main__':
    main()
