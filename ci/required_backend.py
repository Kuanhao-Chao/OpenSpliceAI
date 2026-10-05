"""Run a required optional backend, failing on unavailable dependencies or skips."""
import argparse
import json
import os
from pathlib import Path
import sys
import platform
from importlib.metadata import version


class Results:
    def __init__(self):
        self.counts = {'passed':0,'failed':0,'skipped':0}

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.skipped or (report.when == 'setup' and report.failed):
            self.counts[report.outcome] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend',choices=['gpu','keras'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.backend=='keras':
        assert os.environ.get('TF_USE_LEGACY_KERAS')=='1', 'Set TF_USE_LEGACY_KERAS=1 before startup'
        import tensorflow
        import tf_keras
        import spliceai  # noqa: F401
        assert (Path('models/spliceai/SpliceAI_models_release')/'spliceai1.h5').is_file(), 'Required original weights unavailable'
    import torch
    import numpy
    if args.backend=='gpu':
        assert torch.cuda.is_available(), 'Required CUDA device unavailable'
        assert torch.cuda.device_count()==1, 'Audit reserves exactly one GPU'
    import pytest
    results=Results()
    status=pytest.main(['-m',args.backend,'-q','-p','no:cacheprovider','--tb=short',
                       '--junitxml='+str(args.output_dir/'tests.xml')],plugins=[results])
    metadata={'backend':args.backend,'status':status,'counts':results.counts,
              'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
              'numpy':numpy.__version__,'job_id':os.environ.get('SLURM_JOB_ID')}
    if args.backend=='gpu':
        metadata.update(cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    else:
        metadata.update(tensorflow=tensorflow.__version__,tf_keras=tf_keras.__version__,spliceai=version('spliceai'))
    (args.output_dir/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return status or int(results.counts['skipped']>0 or results.counts['passed']==0)


if __name__=='__main__':
    raise SystemExit(main())
