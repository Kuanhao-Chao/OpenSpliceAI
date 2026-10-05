#!/usr/bin/env python3
"""Run six real CPU workflows on synthetic data and validate their output schemas.

Run from a clone with OpenSpliceAI installed in the selected Python environment.
The small trained model verifies mechanics and has no biological accuracy claim.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pysam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable, help='Interpreter with OpenSpliceAI installed')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output directory must be empty to preserve earlier artifacts')
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg',
                       PYTHONDONTWRITEBYTECODE='1')
    environment['PYTHONPATH'] = str(root)
    commands = []

    def run(name, command):
        argv = [args.python, '-c', 'from openspliceai.openspliceai import main; main()', *map(str, command)]
        commands.append({'name': name, 'argv': argv})
        with (output/f'{name}.log').open('w') as log:
            subprocess.run(argv, cwd=output, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=300)

    rng = np.random.default_rng(2026)
    sequences = {f'chr{i}': ''.join(rng.choice(list('ACGT'), 12000)) for i in range(1, 5)}
    fasta, gff = output/'genome.fa', output/'genes.gff'
    fasta.write_text(''.join(f'>{chrom}\n{sequence}\n' for chrom, sequence in sequences.items()))
    lines = ['##gff-version 3']
    for chrom in sequences:
        for index in range(3):
            start, end = 1001+index*3000, 2500+index*3000
            gene, strand = f'{chrom}_g{index}', '+' if index % 2 == 0 else '-'
            lines += [f'{chrom}\ttutorial\tgene\t{start}\t{end}\t.\t{strand}\t.\tID={gene};gene_biotype=protein_coding',
                      f'{chrom}\ttutorial\tmRNA\t{start}\t{end}\t.\t{strand}\t.\tID={gene}.t;Parent={gene}']
            for exon, left, right in ((1, start, start+499), (2, end-499, end)):
                lines.append(f'{chrom}\ttutorial\texon\t{left}\t{right}\t.\t{strand}\t.\tID={gene}.e{exon};Parent={gene}.t')
    gff.write_text('\n'.join(lines)+'\n')
    data = output/'data'
    run('create-data', ['create-data', '--annotation-gff', gff, '--genome-fasta', fasta,
                        '--output-dir', data, '--split-ratio', '.75', '--val_split_ratio', '.25', '--verify-h5'])
    for split in ('train', 'validation', 'test'):
        with h5py.File(data/f'dataset_{split}.h5') as handle:
            assert handle['X0'].shape[0] > 0 and handle['Y0'].shape[-1] == 3
    common = ['--project-name', 'tutorial', '--flanking-size', '80', '--epochs', '1',
              '--train-dataset', data/'dataset_train.h5', '--validation-dataset', data/'dataset_validation.h5',
              '--test-dataset', data/'dataset_test.h5']
    run('train', ['train', '--output-dir', output/'train', *common])
    checkpoint = output/'train/SpliceAI_tutorial_80_0_rs42/0/models/model_best.pt'
    assert checkpoint.is_file()
    run('transfer', ['transfer', '--output-dir', output/'transfer', '--pretrained-model', checkpoint,
                      '--unfreeze', '0', *common])
    run('calibrate', ['calibrate', '--output-dir', output/'calibrate', '--project-name', 'tutorial',
                       '--flanking-size', '80', '--epochs', '2', '--pretrained-model', checkpoint,
                       '--validation-dataset', data/'dataset_validation.h5', '--test-dataset', data/'dataset_test.h5'])
    calibrated = output/'calibrate/calibrated_model.pt'
    assert calibrated.is_file()
    summary = json.loads((output/'calibrate/calibration/results/validation/summary.json').read_text())
    assert summary['metrics_scope'] == 'complete_selected_split' and summary['observations'] > 0
    small_fasta = output/'prediction.fa'
    small_fasta.write_text('>NC_tutorial:101-1100(-)\n'+sequences['chr1'][100:1100]+'\n')
    run('predict', ['predict', '--model', calibrated, '--flanking-size', '80',
                    '--input-sequence', small_fasta, '--output-dir', output/'predict', '--threshold', '0'])
    bed_rows = 0
    for bed in (output/'predict').rglob('*.bed'):
        for line in bed.read_text().splitlines():
            fields = line.split('\t')
            assert len(fields) == 6 and fields[0] == 'NC_tutorial' and fields[5] == '-'
            assert 100 <= int(fields[1]) < int(fields[2]) <= 1100
            bed_rows += 1
    assert bed_rows > 0
    annotation = output/'annotation.tsv'
    annotation.write_text('#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\n'
                          'TUTORIAL\tchr1\t+\t999\t9000\t999,5999,\t5000,9000,\n')
    vcf = output/'variants.vcf'
    ref = sequences['chr1'][5999]
    alt = next(base for base in 'ACGT' if base != ref)
    vcf.write_text('##fileformat=VCFv4.2\n##contig=<ID=chr1,length=12000>\n'
                  '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
                  f'chr1\t6000\t.\t{ref}\t{alt}\t.\t.\t.\n')
    result = output/'annotated.vcf.gz'
    run('variant', ['variant', '--model', calibrated, '--flanking-size', '80', '--ref-genome', fasta,
                    '--annotation', annotation, '--input-vcf', vcf, '--output-vcf', result, '--precision', '6'])
    with pysam.VariantFile(result) as handle:
        records = list(handle)
        assert len(records) == 1 and len(records[0].info['OpenSpliceAI'][0].split('|')) == 10
    (output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps({'commands_passed': 6, 'bed_rows': bed_rows,
        'calibrated_inference': ['predict', 'variant'], 'synthetic_model': True}, indent=2)+'\n')
    print(f'Six workflows passed; artifacts: {output}')


if __name__ == '__main__':
    main()
