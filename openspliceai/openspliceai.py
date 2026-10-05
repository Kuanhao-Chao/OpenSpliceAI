"""
Filename: openspliceai.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Main script to run OpenSpliceAI toolkit.
"""

import argparse
import sys
import math
from openspliceai import header

# NOTE: the subcommand packages (create_data, train, calibrate, transfer,
# predict, variant) are imported lazily inside main()'s dispatch, NOT here. Each
# one pulls a heavy dependency stack (torch, pandas, scikit-learn/scipy,
# biopython, pysam, matplotlib, ...). Importing them at module load makes even
# `openspliceai` with no arguments or `openspliceai --help` drag in the entire
# stack, so a numpy/alias incompatibility in any single transitive dependency
# (e.g. GitHub issue #19: a dependency reading numpy's removed `np.long`) breaks
# the whole CLI. Deferring these imports keeps argument parsing dependency-free
# and scopes each subcommand's imports to when that subcommand actually runs.

__VERSION__ = header.__version__

def positive_int(value):
    """Argparse type for positive integer sizes and iteration counts."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def nonnegative_float(value):
    """Argparse type rejecting negative and nonfinite coefficients."""
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return number


def probability(value):
    """Argparse type for finite probabilities in [0, 1]."""
    number = nonnegative_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError('must be between zero and one')
    return number


def add_training_options(parser):
    """Shared explicit split, focal-loss and partial-batch options."""
    parser.add_argument('--validation-dataset', help='Explicit validation HDF5; otherwise derive from the training filename')
    parser.add_argument('--focal-alpha', type=nonnegative_float, default=.25, help='Scalar focal-loss weight (default: 0.25)')
    parser.add_argument('--focal-gamma', type=nonnegative_float, default=2., help='Focal focusing exponent (default: 2)')
    parser.add_argument('--drop-last', action='store_true', help='Drop partial training batches; evaluation always retains them')


def parse_args_create_data(subparsers):
    """Register the ``create-data`` subcommand and its arguments (GFF + FASTA -> train/test HDF5 datasets)."""
    parser_create_data = subparsers.add_parser('create-data', help='Create dataset for your genome for SpliceAI model training')
    parser_create_data.add_argument('--annotation-gff', type=str, required=True, help='Path to the GFF file')
    parser_create_data.add_argument('--genome-fasta', type=str, required=True, help='Path to the FASTA file')
    parser_create_data.add_argument('--output-dir', type=str, required=True, help='Output directory to save the data')
    parser_create_data.add_argument('--parse-type', type=str, default='canonical', choices=['canonical', 'all_isoforms'], help='Type of transcript processing')
    parser_create_data.add_argument('--biotype', type=str, default='protein-coding', choices=['protein-coding', 'non-coding', 'all'], help='Biotype of transcript processing')
    parser_create_data.add_argument('--chr-split', type=str, choices=['train-test','test'], default='train-test', help='Whether to obtain testing or both training and testing groups')
    parser_create_data.add_argument('--split-method', type=str, choices=['random', 'human'], default='random', help='Chromosome split method for training and testing dataset')
    parser_create_data.add_argument('--val_split_ratio', type=probability, default=0.1, help='Fraction of training genes assigned to validation')
    parser_create_data.add_argument('--split-ratio', type=probability, default=0.8, help='Approximate fraction of genome assigned to training chromosomes')
    parser_create_data.add_argument('--random-seed', type=int, default=42, help='Seed for chromosome and gene-group splitting')
    parser_create_data.add_argument('--canonical-only', action='store_true', default=False, help='Flag to obtain only canonical splice site pairs')
    parser_create_data.add_argument('--flanking-size', type=int, choices=[80,400,2000,10000], default=80, help='Target model context; stored dataset context is always 10000 bases')
    parser_create_data.add_argument('--verify-h5', action='store_true', default=False, help='Verify the generated HDF5 file(s)')
    parser_create_data.add_argument('--remove-paralogs', action='store_true', default=False, help='Remove paralogous sequences between training and testing dataset')
    parser_create_data.add_argument('--min-identity', type=probability, default=0.8, help='Minimum minimap2 alignment identity for paralog removal between training and testing dataset')
    parser_create_data.add_argument('--min-coverage', type=probability, default=0.5, help='Minimum minimap2 query coverage for paralog removal between training and testing dataset')
    parser_create_data.add_argument('--write-fasta', action='store_true', default=False, help='Flag to write out sequences into fasta files')


def parse_args_train(subparsers):
    """Register the ``train`` subcommand and its arguments (train a SpliceAI model from scratch on HDF5 datasets)."""
    parser_train = subparsers.add_parser('train', help='Train the SpliceAI model')
    add_training_options(parser_train)
    parser_train.add_argument('--epochs', '-n', type=positive_int, default=10, help='Maximum number of optimization epochs')
    parser_train.add_argument('--scheduler', '-s', type=str, default="MultiStepLR", choices=["MultiStepLR", "CosineAnnealingWarmRestarts"], help="Learning rate scheduler")
    parser_train.add_argument('--early-stopping', '-E', action='store_true', default=False, help='Enable early stopping')
    parser_train.add_argument("--patience", '-P', type=positive_int, default=2, help="Number of epochs to wait before early stopping")
    parser_train.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_train.add_argument('--project-name', '-p', type=str, required=True, help="Project name for the train experiment")
    parser_train.add_argument('--exp-num', '-e', type=str, default="0", help="Experiment number")
    parser_train.add_argument('--flanking-size', '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_train.add_argument('--random-seed', '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_train.add_argument('--train-dataset', '-train', type=str, required=True, help="Path to the training dataset")
    parser_train.add_argument('--test-dataset', '-test', type=str, required=True, help="Path to the testing dataset")
    parser_train.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for training")
    parser_train.add_argument('--model', '-m', default="SpliceAI", type=str)


def parse_args_test(subparsers):
    """Register the (currently disabled) ``test`` subcommand and its arguments (evaluate a pretrained model)."""
    parser_test = subparsers.add_parser('test', help='Test the SpliceAI model')
    parser_test.add_argument("--pretrained-model", '-m', type=str, required=True, help="Path to the pre-trained model")
    parser_test.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_test.add_argument('--project-name', '-p', type=str, required=True, help="Project name for the fine-tuning experiment")
    parser_test.add_argument('--exp-num', '-e', type=str, default=0, help="Experiment number")
    parser_test.add_argument('--flanking-size', '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_test.add_argument('--random-seed', '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_test.add_argument('--test-dataset', '-test', type=str, required=True, help="Path to the testing dataset")
    parser_test.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for training")
    parser_test.add_argument('--test-target', '-t', default="OpenSpliceAI", choices=["OpenSpliceAI", "SpliceAI-Keras"], type=str)
    parser_test.add_argument('--log-dir', '-L', default="TEST_LOG", type=str)


def parse_args_calibrate(subparsers):
    """Register the ``calibrate`` subcommand and its arguments (post-hoc temperature scaling of a trained model)."""
    parser_calibrate = subparsers.add_parser('calibrate', help='Calibrate the SpliceAI model')
    parser_calibrate.add_argument('--epochs', '-n', type=positive_int, default=10, help='Maximum number of optimization epochs')
    parser_calibrate.add_argument('--early-stopping', '-E', action='store_true', default=False, help='Enable early stopping')
    parser_calibrate.add_argument("--patience", '-P', type=positive_int, default=2, help="Number of epochs to wait before early stopping")
    parser_calibrate.add_argument("--output-dir", '-o', type=str, required=True, help="Output directory for model checkpoints and logs")
    parser_calibrate.add_argument("--project-name", '-p', type=str, required=True, help="Project name for the fine-tuning experiment")
    parser_calibrate.add_argument("--exp-num", '-e', type=int, default=0, help="Experiment number")
    parser_calibrate.add_argument("--flanking-size", '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_calibrate.add_argument("--random-seed", '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_calibrate.add_argument("--temperature-file", '-T', type=str, default=None, required=False, help="Path to the temperature file")
    parser_calibrate.add_argument("--pretrained-model", '-m', type=str, required=True, help="Path to the pre-trained model")
    splits = parser_calibrate.add_mutually_exclusive_group(required=True)
    splits.add_argument('--validation-dataset', help='Held-out HDF5 used to fit temperatures')
    splits.add_argument('--train-dataset', '-train', help='Legacy shorthand: derive the validation filename by replacing train with validation')
    parser_calibrate.add_argument("--test-dataset", '-test', type=str, required=True, help="Path to the testing dataset")
    parser_calibrate.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss"], help="Temperature fitting uses negative log likelihood")


def parse_args_transfer(subparsers):
    """Register the ``transfer`` subcommand and its arguments (fine-tune a pretrained model on new data, optionally freezing layers)."""
    parser_transfer = subparsers.add_parser('transfer', help='Fine-tune a pretrained SpliceAI model')
    add_training_options(parser_transfer)
    parser_transfer.add_argument('--allow-partial-checkpoint', action='store_true', help='Explicitly allow matching student weights only; report missing weights')
    parser_transfer.add_argument('--epochs', '-n', type=positive_int, default=10, help='Maximum number of optimization epochs')
    parser_transfer.add_argument('--scheduler', '-s', type=str, default="MultiStepLR", choices=["MultiStepLR", "CosineAnnealingWarmRestarts"], help="Learning rate scheduler")
    parser_transfer.add_argument('--early-stopping', '-E', action='store_true', default=False, help='Enable early stopping')
    parser_transfer.add_argument("--patience", '-P', type=positive_int, default=2, help="Number of epochs to wait before early stopping")
    parser_transfer.add_argument("--output-dir", '-o', type=str, required=True, help="Output directory for model checkpoints and logs")
    parser_transfer.add_argument("--project-name", '-p', type=str, required=True, help="Project name for the fine-tuning experiment")
    parser_transfer.add_argument("--exp-num", '-e', type=int, default=0, help="Experiment number")
    parser_transfer.add_argument("--flanking-size", '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_transfer.add_argument("--random-seed", '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_transfer.add_argument("--pretrained-model", '-m', type=str, required=True, help="Path to the pre-trained model")
    parser_transfer.add_argument("--train-dataset", '-train', type=str, required=True, help="Path to the training dataset")
    parser_transfer.add_argument("--test-dataset", '-test', type=str, required=True, help="Path to the testing dataset")
    parser_transfer.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for fine-tuning")
    parser_transfer.add_argument("--unfreeze-all", '-A', action='store_true', default=False, help='Unfreeze all layers for fine-tuning (default: freeze all but the last --unfreeze residual units)')
    parser_transfer.add_argument("--unfreeze", '-u', type=int, default=1, help="Number of residual units (from the output end) to unfreeze for fine-tuning. Tip: on narrow data, prefer progressive unfreezing -- start small (e.g. 2) and increase across successive transfer rounds (each resuming from the previous round's checkpoint via --pretrained-model) while watching --genomic-eval-dataset, instead of --unfreeze-all.")
    # --- Catastrophic-forgetting mitigation (all optional, default-off; see docs) ---
    parser_transfer.add_argument("--weight-decay", type=nonnegative_float, default=0.01,
                                 help="AdamW weight decay. Default 0.01 decays every trainable weight toward zero each step, which erodes pretrained splice features when finetuning on narrow data; set 0 (or use --l2sp) to reduce catastrophic forgetting.")
    parser_transfer.add_argument("--l2sp", type=nonnegative_float, default=0.0,
                                 help="L2-SP regularization strength: penalize drift of trainable weights away from the pretrained weights (toward the starting point instead of toward zero). 0 (default) disables. Only active alongside a distillation teacher (--distill-weight>0), whose weights serve as the reference.")
    parser_transfer.add_argument("--genomic-eval-dataset", type=str, default=None,
                                 help="Held-out genomic HDF5 (e.g. MANE test-chromosome shards). PURE MEASUREMENT, not training: if set, after each epoch the model is evaluated on this set in eval mode (no loss, no gradient, no effect on training or checkpoints) and donor/acceptor AUPRC + top-k are appended to LOG/GENOMIC/ -- a per-epoch 'forgetting curve' for canonical genomic splice sites. Distinct from --test-dataset (which is your fine-tuning distribution). Off by default.")
    parser_transfer.add_argument("--rehearsal-dataset", type=str, default=None,
                                 help="Genomic HDF5 (e.g. a MANE dataset_train_*.h5 created at the same flanking size) used for experience replay. CHANGES TRAINING: its shards (with their true labels) are interleaved with the training shards, so part of every epoch's gradient comes from genome-wide data and the model can't drift off it. Pair with --rehearsal-shards to set the mix ratio. Off by default.")
    parser_transfer.add_argument("--rehearsal-shards", type=int, default=-1,
                                 help="Number of genomic anchor shards from --rehearsal-dataset to interleave (-1 = all). Controls the rehearsal mix ratio.")
    parser_transfer.add_argument("--distill-weight", type=nonnegative_float, default=0.0,
                                 help="Weight (lambda) for the knowledge-distillation / Learning-without-Forgetting auxiliary loss. Each step adds lambda * cross-entropy(teacher_soft_targets, student) on genomic anchor windows so the student keeps the pretrained model's genome-wide predictions. 0 (default) disables. Typical 0.1-1.0.")
    parser_transfer.add_argument("--distill-teacher", type=str, default=None,
                                 help="Frozen teacher checkpoint for the distillation loss. Defaults to --pretrained-model when --distill-weight>0.")
    parser_transfer.add_argument("--distill-shards", type=str, default=None,
                                 help="Genomic anchor HDF5 the teacher scores each step to provide soft targets (no labels needed). Required when --distill-weight>0; may reuse --genomic-eval-dataset.")
    parser_transfer.add_argument("--distill-batch-size", type=int, default=-1,
                                 help="Batch size for the genomic anchor (distillation) loader (-1 = same as the training batch size). Lower it if running the teacher + student together exhausts GPU memory at large flanking sizes.")


def parse_args_predict(subparsers):
    """Register the ``predict`` subcommand and its arguments (score a FASTA and write donor/acceptor BED predictions)."""
    parser_predict = subparsers.add_parser('predict', help='Predict splice sites in a given sequence using the SpliceAI model')
    parser_predict.add_argument('--input-sequence', '-i', type=str, required=True, help="Path to FASTA file of the input sequence")
    parser_predict.add_argument('--model', '-m', type=str, required=True, help='Path to a PyTorch SpliceAI model file')
    parser_predict.add_argument('--flanking-size', '-f', type=int, required=True, choices=[80, 400, 2000, 10000], help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_predict.add_argument('--output-dir', '-o', type=str, default="./predict_out", help='Output directory to save the data')
    parser_predict.add_argument('--annotation-file', '-a', type=str, required=False, help="Path to GFF file of coordinates for genes")
    parser_predict.add_argument('--gene-flank', type=int, default=-1, help="With -a/--annotation: bp of REAL genomic flanking sequence to include on each side of every extracted gene so the model sees true context instead of 'N' padding at gene boundaries. Default (-1) uses flanking_size/2 (the model's required context); set 0 to extract the bare gene body (legacy behavior).")
    parser_predict.add_argument('--threshold', '-t', type=probability, default=1e-6, help="Threshold to determine acceptor and donor sites")
    parser_predict.add_argument('--predict-all', '-p', action='store_true', required=False, help="Writes all collected predictions to an intermediate file (Warning: on full genomes, will consume much space.)")
    parser_predict.add_argument('--debug', '-D', action='store_true', required=False, help="Run in debug mode (debug statements are printed to stderr)")
    '''AM: very optional flags below vv'''
    parser_predict.add_argument('--hdf-threshold', type=int, default=0, help='Maximum size before reading sequence into an HDF file for storage')
    parser_predict.add_argument('--flush-threshold', type=positive_int, default=500, help='Maximum number of predictions before flushing to file')
    parser_predict.add_argument('--split-threshold', type=positive_int, default=1500000, help='Maximum length of FASTA entry before splitting')
    parser_predict.add_argument('--chunk-size', type=positive_int, default=100, help='Chunk size for loading HDF5 dataset')


def parse_args_variant(subparsers):
    """Register the ``variant`` subcommand and its arguments (annotate a VCF with splicing delta scores)."""
    parser_variant = subparsers.add_parser('variant', help='Label genetic variations with their predicted effects on splicing.')
    parser_variant.add_argument('-R', '--ref-genome', metavar='reference', required=True, help='path to the reference genome fasta file')
    parser_variant.add_argument('-A', '--annotation', metavar='annotation', required=True, help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                                                                                'package), "grch38" (GENCODE V24 canonical annotation file in '
                                                                                'package), or path to a similar custom gene annotation file')
    parser_variant.add_argument('-I', '--input-vcf', metavar='input', nargs='?', default=sys.stdin, help='path to the input VCF file, defaults to standard in')
    parser_variant.add_argument('-O', '--output-vcf', metavar='output', nargs='?', default=sys.stdout, help='path to the output VCF file, defaults to standard out')
    parser_variant.add_argument('-D', '--distance', metavar='distance', nargs='?', default=50, type=int, choices=range(0, 5000),
                                    help='maximum distance between the variant and gained/lost splice '
                                        'site, defaults to 50')
    parser_variant.add_argument('-M', '--mask', metavar='mask', nargs='?', default=0, type=int, choices=[0, 1], 
                                    help='mask scores representing annotated acceptor/donor gain and '
                                        'unannotated acceptor/donor loss, defaults to 0')
    '''AM: newly added flags below vv'''
    parser_variant.add_argument('--model', '-m', required=True, type=str, help='Path to a model file or ensemble directory; SpliceAI selects original Keras models with -t keras -f 10000')
    parser_variant.add_argument('--flanking-size', '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_variant.add_argument('--model-type', '-t', type=str, choices=['keras', 'pytorch'], default='pytorch', help='Type of model file (keras or pytorch)')
    parser_variant.add_argument('--precision', '-p', type=int, default=2, help='Number of decimal places to round the output scores')
    parser_variant.add_argument('--batch-size', '-b', type=positive_int, default=1, help='Number of windows per GPU forward pass. >1 enables batched inference '
                                '(pytorch only) for large speedups on many-variant inputs; 1 (default) preserves the exact original per-variant path')


def build_parser():
    """Build the dependency-free parser used by CLI, examples and generated reference."""
    parser = argparse.ArgumentParser(description='OpenSpliceAI toolkit to help you retrain your own splice site predictor')
    # Create a parent subparser to house the common subcommands.
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands: create-data, train, calibrate, predict, transfer, variant')
    parse_args_create_data(subparsers)
    parse_args_train(subparsers)
    # parse_args_test(subparsers)
    parse_args_calibrate(subparsers)
    parse_args_transfer(subparsers)
    parse_args_predict(subparsers)
    parse_args_variant(subparsers)
    return parser


def parse_args(arglist=None):
    """Parse an argument list or sys.argv with the public command contracts."""
    parser = build_parser()
    args = parser.parse_args(arglist)
    if hasattr(args, 'random_seed') and not 0 <= args.random_seed < 2**32:
        parser.error('--random-seed must be in [0, 2**32)')
    if hasattr(args, 'precision') and not 0 <= args.precision <= 12:
        parser.error('--precision must be between 0 and 12')
    return args


def main(arglist=None):
    """Console entry point: print the banner, parse args, and dispatch to the chosen subcommand.

    Dispatches ``args.command`` to its package: ``create-data`` runs
    ``create_datafile`` then ``create_dataset`` (and ``verify_h5`` if requested),
    and ``train`` / ``calibrate`` / ``transfer`` / ``predict`` / ``variant`` each
    call their respective entry-point function.
    """
    # ANSI Shadow
    banner = '''
============================================================
Deep learning framework that decodes splicing across species
============================================================


 ██████╗ ██████╗ ███████╗███╗   ██╗███████╗██████╗ ██╗     ██╗ ██████╗███████╗ █████╗ ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██╔══██╗██║     ██║██╔════╝██╔════╝██╔══██╗██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████╗██████╔╝██║     ██║██║     █████╗  ███████║██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════██║██╔═══╝ ██║     ██║██║     ██╔══╝  ██╔══██║██║
╚██████╔╝██║     ███████╗██║ ╚████║███████║██║     ███████╗██║╚██████╗███████╗██║  ██║██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚══════╝╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝
    '''
    print(banner, file=sys.stderr)
    print(f"{__VERSION__}\n", file=sys.stderr)
    args = parse_args(arglist)
    
    try:
        return dispatch(args)
    except (ValueError, OSError) as error:
        print(f'OpenSpliceAI error: {error}', file=sys.stderr)
        raise SystemExit(1) from error


def dispatch(args):
    """Run a parsed command; library callers receive ordinary exceptions."""
    if args.command == 'create-data':
        from openspliceai.create_data import create_datafile, create_dataset, verify_h5_file
        create_datafile.create_datafile(args)
        create_dataset.create_dataset(args)
        if args.verify_h5:
            verify_h5_file.verify_h5(args)
    elif args.command == 'train':
        from openspliceai.train import train
        train.train(args)
    # elif args.command == 'test':
    #     from openspliceai.test import test
    #     test.test(args)
    elif args.command == 'calibrate':
        from openspliceai.calibrate import calibrate
        calibrate.calibrate(args)
    elif args.command == 'transfer':
        from openspliceai.transfer import transfer
        transfer.transfer(args)
    elif args.command == 'predict':
        from openspliceai.predict import predict
        predict.predict_cli(args)
    elif args.command == 'variant':
        from openspliceai.variant import variant
        variant.variant(args)
