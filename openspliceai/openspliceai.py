"""
Filename: openspliceai.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Main script to run OpenSpliceAI toolkit.
"""

import argparse
import sys
from openspliceai import header
from openspliceai.create_data import create_datafile, create_dataset, verify_h5_file
from openspliceai.train import train
# from openspliceai.test import test
from openspliceai.calibrate import calibrate
from openspliceai.transfer import transfer
from openspliceai.predict import predict
from openspliceai.variant import variant

__VERSION__ = header.__version__

def parse_args_create_data(subparsers):
    parser_create_data = subparsers.add_parser('create-data', help='Create dataset for your genome for SpliceAI model training')
    parser_create_data.add_argument('--annotation-gff', type=str, required=True, help='Path to the GFF file')
    parser_create_data.add_argument('--genome-fasta', type=str, required=True, help='Path to the FASTA file')
    parser_create_data.add_argument('--output-dir', type=str, required=True, help='Output directory to save the data')
    parser_create_data.add_argument('--parse-type', type=str, default='canonical', choices=['canonical', 'all_isoforms'], help='Type of transcript processing')
    parser_create_data.add_argument('--biotype', type=str, default='protein-coding', choices=['protein-coding', 'non-coding', 'all'], help='Biotype of transcript processing')
    parser_create_data.add_argument('--chr-split', type=str, choices=['train-test','test'], default='train-test', help='Whether to obtain testing or both training and testing groups')
    parser_create_data.add_argument('--split-method', type=str, choices=['random', 'human'], default='random', help='Chromosome split method for training and testing dataset')
    parser_create_data.add_argument('--val_split_ratio', type=float, default=0.1, help='Ratio of training and validation dataset')
    parser_create_data.add_argument('--split-ratio', type=float, default=0.8, help='Ratio of training and testing dataset')
    parser_create_data.add_argument('--canonical-only', action='store_true', default=False, help='Flag to obtain only canonical splice site pairs')
    parser_create_data.add_argument('--flanking-size', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_create_data.add_argument('--verify-h5', action='store_true', default=False, help='Verify the generated HDF5 file(s)')
    parser_create_data.add_argument('--remove-paralogs', action='store_true', default=False, help='Remove paralogous sequences between training and testing dataset')
    parser_create_data.add_argument('--min-identity', type=float, default=0.8, help='Minimum minimap2 alignment identity for paralog removal between training and testing dataset')
    parser_create_data.add_argument('--min-coverage', type=float, default=0.5, help='Minimum minimap2 alignment coverage for paralog removal between training and testing dataset')
    parser_create_data.add_argument('--write-fasta', action='store_true', default=False, help='Flag to write out sequences into fasta files')


def parse_args_train(subparsers):
    parser_train = subparsers.add_parser('train', help='Train the SpliceAI model')
    parser_train.add_argument('--epochs', '-n', type=int, default=10, help='Number of epochs for training')
    parser_train.add_argument('--scheduler', '-s', type=str, default="MultiStepLR", choices=["MultiStepLR", "CosineAnnealingWarmRestarts"], help="Learning rate scheduler")
    parser_train.add_argument('--early-stopping', '-E', action='store_true', default=False, help='Enable early stopping')
    parser_train.add_argument("--patience", '-P', type=int, default=2, help="Number of epochs to wait before early stopping")
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
    parser_calibrate = subparsers.add_parser('calibrate', help='Calibrate the SpliceAI model')
    parser_calibrate.add_argument('--epochs', '-n', type=int, default=10, help='Number of epochs for training')
    parser_calibrate.add_argument('--early-stopping', '-E', action='store_true', default=False, help='Enable early stopping')
    parser_calibrate.add_argument("--patience", '-P', type=int, default=2, help="Number of epochs to wait before early stopping")
    parser_calibrate.add_argument("--output-dir", '-o', type=str, required=True, help="Output directory for model checkpoints and logs")
    parser_calibrate.add_argument("--project-name", '-p', type=str, required=True, help="Project name for the fine-tuning experiment")
    parser_calibrate.add_argument("--exp-num", '-e', type=int, default=0, help="Experiment number")
    parser_calibrate.add_argument("--flanking-size", '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_calibrate.add_argument("--random-seed", '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_calibrate.add_argument("--temperature-file", '-T', type=str, default=None, required=False, help="Path to the temperature file")
    parser_calibrate.add_argument("--pretrained-model", '-m', type=str, required=True, help="Path to the pre-trained model")
    parser_calibrate.add_argument("--train-dataset", '-train', type=str, required=True, help="Path to the training dataset")
    parser_calibrate.add_argument("--test-dataset", '-test', type=str, required=True, help="Path to the testing dataset")
    parser_calibrate.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for fine-tuning")


def parse_args_transfer(subparsers):
    parser_transfer = subparsers.add_parser('transfer', help='transfer a pre-trained SpliceAI model on new data.')
    parser_transfer.add_argument('--epochs', '-n', type=int, default=10, help='Number of epochs for training')
    parser_transfer.add_argument('--scheduler', '-s', type=str, default="MultiStepLR", choices=["MultiStepLR", "CosineAnnealingWarmRestarts"], help="Learning rate scheduler")
    parser_transfer.add_argument('--early-stopping', '-E', action='store_true', default=False, help='Enable early stopping')
    parser_transfer.add_argument("--patience", '-P', type=int, default=2, help="Number of epochs to wait before early stopping")
    parser_transfer.add_argument("--output-dir", '-o', type=str, required=True, help="Output directory for model checkpoints and logs")
    parser_transfer.add_argument("--project-name", '-p', type=str, required=True, help="Project name for the fine-tuning experiment")
    parser_transfer.add_argument("--exp-num", '-e', type=int, default=0, help="Experiment number")
    parser_transfer.add_argument("--flanking-size", '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_transfer.add_argument("--random-seed", '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_transfer.add_argument("--pretrained-model", '-m', type=str, required=True, help="Path to the pre-trained model")
    parser_transfer.add_argument("--train-dataset", '-train', type=str, required=True, help="Path to the training dataset")
    parser_transfer.add_argument("--test-dataset", '-test', type=str, required=True, help="Path to the testing dataset")
    parser_transfer.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for fine-tuning")
    parser_transfer.add_argument("--unfreeze-all", '-A', action='store_true', default=True, help='Unfreeze all layers for fine-tuning')
    parser_transfer.add_argument("--unfreeze", '-u', type=int, default=1, help="Number of layers to unfreeze for fine-tuning")
    parser_transfer.add_argument('--rbp-expression', type=str, help='Path to an RBP expression vector (JSON/NPY) used for FiLM conditioning')
    parser_transfer.add_argument('--film-start-layer', type=int, default=None, help='1-based residual unit index to start FiLM conditioning (defaults to half of the network)')


def parse_args_predict(subparsers):
    parser_predict = subparsers.add_parser('predict', help='Predict splice sites in a given sequence using the SpliceAI model')
    parser_predict.add_argument('--input-sequence', '-i', type=str, required=True, help="Path to FASTA file of the input sequence")
    parser_predict.add_argument('--model', '-m', type=str, required=True, help='Path to a PyTorch SpliceAI model file')
    parser_predict.add_argument('--flanking-size', '-f', type=int, required=True, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_predict.add_argument('--output-dir', '-o', type=str, default="./predict_out", help='Output directory to save the data')
    parser_predict.add_argument('--annotation-file', '-a', type=str, required=False, help="Path to GFF file of coordinates for genes")
    parser_predict.add_argument('--threshold', '-t', type=float, default=1e-6, help="Threshold to determine acceptor and donor sites")
    parser_predict.add_argument('--predict-all', '-p', action='store_true', required=False, help="Writes all collected predictions to an intermediate file (Warning: on full genomes, will consume much space.)")
    parser_predict.add_argument('--debug', '-D', action='store_true', required=False, help="Run in debug mode (debug statements are printed to stderr)")
    '''AM: very optional flags below vv'''
    parser_predict.add_argument('--hdf-threshold', type=int, default=0, help='Maximum size before reading sequence into an HDF file for storage')
    parser_predict.add_argument('--flush-threshold', type=int, default=500, help='Maximum number of predictions before flushing to file')
    parser_predict.add_argument('--split-threshold', type=int, default=1500000, help='Maximum length of FASTA entry before splitting')
    parser_predict.add_argument('--chunk-size', type=int, default=100, help='Chunk size for loading HDF5 dataset')


def parse_args_variant(subparsers):
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
    parser_variant.add_argument('--model', '-m', default="SpliceAI", type=str, help='Path to a SpliceAI model file, or path to a directory of SpliceAI models, or "SpliceAI" for the default model')
    parser_variant.add_argument('--flanking-size', '-f', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_variant.add_argument('--model-type', '-t', type=str, choices=['keras', 'pytorch'], default='pytorch', help='Type of model file (keras or pytorch)')
    parser_variant.add_argument('--precision', '-p', type=int, default=2, help='Number of decimal places to round the output scores')
    parser_variant.add_argument('--rbp-expression', type=str, help='Path to an RBP expression vector (JSON/NPY) used when the model includes FiLM layers')
 

def parse_args(arglist):
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
    if arglist is not None:
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()
    return args


def main(arglist=None):
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
    
    if args.command == 'create-data':
        create_datafile.create_datafile(args)
        create_dataset.create_dataset(args)
        if args.verify_h5:
            verify_h5_file.verify_h5(args)
    elif args.command == 'train':
        train.train(args)
    # elif args.command == 'test':
    #     test.test(args)
    elif args.command == 'calibrate':
        calibrate.calibrate(args)
    elif args.command == 'transfer':
        transfer.transfer(args)
    elif args.command == 'predict':
        predict.predict_cli(args)
    elif args.command == 'variant':
        variant.variant(args)
