from openspliceai.checkpoints import unpack_checkpoint, CalibratedSpliceAI, CheckpointError
import os
import sys
import glob
import re
from urllib.parse import quote, unquote
import numpy as np
from openspliceai.model_config import model_hyperparameters
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
import h5py
import time
from contextlib import ExitStack
from pyfaidx import Fasta
from openspliceai.train_base.openspliceai import SpliceAI
import openspliceai.predict.utils as utils

################
##   STEP 1   ##
################

def process_gff(fasta_file, gff_file, output_dir, gene_flank=0):
    """
    Processes a GFF file to extract gene regions and creates a new FASTA file
    with the extracted gene sequences.

    Parameters:
    - gff_file: Path to the GFF file.
    - fasta_file: Path to the input FASTA file.
    - output_fasta_file: Path to the output FASTA file.
    - gene_flank: bp of REAL genomic sequence to include on each side of every
      gene (clamped to the contig ends). The deep model needs flanking_size/2 of
      true context on each side; without it the boundaries of every extracted
      gene are padded with 'N' downstream, which suppresses splice-site scores
      near the gene ends. Pass flanking_size//2 (default 0 keeps the legacy
      bare-gene-body behavior).
    """
    # define output FASTA file
    output_fasta_file = f'{output_dir}{os.path.splitext(os.path.basename(fasta_file))[0]}_genes.fa'

    # read the input FASTA file
    with Fasta(fasta_file) as fasta:

        # open the output FASTA file for writing
        count = 0
        with open(output_fasta_file, 'w') as output_fasta:
            # read the GFF file
            with open(gff_file, 'r') as gff:
                for line in gff:
                    if line.startswith('#'):
                        continue

                    # split the GFF line into fields
                    fields = line.strip().split('\t')
                    if len(fields) < 9:
                        print(f'\t[ERR] line does not have enough fields:\n{line}')
                        continue  # lines with not enough fields are erroneous?

                    # extract relevant information from the GFF fields
                    seqid = fields[0]
                    feature_type = fields[2]
                    start = int(fields[3])
                    end = int(fields[4])
                    strand = fields[6]
                    attributes = fields[8]

                    # process only gene features
                    if feature_type != 'gene':
                        continue

                    # extract gene ID, or if not then Name, from the attributes
                    gene_id = 'unknown_gene'
                    for attribute in attributes.split(';'):
                        if attribute.startswith('ID=') or attribute.startswith('Name='):
                            gene_id = attribute.split('=')[1]
                            break

                    # extend the gene region by gene_flank bp of REAL genomic context
                    # on each side (clamped to the contig bounds) so the model is not
                    # fed 'N' padding at the gene boundaries; report the extended
                    # coordinates in the header so the BED output still maps to genome
                    contig_len = len(fasta[seqid])
                    ext_start = max(1, start - gene_flank)
                    ext_end = min(contig_len, end + gene_flank)

                    # extract the gene sequence (+ flank) from the FASTA file
                    sequence = fasta[seqid][ext_start-1:ext_end]  # adjust for 0-based indexing

                    # reverse complement the sequence if on the negative strand
                    if strand == '-':
                        sequence = sequence.reverse.complement

                    # write the gene sequence to the output FASTA file
                    output_fasta.write(f'>{gene_id} {seqid}:{ext_start}-{ext_end}({strand})\n')
                    output_fasta.write(str(sequence) + '\n')
                    count += 1

    print(f"\t[INFO] {count} gene sequences have been extracted to {output_fasta_file}")

    return output_fasta_file

def split_fasta(genes, split_fasta_file, CL_max, split_fasta_threshold, neg_strands=None):
    """Split with inference halos and nonoverlapping output ownership intervals.

    OSAI_CORE is a zero-based half-open interval in the segment's orientation;
    OSAI_ORIGIN preserves its original output name. Genomic headers support any
    contig name and reverse-complement coordinate mapping on the minus strand.
    """
    if split_fasta_threshold < 1 or CL_max < 0 or CL_max % 2:
        raise ValueError('Split threshold must be positive and context nonnegative/even')
    pattern = re.compile(r'(.*?)([^\s:]+):(\d+)-(\d+)\(([-+.])\)')
    with open(split_fasta_file, 'w') as output:
        for record in genes:
            length = len(record)
            if length == 0:
                raise ValueError(f'Empty FASTA entry: {record.name}')
            original = record.long_name
            match = pattern.search(original)
            reverse = neg_strands is not None and record.name in neg_strands
            if match:
                prefix, chrom, absolute_start, absolute_end, strand = match.groups()
                absolute_start, absolute_end = int(absolute_start), int(absolute_end)
            else:
                prefix, chrom, absolute_start, absolute_end, strand = '', record.name, 1, length, '-' if reverse else '.'
            if absolute_start < 1 or absolute_end-absolute_start+1 != length:
                raise ValueError(f'FASTA coordinate span does not match sequence length: {original}')
            source = genes[record.name]
            if reverse:
                source = source[:].reverse.complement
            for core_start in range(0, length, split_fasta_threshold):
                core_end = min(length, core_start+split_fasta_threshold)
                first = max(0, core_start-CL_max//2)
                last = min(length, core_end+CL_max//2)
                if strand == '-':
                    start, end = absolute_end-last+1, absolute_end-first
                else:
                    start, end = absolute_start+first, absolute_start+last-1
                name = f'{prefix}{chrom}:{start}-{end}({strand})'
                if length > split_fasta_threshold:
                    name += f' OSAI_CORE={core_start-first}:{core_end-first} OSAI_ORIGIN={quote(original, safe="")}'
                output.write(f'>{name}\n{source[first:last].seq}\n')


def get_sequences(fasta_file, output_dir, CL_max, hdf_threshold_len=0, split_fasta_threshold=1500000, neg_strands=None, debug=False):
    """
    Extract sequences for each protein-coding gene, process them based on strand orientation,
    and save data in file depending on the sequence size (HDF file if sequence >HDF_THRESHOLD_LEN,
    else temp file).

    Parameters:
    - fasta_file: Path to the FASTA file.
    - output_dir: Directory to save the output files.
    - neg_strands (list): List of IDs for fasta entries where the sequence is on the reverse strand. Default is None.

    Returns:
    - Path to datafile.
    """

    # detect sequence length, determine file saving method to use
    total_length = 0
    use_hdf = False
    need_splitting = False

    # NOTE: always creates uppercase sequence, uses [1,0]-indexed sequence names (does not affect slicing), takes simple name from FASTA
    with ExitStack() as resources:
        genes = resources.enter_context(Fasta(fasta_file, one_based_attributes=True, read_long_names=False, sequence_always_upper=True))

        for record in genes:
            record_length = len(genes[record.name])
            total_length += record_length
            if not use_hdf and total_length > hdf_threshold_len:
                use_hdf = True
                print(f'\t[INFO] Input FASTA sequences over {hdf_threshold_len}: use_hdf = True.')
            if not need_splitting and record_length > split_fasta_threshold:
                need_splitting = True
                print(f'\t[INFO] Input FASTA contains sequence(s) over {split_fasta_threshold}: need_splitting = True')
            if use_hdf and need_splitting:
                break

        if need_splitting:
            split_fasta_file = f'{output_dir}{os.path.splitext(os.path.basename(fasta_file))[0]}_split.fa'
            print(f'\t[INFO] Splitting {fasta_file}.')

            split_fasta(genes, split_fasta_file, CL_max, split_fasta_threshold, neg_strands=neg_strands)

            # re-loads the pyfaidx Fasta object with split genes
            genes = resources.enter_context(Fasta(split_fasta_file, one_based_attributes=True, read_long_names=True, sequence_always_upper=True)) # need long name to handle duplicate seqids after splits
            print(f"\t[INFO] Saved and loaded {split_fasta_file}.")

        NAME = [] # Gene Header
        SEQ  = [] # Sequences
        short_records = []  # records shorter than the model's required context

        # obtain the headers and sequences from FASTA file
        for record in genes:
            seq_id = record.long_name
            sequence = genes[record.name][:].seq
            match = re.search(r"([^\s:]+):(\d+)-(\d+)\(([-+.])\)", seq_id)
            if not sequence or (match and (int(match[2]) < 1 or int(match[3])-int(match[2])+1 != len(sequence))):
                raise ValueError(f"FASTA coordinate span does not match sequence length: {seq_id}")

            # reverse strand if explicitly specified, name with strand info
            if neg_strands is not None and record.name in neg_strands:
                seq_id = str(seq_id) + ':-'
                # ``sequence`` was already materialised to a str (.seq) above, so reverse-
                # complement via the pyfaidx Sequence object and take its .seq (a bare str
                # has no .reverse/.complement -> this path previously crashed).
                sequence = genes[record.name][:].reverse.complement.seq
            else:
                seq_id = str(seq_id) + ':+'

            # flag sequences too short to supply the model's flanking context: they
            # are padded with 'N' downstream, which strongly suppresses splice scores
            if len(sequence) < CL_max:
                short_records.append((record.name, len(sequence)))

            seq_id = seq_id[:-2] + f' OSAI_LENGTH={len(sequence)}' + seq_id[-2:]
            NAME.append(seq_id)
            SEQ.append(str(sequence))

        if short_records:
            preview = ', '.join(f'{n} ({l} bp)' for n, l in short_records[:5])
            more = '' if len(short_records) <= 5 else f' (+{len(short_records) - 5} more)'
            print(f"\t[WARN] {len(short_records)} input sequence(s) are shorter than the model's "
                  f"required context (CL_max={CL_max} bp): {preview}{more}.", file=sys.stderr)
            print(f"\t[WARN] Short sequences are padded with 'N' on both ends, which suppresses "
                  f"donor/acceptor scores. Supply >= {CL_max // 2} bp of REAL flanking sequence on "
                  f"each side — extract the gene +/- {CL_max // 2} bp, or pass the whole chromosome "
                  f"with -a/--annotation so genes are scored in their genomic context.", file=sys.stderr)

        # write the sequences to datafile
        if use_hdf:
            datafile_path = f'{output_dir}datafile.h5'
            dt = h5py.string_dtype(encoding='utf-8')
            with h5py.File(datafile_path, 'w') as datafile: # hdf5 information file
                datafile.create_dataset('NAME', data=np.asarray(NAME, dtype=dt), dtype=dt)
                datafile.create_dataset('SEQ', data=np.asarray(SEQ, dtype=dt), dtype=dt)
        else:
            datafile_path = f'{output_dir}datafile.txt'
            with open(datafile_path, 'w') as datafile: # temp sequence file
                for name, seq in zip(NAME, SEQ):
                    datafile.write(f'{name}\n{seq}\n')

        if debug:
            print(f'\t[DEBUG] len(NAME): {len(NAME)}, len(SEQ): {len(SEQ)}', file=sys.stderr)

    return datafile_path, NAME, SEQ

    # check_and_count_motifs(gene_seq, labels, gene.strand) # maybe adapt to count motifs that were found in the predicted file...


################
##   STEP 2   ##
################

def create_datapoints(input_string, SL, CL_max, debug=False):
    """
    Parameters:
    - input_string (str): The nucleotide sequence.

    Returns:
    - X (np.ndarray): The one-hot encoded input nucleotide sequence.
    """

    def reformat_data(X0):
        """
        Breaks up an input sequence into overlapping windows of size SL + CL_max.

        Parameters:
        - X0 (numpy.ndarray): Original sequence data as an array of integer encodings.
        - (global) CL_max: Maximum context length for sequence prediction (flanking size sum).
        - (global) SL: Sequence length for prediction, default = 5000.

        Returns:
        - numpy.ndarray: Reformatted sequence data.
        """

        if debug:
            print('\n\t[DEBUG] reformat_data', file=sys.stderr)
            print('\tlen(X0)', len(X0), file=sys.stderr)
            print('\tSL', SL, ' CL_max', CL_max, file=sys.stderr)
        # Calculate the number of data points needed
        num_points = utils.ceil_div(len(X0) - CL_max, SL) # NOTE: subtracting the flanking here because X0 is already padded at the ends by create_datapoints and only want window on actual sequence length
        if debug:
            print('\tnum_points', num_points, file=sys.stderr)
        # Initialize arrays to hold the reformatted data
        Xd = np.zeros((num_points, SL + CL_max))
        if debug:
            print('\tXd.shape', Xd.shape, file=sys.stderr)
        # Pad the end sequence to ensure divisibility
        padding_length = ((SL + CL_max) * num_points) - len(X0)
        X0 = np.pad(X0, (0, padding_length), 'constant', constant_values=0)
        if debug:
            print('\tpadding_length', padding_length, file=sys.stderr)
            print('\tnew len(X0)', len(X0), file=sys.stderr)

        # Fill the initialized arrays with data in blocks
        for i in range(num_points):
            Xd[i] = X0[SL * i : SL * (i + 1) + CL_max]

        return Xd

    def one_hot_encode(Xd):
        """
        Perform one-hot encoding on the input sequence data (Xd).

        Parameters:
        - Xd (numpy.ndarray): An array of integers representing the input sequence data where each nucleotide
            is encoded as an integer (1 for 'A', 2 for 'C', 3 for 'G', 4 for 'T', and 0 for padding).

        Returns:
        - numpy.ndarray: the one-hot encoded input sequence data.
        """

        # One-hot encoding of the inputs:
        # 1: A;  2: C;  3: G;  4: T;  0: padding
        IN_MAP = np.asarray([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        return IN_MAP[Xd.astype('int8')]

    # NOTE: No need to reverse complement the sequence, as sequence is already reverse complemented from previous step

    # Replace all non-ACTG to N
    allowed_chars = {'A', 'C', 'G', 'T'} # NOTE: this will turn all lowercase actg into N! (will not happen in here as seq already uppered)
    seq = ''.join(char if char in allowed_chars else 'N' for char in input_string)

    # Convert to vector array
    seq = 'N' * (CL_max // 2) + seq + 'N' * (CL_max // 2)
    seq = seq.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')

    # One-hot-encode the inputs
    if debug:
        print('\n\t[DEBUG] create_datapoints:', file=sys.stderr)
        print('\tlen(seq):', len(seq), file=sys.stderr)
    X0 = np.asarray(list(map(int, list(seq)))) # convert string to np array
    if debug:
        print('\tX0.shape', X0.shape, file=sys.stderr)
    Xd = reformat_data(X0) # apply window size
    if debug:
        print('\tXd.shape', Xd.shape, file=sys.stderr)
    X = one_hot_encode(Xd) # one-hot encode
    if debug:
        print('\tX', X.shape, file=sys.stderr)
    return X

def convert_sequences(datafile_path, output_dir, SL, CL_max, chunk_size=100, SEQ=None, debug=False):
    '''
    Script to convert datafile into a one-hot encoded dataset ready to input to model.
    If HDF5 file used, data is chunked for loading.

    Parameters:
    - datafile_path: path to the datafile
    - output_dir: output directory path
    - SEQ: list of sequences

    Returns:
    - Path to the dataset.
    - LEN: list of how many chunks each gene takes up
    '''

    def process_chunk(NEW_CHUNK_SIZE, i, SEQ, LEN, out_h5f):
        '''Processes an entire chunk and writes to dataset.'''
        X_batch = []
        for j in range(NEW_CHUNK_SIZE):
            idx = i * chunk_size + j

            seq_decode = SEQ[idx]
            X = create_datapoints(seq_decode, SL, CL_max, debug=debug)
            if debug:
                print('\tX.shape:', X.shape, file=sys.stderr)
            LEN.append(len(X))
            X_batch.extend(X)

        # Convert batches to arrays and save as HDF5
        X_batch = np.asarray(X_batch).astype('int8')
        if debug:
            print('\tNEW_CHUNK_SIZE:', NEW_CHUNK_SIZE, file=sys.stderr)
            print("\tX_batch.shape:", X_batch.shape, file=sys.stderr)
            utils.log_memory_usage()
        out_h5f.create_dataset('X' + str(i), data=X_batch)

    # determine whether to convert an h5 or txt file
    file_ext = os.path.splitext(datafile_path)[1]
    assert file_ext in ['.h5', '.txt']
    use_h5 = file_ext == '.h5'

    # read the given input file if both datastreams were not provided
    if SEQ is None:
        print(f"\t[INFO] Reading {datafile_path} ... ")
        if use_h5:
            with h5py.File(datafile_path, 'r') as in_h5f:
                SEQ = in_h5f['SEQ'][:]
        else:
            SEQ = []
            with open(datafile_path, 'r') as in_file:
                lines = in_file.readlines()
                for i, line in enumerate(lines):
                    if i % 2 == 1:
                        SEQ.append(line)
    else:
        print('\t[INFO] NAME and SEQ data provided, skipping reading ...')

    num_seqs = len(SEQ)
    if debug:
        print('\n\t[DEBUG] convert_sequences', file=sys.stderr)
        print("\tnum_seqs: ", num_seqs, file=sys.stderr)

    LEN = [] # number of batches for each sequence

    # write to h5 file by chunking and one-hot encoding inputs
    if use_h5:
        dataset_path = f'{output_dir}dataset.h5'

        print(f"\t[INFO] Writing {dataset_path} ... ")

        with h5py.File(dataset_path, 'w') as out_h5f:

            # create dataset
            num_chunks = utils.ceil_div(num_seqs, chunk_size) # ensures that even if num_seqs < CHUNK_SIZE, will still create a chunk
            for i in tqdm(range(num_chunks), desc='Processing chunks...'):

                # each dataset has CHUNK_SIZE genes
                if i == num_chunks - 1: # if last chunk, process remainder or full chunk size if no remainder
                    NEW_CHUNK_SIZE = num_seqs % chunk_size or chunk_size
                else:
                    NEW_CHUNK_SIZE = chunk_size

                # chunk conversion
                process_chunk(NEW_CHUNK_SIZE, i, SEQ, LEN, out_h5f)

    # convert to tensor and write directly to a binary PyTorch file for quick loading
    else:
        dataset_path = f'{output_dir}/dataset.pt'

        print(f"\t[INFO] Writing {dataset_path} ... ")
        X_all = []
        for idx in range(num_seqs):
            seq_decode = SEQ[idx]
            if isinstance(seq_decode, bytes):
                seq_decode = seq_decode.decode('ascii')
            X = create_datapoints(seq_decode, SL, CL_max, debug=debug)
            if debug:
                print('\tX.shape:', X.shape, file=sys.stderr)
                utils.log_memory_usage()
            LEN.append(len(X))
            X_all.extend(X)

        # convert batches to a tensor
        X_tensor = torch.from_numpy(np.asarray(X_all, dtype=np.int8))

        # save as a binary file
        torch.save(X_tensor, dataset_path)

    return dataset_path, LEN


################
##   STEP 3   ##
################

def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" and torch.backends.mps.is_available() else "cpu"
    return torch.device(device_str)

def load_pytorch_models(model_path, device, SL, CL):
    """
    Loads a SpliceAI PyTorch model from given state, inferring device.

    Params:
    - model_path (str): Path to the model state dict, or a directory of models
    - CL (int): Context length parameter for model conversion.

    Returns:
    - loaded_models (list): SpliceAI model(s) loaded with given state.
    """

    def load_model(device, flanking_size):
        """Loads the given model."""
        # Hyper-parameters:
        # L: Number of convolution kernels
        # W: Convolution window size in each residual unit
        # AR: Atrous rate in each residual unit
        L, N_GPUS, W, AR, BATCH_SIZE = model_hyperparameters(flanking_size)
        CL = 2 * np.sum(AR*(W-1))

        print(f"\t[INFO] Context nucleotides {CL}")
        print(f"\t[INFO] Sequence length (output): {SL}")

        model = SpliceAI(L, W, AR).to(device)
        params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}

        return model, params

    # Load all model state dicts given the supplied model path
    if os.path.isdir(model_path):
        # Find all '*.pth' and '*.pt' files in the directory
        model_files = glob.glob(os.path.join(model_path, '*.pth')) + glob.glob(os.path.join(model_path, '*.pt')) # gets all PyTorch models from supplied directory
        model_files = sorted(model_files)
        if not model_files:
            print(f"\t[ERR] No PyTorch model files found in directory: {model_path}")
            raise CheckpointError(f'Unable to load checkpoint at {model_path}')

        models = []
        for model_file in model_files:
            try:
                model = torch.load(model_file, map_location=device, weights_only=True)
                models.append(model)
            except Exception as e:
                print(f"\t[ERR] Error loading PyTorch model from file {model_file}: {e}. Aborting ensemble load.")
                raise CheckpointError(f'Unable to load checkpoint at {model_path}: {e}') from e

        if not models:
            print(f"\t[ERR] No valid PyTorch models found in directory: {model_path}")
            raise CheckpointError(f'Unable to load checkpoint at {model_path}')

    elif os.path.isfile(model_path):
        try:
            models = [torch.load(model_path, map_location=device, weights_only=True)]
        except Exception as e:
            print(f"\t[ERR] Error loading PyTorch model from file {model_path}: {e}.")
            raise CheckpointError(f'Unable to load checkpoint at {model_path}')

    else:
        print(f"\t[ERR] Invalid path: {model_path}")
        raise CheckpointError(f'Unable to load checkpoint at {model_path}')

    # Load state of model to device
    # NOTE: supplied model paths should be state dicts, not model files
    loaded_models = []

    mismatch_hint = (
        "\t[HINT] Ensure the provided checkpoint was trained with the same --flanking-size "
        f"({CL}) you passed to predict. For the bundled releases, pick a model from the "
        f"`.../{CL}nt/` directory."
    )

    for state_dict in models:
        model, params = load_model(device, CL)  # loads new SpliceAI model with correct hyperparams
        try:
            state_dict, temperature = unpack_checkpoint(state_dict, CL)
            model.load_state_dict(state_dict)   # loads state dict
        except (RuntimeError, ValueError) as e:
            err_msg = str(e)
            if "size mismatch" in err_msg or "shape" in err_msg:
                print("\t[WARN] Cannot load model due to incompatible tensor shapes.")
                print("\t[WARN] This typically happens when the checkpoint was trained with a different flanking size.")
                print(mismatch_hint)
                raise CheckpointError(f'Unable to load checkpoint at {model_path}: {e}') from e
            print(f"\t[ERR] Error processing model for device: {err_msg}. Aborting ensemble load.")
            raise CheckpointError(f'Unable to load checkpoint at {model_path}: {e}') from e

        if temperature is not None:
            model = CalibratedSpliceAI(model, temperature)
        model = model.to(device)                # puts model on device
        model.eval()                            # puts model in evaluation mode
        loaded_models.append(model)             # appends model to list of loaded models

    if not loaded_models:
        print("\t[ERR] No models were successfully loaded to the device.")
        raise CheckpointError(f'Unable to load checkpoint at {model_path}')

    return loaded_models, params # NOTE: returns the last params, assuming all models have the same hyperparameters



################
##   STEP 4   ##
################

# only used when file in hdf5
def load_shard(h5f, batch_size, shard_idx):
    '''
    Loads a selected shard from HDF5 file.

    Parameters:
    - h5f: an OPEN dataset file in read mode
    '''

    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    X = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

def flush_predictions(predictions, file_path):
    """
    Flush predictions continuously to HDF5 file, in cases where too many predictions are currently in memory.

    Parameters:
    - predictions: Tensor of predictions to save.
    - file_path: Path to the file where predictions are saved.
    """
    with h5py.File(file_path, 'a') as f:
        if 'predictions' in f:
            dataset = f['predictions']
            dataset.resize((dataset.shape[0] + predictions.shape[0]), axis=0)
            dataset[-predictions.shape[0]:] = predictions.numpy()
        else:
            maxshape = (None,) + tuple(predictions.shape[1:])
            f.create_dataset('predictions', data=predictions.numpy(), maxshape=maxshape, chunks=True)

def get_prediction(models, dataset_path, device, batch_size, output_dir, flush_predict_threshold=500, debug=False):
    """
    Parameters:
    - model (torch.nn.Module): The SpliceAI model to be evaluated.
    - dataset_path (path): Path to the selected dataset.
    - device (torch.device): The computational device (CUDA, MPS, CPU).
    - batch_size (int): The batch size for prediction.
    - output_dir (str): Root of output directory for predict file.
    - flush_predict_threshold (int): The number of predictions to collect before flushing to file.

    Returns:
    - Path to predictions binary file
    """
    # define batch_size and count
    count = 0
    print(f'\t[INFO] Batch size: {batch_size}')
    if debug:
        print('\n\t[DEBUG] get_prediction', file=sys.stderr)

    # put model in evaluation mode
    for model in models:
        model.eval()
    print('\t[INFO] Model in evaluation mode.')

    # determine which file to proceed with
    file_ext = os.path.splitext(dataset_path)[1]
    assert file_ext in ['.h5', '.pt']
    use_h5 = file_ext == '.h5'

    if use_h5: # read from the h5 file

        # initialize predict file (to prevent continuous appending)
        predict_path = f'{output_dir}predict.h5'
        pfile = h5py.File(predict_path, 'w')
        pfile.close()

        # read dataset and iterate over shards in index
        h5f = h5py.File(dataset_path, 'r')
        idxs = np.arange(len(h5f.keys()))
        if debug:
            print('\th5 indices:', idxs, file=sys.stderr)

        for i, shard_idx in enumerate(idxs, 1):

            if debug:
                print('\tshard_idx, h5 len', shard_idx, len(h5f[f'X{shard_idx}']), file=sys.stderr)
            loader = load_shard(h5f, batch_size, shard_idx)
            if debug:
                print('\t\tloader batches ', len(loader), file=sys.stderr)

            batch_ypred = [] # list of tensors containing the predictions from model
            pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(idxs)}')
            for batch in pbar:
                DNAs = batch[0].to(device)
                if debug:
                    print('\t\t\tbatch DNA ', len(DNAs), end='', file=sys.stderr)
                DNAs = DNAs.to(torch.float32).to(device)
                # with torch.no_grad():
                    #     y_pred = model(DNAs)
                    # y_pred = y_pred.detach().cpu()
                with torch.no_grad():
                    y_pred = torch.mean(torch.stack([models[m](DNAs).detach().cpu() for m in range(len(models))]), axis=0)

                if debug:
                    print('\tbatch ', len(y_pred), file=sys.stderr)

                batch_ypred.append(y_pred)
                count += 1

                # write predictions to file if exceeding threshold
                if len(batch_ypred) > flush_predict_threshold:
                    print(f'\t[INFO] Reached {flush_predict_threshold} predictions. Flushing to file...', file=sys.stderr)
                    batch_ypred_tensor = torch.cat(batch_ypred, dim=0)
                    flush_predictions(batch_ypred_tensor, predict_path)
                    batch_ypred = []  # reset the list after flushing

                pbar.update(1)

            if debug:
                utils.log_memory_usage()

            pbar.close()

            # flush any remaining predictions
            if batch_ypred:
                print(f'\t[INFO] Flushing remaining {len(batch_ypred)} predictions..', file=sys.stderr)
                batch_ypred_tensor = torch.cat(batch_ypred, dim=0)
                flush_predictions(batch_ypred_tensor, predict_path)

        h5f.close()

    else: # read from the PyTorch file
        predict_path = f'{output_dir}predict.pt'

        # load all data and reshape to (N, channels, length)
        X = torch.load(dataset_path, map_location="cpu", weights_only=True).permute(0, 2, 1)
        X = X.to(torch.float32)
        ds = TensorDataset(X)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        batch_ypred = [] # list of tensors containing the predictions from model
        pbar = tqdm(loader, leave=False, total=len(loader))
        for batch in pbar:
            DNAs = batch[0].to(device)
            DNAs = DNAs.to(torch.float32).to(device)

            with torch.no_grad():
                y_pred = torch.mean(torch.stack([models[m](DNAs).detach().cpu() for m in range(len(models))]), axis=0)

            batch_ypred.append(y_pred)
            count += 1

            pbar.update(1)

        pbar.close()

        print('\t[INFO] Saving predictions...')
        predictions = torch.cat(batch_ypred, dim=0)
        torch.save(predictions, predict_path)

    # preview information
    print(f'\t[INFO] {count} predictions collected.')
    if debug:
        try:
            head_length = 5 if len(batch_ypred) >= 5 else len(batch_ypred)
            print(f'\tPreview predictions: {batch_ypred[:head_length]}', file=sys.stderr)
        except Exception as e:
            print(f'Preview could not be generated: {e}.', file=sys.stderr)
    print(f'\t[INFO] Predictions saved to {predict_path}.')

    return predict_path


################
##   STEP 5   ##
################

def prediction_interval(seq_name, predicted_length):
    """Resolve contig, genomic span, strand, output name, true length and owned core."""
    manual = re.search(r':([-+])$', seq_name)
    strand_override = manual[1] if manual else None
    clean = seq_name[:manual.start()] if manual else seq_name
    metadata = dict(re.findall(r'OSAI_(CORE|ORIGIN|LENGTH)=(\S+)', clean))
    base = re.split(r' OSAI_(?:CORE|ORIGIN|LENGTH)=', clean, maxsplit=1)[0]
    original = unquote(metadata.get('ORIGIN', base))
    match = re.search(r'(?:^|\s|::)([^\s:]+):(\d+)-(\d+)\(([-+.])\)', base)
    if match:
        chrom, start, end, strand = match.groups()
        start, end = int(start), int(end)
        if start < 1 or end < start:
            raise ValueError('FASTA genomic coordinates must be positive and ordered')
        if strand == '.':
            strand = strand_override
        length = min(predicted_length, end-start+1)
    else:
        chrom, start, end, strand = base.split()[0], 1, predicted_length, strand_override or '+'
        length = predicted_length
    if 'LENGTH' in metadata:
        length = min(length, int(metadata['LENGTH']))
        if not match:
            end = length
    if strand not in ('+', '-'):
        return None
    first, last = 0, length
    if 'CORE' in metadata:
        first, last = map(int, metadata['CORE'].split(':'))
    if not 0 <= first <= last <= length:
        raise ValueError('Split ownership must be within the true sequence span')
    return chrom, start, end, strand, original, length, first, last


def write_batch_to_bed(seq_name, gene_predictions, acceptor_bed, donor_bed, threshold=1e-6, debug=False):
    """Write BED6 for true owned positions; inference halos and padding are excluded.

    Acceptor at position p describes [p-1,p); donor describes [p,p+1).
    Coordinates are transformed to the forward genomic frame on both strands.
    For an unannotated FASTA, its record ID is the contig and coordinates are local.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Prediction threshold must be between zero and one')
    if gene_predictions.ndim != 3 or gene_predictions.shape[1] != 3:
        raise ValueError('Predictions must have shape (windows, 3, positions)')
    values = gene_predictions.detach().cpu().permute(0, 2, 1).reshape(-1, 3).numpy()
    if not np.isfinite(values).all():
        raise ValueError('Predictions must be finite')
    interval = prediction_interval(seq_name, len(values))
    if interval is None:
        print(f'Undefined strand. Skipping {seq_name} batch...', file=sys.stderr)
        return
    chrom, start, end, strand, name, length, first, last = interval
    if debug:
        print(f'BED {chrom}:{start}-{end} strand={strand}, owned={first}:{last}, length={length}', file=sys.stderr)
    for position in range(first, last):
        for index, label, handle, offset in ((1, 'Acceptor', acceptor_bed, -1),
                                              (2, 'Donor', donor_bed, 0)):
            score = values[position, index]
            left, right = position+offset, position+offset+1
            if score <= threshold or left < 0 or right > length:
                continue
            if strand == '+':
                left, right = start-1+left, start-1+right
            else:
                left, right = end-right, end-left
            handle.write(f'{chrom}\t{left}\t{right}\t{name}_{label}\t{score:.6f}\t{strand}\n')


# NOTE: need to handle naming when gff file not provided.
def generate_bed(predict_file, NAME, LEN, output_dir, threshold=1e-6, batch_ypred=None, debug=False):
    '''
    Generates the BED file pertaining to the predictions
    '''
    # determine which file to proceed with
    file_ext = os.path.splitext(predict_file)[1]
    assert file_ext in ['.h5', '.pt']
    use_h5 = file_ext == '.h5'

    # load the predictions (if not already there)
    if use_h5:
        print('\t[INFO] Loading predictions...')
        with h5py.File(predict_file, 'r') as h5f:
            batch_ypred_np = h5f['predictions'][:]
        batch_ypred = torch.tensor(batch_ypred_np)
    elif batch_ypred is None:
        batch_ypred = torch.load(predict_file, map_location="cpu", weights_only=True)

    if debug:
        print('\n\t[DEBUG] generate_bed', file=sys.stderr)
        print('\tlen(NAME)', len(NAME), 'len(LEN)', len(LEN), 'sum(LEN)', sum(LEN), 'len(batch_ypred)', len(batch_ypred), file=sys.stderr)
    assert len(NAME) == len(LEN)
    assert sum(LEN) == len(batch_ypred)

    print('\t[INFO] Batch predictions loaded.')
    print(f'\t[INFO] Shape of predictions: {batch_ypred.shape}')
    print(f'\t[INFO] {len(LEN)} targets detected.')

    acceptor_bed_path = f'{output_dir}acceptor_predictions.bed'
    donor_bed_path = f'{output_dir}donor_predictions.bed'

    with open(acceptor_bed_path, 'w') as acceptor_bed, open(donor_bed_path, 'w') as donor_bed:
        start_idx = 0
        for i in tqdm(range(len(NAME)), desc='Generating BED tasks...'):
            seq_name = NAME[i]
            num_batches = LEN[i]  # number of batches for this gene
            end_idx = start_idx + num_batches

            # extract predictions for the current gene
            gene_predictions = batch_ypred[start_idx:end_idx]

            # write to BED file
            write_batch_to_bed(seq_name, gene_predictions, acceptor_bed, donor_bed, threshold=threshold, debug=debug)

            # update the start index for the next gene
            start_idx = end_idx

    print(f"\t[INFO] Acceptor BED file saved to {acceptor_bed_path}")
    print(f"\t[INFO] Donor BED file saved to {donor_bed_path}")


#################
##   STEP 4o   ##
#################

def predict_and_write(models, dataset_path, device, batch_size, NAME, LEN, output_dir, threshold=1e-6, debug=False):
    # define batch_size
    """Stream ensemble probabilities to BED6 while excluding halos and padded positions."""
    print(f'\t[INFO] Batch size: {batch_size}')
    if debug:
        print('\n\t[DEBUG] predict_and_write', file=sys.stderr)
        print('\tlen(NAME)', len(NAME), 'len(LEN)', len(LEN), 'sum(LEN)', sum(LEN), file=sys.stderr)
    assert len(NAME) == len(LEN)

    # put model in evaluation mode
    for model in models:
        model.eval()
    print('\t[INFO] Model in evaluation mode.')

    # determine which file to proceed with
    file_ext = os.path.splitext(dataset_path)[1]
    assert file_ext in ['.h5', '.pt']
    use_h5 = file_ext == '.h5'

    # initialize output BED files
    acceptor_bed_path = f'{output_dir}acceptor_predictions.bed'
    donor_bed_path = f'{output_dir}donor_predictions.bed'
    acceptor_bed = open(acceptor_bed_path, 'w')
    donor_bed = open(donor_bed_path, 'w')

    count = 0
    len_idx = 0
    accumulated_predictions = []
    accumulated_length = 0

    if use_h5: # read from the h5 file

        with h5py.File(dataset_path, 'r') as h5f:

            # iterate over shards in index
            idxs = np.arange(len(h5f.keys()))
            if debug:
                print('\th5 indices:', idxs, file=sys.stderr)

            for i, shard_idx in enumerate(idxs, 1):

                if debug:
                    print('\tshard_idx, h5 len', shard_idx, len(h5f[f'X{shard_idx}']), file=sys.stderr)
                loader = load_shard(h5f, batch_size, shard_idx)
                if debug:
                    print('\t\tloader batches ', len(loader), file=sys.stderr)

                pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(idxs)}')
                for batch in pbar:

                    # getting predictions
                    DNAs = batch[0].to(device)
                    if debug:
                        print('\t\t\tbatch DNA ', len(DNAs), end='', file=sys.stderr) # should be 36 -> referring to 5k blocks
                    DNAs = DNAs.to(torch.float32).to(device)
                    # with torch.no_grad():
                    #     y_pred = model(DNAs)
                    # y_pred = y_pred.detach().cpu()
                    with torch.no_grad():
                        y_pred = torch.mean(torch.stack([models[m](DNAs).detach().cpu() for m in range(len(models))]), axis=0)
                    count += len(y_pred)  # update the count for the current batch

                    if debug:
                        print('\tbatch ', len(y_pred), file=sys.stderr)
                        print('\tcount ', count, 'cumsum of lengths ', sum(LEN[:len_idx+1]), file=sys.stderr)

                    # handle predictions and write to BED
                    accumulated_predictions.append(y_pred)
                    accumulated_length += len(y_pred)

                    while len_idx < len(LEN) and accumulated_length >= LEN[len_idx]:
                        # enough predictions to cover the current gene
                        seq_name = NAME[len_idx]
                        gene_length = LEN[len_idx]

                        # combine accumulated predictions into a tensor for the current gene
                        accumulated_predictions = torch.cat(accumulated_predictions, dim=0)
                        combined_predictions = accumulated_predictions[:gene_length]

                        if debug:
                            print(f'\t[predict_and_write] Writing gene {seq_name} with length {gene_length}', file=sys.stderr)
                            print(f'\tlen(accumulated_predictions) {len(accumulated_predictions)}, accumulated_length, {accumulated_length}', file=sys.stderr)

                        write_batch_to_bed(seq_name, combined_predictions, acceptor_bed, donor_bed, threshold=threshold, debug=debug)

                        # move to the next gene
                        accumulated_predictions = [accumulated_predictions[gene_length:]]
                        accumulated_length -= gene_length
                        len_idx += 1

                    pbar.update(1)

                if debug:
                    utils.log_memory_usage()

                pbar.close()

    else: # read from the PyTorch file

        # load all data and reshape to (N, channels, length)
        X = torch.load(dataset_path, map_location="cpu", weights_only=True).permute(0, 2, 1)
        X = X.to(torch.float32)
        ds = TensorDataset(X)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        pbar = tqdm(loader, leave=False, total=len(loader))
        for batch in pbar:

            # getting prediction (ensemble mean across all loaded models)
            DNAs = batch[0].to(device)
            DNAs = DNAs.to(torch.float32).to(device)
            with torch.no_grad():
                y_pred = torch.mean(torch.stack([models[m](DNAs).detach().cpu() for m in range(len(models))]), axis=0)
            count += len(y_pred)

            # accumulate predictions and flush each completed gene to BED. Batches do not
            # align 1:1 with genes, so map them via LEN exactly like the HDF5 branch above.
            accumulated_predictions.append(y_pred)
            accumulated_length += len(y_pred)

            while len_idx < len(LEN) and accumulated_length >= LEN[len_idx]:
                seq_name = NAME[len_idx]
                gene_length = LEN[len_idx]
                accumulated_predictions = torch.cat(accumulated_predictions, dim=0)
                combined_predictions = accumulated_predictions[:gene_length]
                write_batch_to_bed(seq_name, combined_predictions, acceptor_bed, donor_bed, threshold=threshold, debug=debug)
                accumulated_predictions = [accumulated_predictions[gene_length:]]
                accumulated_length -= gene_length
                len_idx += 1

            pbar.update(1)

        pbar.close()

    acceptor_bed.close()
    donor_bed.close()

    print(f'\t[INFO] {count} predictions processed.')
    print(f"\t[INFO] Acceptor BED file saved to {acceptor_bed_path}")
    print(f"\t[INFO] Donor BED file saved to {donor_bed_path}")


################
##   DRIVER   ##
################

# Command-line prediction
def predict_cli(args):
    '''
    Parameters:
    - args (argparse.args):
        - model: Path to SpliceAI model
        - output_dir: Output directory
        - flanking_size: Flanking sequence size
        - input_sequence: FASTA file
        - gff_file (opt): GFF file
        - threshold (opt): Threshold for prediction
        - debug (opt): Debug mode
        - predict_all (opt): Run steps 4 and 5 vs 4o
        - hdf_threshold (opt): Threshold for HDF5 file
        - flush_threshold (opt): Threshold for flushing predictions
        - split_threshold (opt): Threshold for splitting FASTA file
        - chunk_size (opt): Chunk size for processing

    '''
    print("Running SpliceAI-toolkit with 'predict' mode")

    # get all input args
    output_dir = args.output_dir
    flanking_size = args.flanking_size
    model_path = args.model
    input_sequence = args.input_sequence
    gff_file = args.annotation_file
    gene_flank = getattr(args, 'gene_flank', -1)
    threshold = np.float32(args.threshold)
    debug = args.debug
    predict_all = args.predict_all
    hdf_threshold_len = args.hdf_threshold
    flush_predict_threshold = args.flush_threshold
    split_fasta_threshold = args.split_threshold
    chunk_size = args.chunk_size

    # initialize global variables
    consts = utils.initialize_constants(flanking_size, hdf_threshold_len, flush_predict_threshold, chunk_size, split_fasta_threshold)

    print(f'''Running predict with SL: {consts['SL']}, flanking_size: {flanking_size}, threshold: {threshold}, in {'debug, ' if debug else ''}{'turbo' if not predict_all else 'all'} mode.
          model: {model_path},
          input_sequence: {input_sequence},
          gff_file: {gff_file},
          output_dir: {output_dir},
          hdf_threshold_len: {hdf_threshold_len}, flush_predict_threshold: {flush_predict_threshold}, split_fasta_threshold: {split_fasta_threshold}, chunk_size: {chunk_size}''')

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_base = utils.initialize_paths(output_dir, flanking_size, consts['SL'])
    print("Output path: ", output_base, file=sys.stderr)
    print("Model path: ", model_path, file=sys.stderr)
    print("Flanking sequence size: ", flanking_size, file=sys.stderr)
    print("Sequence length: ", consts['SL'], file=sys.stderr)

    ### PART 1: Extracting input sequence
    print("--- Step 1: Extracting input sequence ... ---", flush=True)
    start_time = time.time()

    # if gff file is provided, extract just the gene regions into a new FASTA file
    if gff_file:
        # default (-1) => include the model's required context (CL_max/2) of real
        # genomic flanking on each side of every gene; minus-strand genes are
        # reverse-complemented automatically inside process_gff
        resolved_gene_flank = (consts['CL_max'] // 2) if gene_flank is None or gene_flank < 0 else gene_flank
        print(f'\t[INFO] GFF file provided: extracting gene sequences (+/- {resolved_gene_flank} bp genomic flank, minus-strand genes auto reverse-complemented).')
        new_fasta = process_gff(input_sequence, gff_file, output_base, gene_flank=resolved_gene_flank)
        datafile_path, NAME, SEQ = get_sequences(new_fasta, output_base, consts['CL_max'], hdf_threshold_len=consts['HDF_THRESHOLD_LEN'], split_fasta_threshold=consts['SPLIT_FASTA_THRESHOLD'], debug=debug)
    else:
        # otherwise, collect all sequences from FASTA into file
        print("\t[INFO] No -a/--annotation given: each FASTA entry is scored on the strand AS PROVIDED. "
              "Genes on the minus strand must be reverse-complemented first (or pass -a so OpenSpliceAI "
              "handles strand and genomic flanking for you).")
        datafile_path, NAME, SEQ = get_sequences(input_sequence, output_base, consts['CL_max'], hdf_threshold_len=consts['HDF_THRESHOLD_LEN'], split_fasta_threshold=consts['SPLIT_FASTA_THRESHOLD'], debug=debug)

    print("--- %s seconds ---" % (time.time() - start_time))

    ### PART 2: Getting one-hot encoding of inputs
    print("--- Step 2: Creating one-hot encoding ... ---", flush=True)
    start_time = time.time()

    dataset_path, LEN = convert_sequences(datafile_path, output_base, consts['SL'], consts['CL_max'], chunk_size=consts['CHUNK_SIZE'], SEQ=SEQ, debug=debug)

    print("--- %s seconds ---" % (time.time() - start_time))

    ### PART 3: Loading model
    print("--- Step 3: Load model ... ---", flush=True)
    start_time = time.time()

    # setup device
    device = setup_device()

    # load model from current state
    models, params = load_pytorch_models(model_path, device, consts['SL'], flanking_size)
    print(f"\t[INFO] Device: {device}, Models: {models}, Params: {params}")

    print("--- %s seconds ---" % (time.time() - start_time))

    if predict_all: # predict using intermediate files

        ### PART 4: Get predictions
        print("--- Step 4: Get predictions ... ---", flush=True)
        start_time = time.time()

        predict_file = get_prediction(models, dataset_path, device, params['BATCH_SIZE'], output_base, flush_predict_threshold=consts['FLUSH_PREDICT_THRESHOLD'], debug=debug)

        print("--- %s seconds ---" % (time.time() - start_time))

        ### PART 5: Generate BED report
        print("--- Step 5: Generating BED report ... ---", flush=True)
        start_time = time.time()

        generate_bed(predict_file, NAME, LEN, output_base, threshold=threshold, debug=debug)

        print("--- %s seconds ---" % (time.time() - start_time))

    else: # combine prediction and output

        ### PART 4o: Get only predictions and write to BED
        print("--- Step 4o: Extract predictions to BED ... ---", flush=True)
        start_time = time.time()

        predict_and_write(models, dataset_path, device, params['BATCH_SIZE'], NAME, LEN, output_base, threshold=threshold, debug=debug)

        print("--- %s seconds ---" % (time.time() - start_time))


# Simplified in-memory prediction
def predict(input_sequence, model_path, flanking_size):
    '''
    Parameters:
    - input_sequence (str): Raw gene sequence
    - model_path (str): Path to SpliceAI model
    - flanking_size (int): Size of flanking sequences

    Outputs:
    - Raw predicted tensors
    '''

    # Initialize global variables to defaults
    consts = utils.initialize_constants(flanking_size)
    sequence_length = len(input_sequence)

    # One-hot encode input
    X = create_datapoints(input_sequence, SL=consts['SL'], CL_max=consts['CL_max'])
    X = torch.tensor(X, dtype=torch.float32)  # Convert to tensor
    X = X.permute(0, 2, 1)

    # Setup device and model
    device = setup_device()
    print(f'\t[INFO] Device: {device}')
    models, params = load_pytorch_models(model_path, device, consts['SL'], flanking_size)

    # Get predictions
    DNAs = X.to(device)
    # with torch.no_grad():
    #     y_pred = model(DNAs)
    # y_pred = y_pred.detach().cpu()
    with torch.no_grad():
        y_pred = torch.mean(torch.stack([models[m](DNAs).detach().cpu() for m in range(len(models))]), axis=0)
    y_pred = y_pred.permute(0, 2, 1).contiguous().view(-1, y_pred.shape[1])
    y_pred = y_pred[:sequence_length, :] # crop out the extra padding

    return y_pred
