from importlib.resources import files
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import logging
import platform
import os
import re
import glob
from openspliceai.train_base.openspliceai import SpliceAI
from openspliceai.constants import *
from openspliceai.predict.predict import *
from openspliceai.predict.utils import *
    
##############################################
## LOADING PYTORCH AND KERAS MODELS
##############################################

def setup_device():
        """Select computation device based on availability."""
        device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
        return torch.device(device_str)


def _resolve_builtin_annotation(name):
    """
    Resolve a built-in annotation shortcut ('grch37'/'grch38') to its packaged file path.
    The annotation tables ship inside the package (openspliceai/variant/annotations/), so this
    works regardless of the current working directory or how the package was installed.
    """
    return str(files('openspliceai.variant').joinpath('annotations', f'{name}.txt'))


def _resolve_default_spliceai_models():
    """
    Resolve the paths to the bundled original SpliceAI Keras models (spliceai1-5.h5),
    relative to the repository checkout. Returns a list of 5 paths.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(repo_root, 'models', 'spliceai', 'SpliceAI_models_release')
    return [os.path.join(model_dir, f'spliceai{x}.h5') for x in range(1, 6)]

def load_pytorch_models(model_path, CL):
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
        L = 32
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        N_GPUS = 2
        BATCH_SIZE = 18*N_GPUS

        if int(flanking_size) == 80:
            W = np.asarray([11, 11, 11, 11])
            AR = np.asarray([1, 1, 1, 1])
            BATCH_SIZE = 18*N_GPUS
        elif int(flanking_size) == 400:
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
            BATCH_SIZE = 18*N_GPUS
        elif int(flanking_size) == 2000:
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                            21, 21, 21, 21])
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                            10, 10, 10, 10])
            BATCH_SIZE = 12*N_GPUS
        elif int(flanking_size) == 10000:
            W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                            21, 21, 21, 21, 41, 41, 41, 41])
            AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                            10, 10, 10, 10, 25, 25, 25, 25])
            BATCH_SIZE = 6*N_GPUS
        else:
            raise ValueError(
                f"Unsupported flanking_size {flanking_size}; expected one of 80, 400, 2000, 10000."
            )

        CL = 2 * np.sum(AR*(W-1))

        print(f"\t[INFO] Context nucleotides {CL}")
        print(f"\t[INFO] Sequence length (output): {SL}")
        
        model = SpliceAI(L, W, AR).to(device)
        params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}

        return model, params
    
    # Setup device
    device = setup_device()
    
    # Load all model state dicts given the supplied model path
    if os.path.isdir(model_path):
        model_files = glob.glob(os.path.join(model_path, '*.p[th]')) # gets all PyTorch models from supplied directory
        if not model_files:
            logging.error(f"No PyTorch model files found in directory: {model_path}")
            exit()
            
        models = []
        for model_file in model_files:
            try:
                model = torch.load(model_file, map_location=device)
                models.append(model)
            except Exception as e:
                logging.error(f"Error loading PyTorch model from file {model_file}: {e}. Skipping...")
                
        if not models:
            logging.error(f"No valid PyTorch models found in directory: {model_path}")
            exit()
    
    elif os.path.isfile(model_path):
        try:
            models = [torch.load(model_path, map_location=device)]
        except Exception as e:
            logging.error(f"Error loading PyTorch model from file {model_path}: {e}.")
            exit()
        
    else:
        logging.error(f"Invalid path: {model_path}")
        exit()
    
    # Load state of model to device
    # NOTE: supplied model paths should be state dicts, not model files  
    loaded_models = []
    
    mismatch_hint = (
        "[HINT] Ensure the checkpoint was trained with the same flanking size "
        f"({CL}) that you supplied. For packaged weights, choose the model from "
        f"the `{CL}nt` directory."
    )

    for state_dict in models:
        model, params = load_model(device, CL)  # loads new SpliceAI model with correct hyperparams
        try:
            model.load_state_dict(state_dict)   # loads state dict
        except RuntimeError as e:
            err_msg = str(e)
            if "size mismatch" in err_msg or "shape" in err_msg:
                logging.warning("Skipping model due to incompatible tensor shapes.")
                logging.warning("This typically indicates a flanking-size mismatch between the model and CLI arguments.")
                logging.warning(mismatch_hint)
                continue
            logging.error(f"Error processing model for device: {err_msg}. Skipping...")
            continue

        model = model.to(device)                # puts model on device
        model.eval()                            # puts model in evaluation mode
        loaded_models.append(model)             # appends model to list of loaded models  
            
    if not loaded_models:
        logging.error("No models were successfully loaded to the device.")
        exit()
        
    return loaded_models

def load_keras_models(model_path):
    """
    Loads Keras models from given path.
    
    Params:
    - model_path (str): Path to the model file or directory of models.
    
    Returns:
    - models (list): List of loaded Keras models.
    """
    from tensorflow import keras
    
    if os.path.isdir(model_path): # directory supplied
        model_files = glob.glob(os.path.join(model_path, '*.h5')) # get all Keras models from a directory
        if not model_files:
            logging.error(f"No Keras model files found in directory: {model_path}")
            exit()
            
        models = []
        for model_file in model_files:
            try:
                model = keras.models.load_model(model_file)
                models.append(model)
            except Exception as e:
                logging.error(f"Error loading Keras model from file {model_file}: {e}. Skipping...")

        if not models:
            logging.error(f"No valid Keras models found in directory: {model_path}")
            exit()
            
        return models
    
    elif os.path.isfile(model_path): # file supplied
        try:
            return [keras.models.load_model(model_path)]
        except Exception as e:
            logging.error(f"Error loading Keras model from file {model_path}: {e}")
            exit()
        
    else: # invalid path
        logging.error(f"Invalid path: {model_path}")
        exit()

##############################################
## FORMATTING INPUT DATA FOR PREDICTION
##############################################

def one_hot_encode(seq):
    """
    One-hot encode a DNA sequence.
    
    Args:
        seq (str): DNA sequence to be encoded.
    
    Returns:
        np.ndarray: One-hot encoded representation of the sequence.
    """

    # Define a mapping matrix for nucleotide to one-hot encoding
    map = np.asarray([[0, 0, 0, 0],  # N or any invalid character
                      [1, 0, 0, 0],  # A
                      [0, 1, 0, 0],  # C
                      [0, 0, 1, 0],  # G
                      [0, 0, 0, 1]]) # T

    # Per the map above, index 0 is "N or any invalid character". Fold every non-ACGT
    # base (N, IUPAC ambiguity codes like R/Y/S, gaps, etc.) to N *before* the latin-1
    # ``% 5`` trick -- otherwise an unexpected byte would alias onto a concrete base
    # instead of the intended all-zero row. ACGTN inputs are unaffected (bit-identical),
    # so real-genome variant scores are unchanged.
    seq = re.sub('[^ACGT]', 'N', seq.upper())
    seq = seq.replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    # Convert the sequence to one-hot encoded numpy array.
    # np.fromstring(binary) was removed in NumPy 2.0; frombuffer works across NumPy versions.
    return map[np.frombuffer(seq.encode('latin-1'), dtype=np.int8) % 5]


####################################################################################################################################
#######                                                                                                                      #######
#######                                             ANNOTATOR CLASS                                                          #######
#######                                                                                                                      #######
####################################################################################################################################

class Annotator:
    """
    Annotator class to handle gene annotations and reference sequences.
    It initializes with the reference genome, annotation data, and optional model configuration.
    """
    
    def __init__(self, ref_fasta, annotations, model_path='SpliceAI', model_type='keras', CL=80):
        """
        Initializes the Annotator with reference genome, annotations, and model settings.
        
        Args:
            ref_fasta (str): Path to the reference genome FASTA file.
            annotations (str): Path or name of the annotation file (e.g., 'grch37', 'grch38').
            model_path (str, optional): Path to the model file or type of model ('SpliceAI'). Defaults to SpliceAI.
            model_type (str, optional): Type of model ('keras' or 'pytorch'). Defaults to 'keras'.
            CL (int, optional): Context length parameter for model conversion. Defaults to 80.
        """

        # Load annotation file based on provided annotations type
        if annotations == 'grch37':
            annotations = _resolve_builtin_annotation('grch37')
        elif annotations == 'grch38':
            annotations = _resolve_builtin_annotation('grch38')

        # Load and parse the annotation file
        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            # Extract relevant columns into numpy arrays for efficient access
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy() + 1  # Transcription start sites (1-based indexing)
            self.tx_ends = df['TX_END'].to_numpy()  # Transcription end sites
            
            # Extract and process exon start and end sites, convert into numpy array format
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i]) + 1
                                for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                              for c in df['EXON_END'].to_numpy()]
        except IOError as e:
            logging.error('{}'.format(e)) 
            exit()  # Exit if the file cannot be read
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()  # Exit if the file format is incorrect

        # Load the reference genome fasta file
        try:
            self.ref_fasta = Fasta(ref_fasta, sequence_always_upper=True, rebuild=False)
        except IOError as e:
            logging.error('{}'.format(e))  # Log file read error
            exit()  # Exit if the file cannot be read

        # Load models based on the specified model type or file
        if model_path == 'SpliceAI':
            from tensorflow import keras
            paths = _resolve_default_spliceai_models()  # bundled original SpliceAI Keras models
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                logging.error('Default SpliceAI Keras models not found: {}. '
                              'Pass an explicit --model path/directory instead.'.format(missing))
                exit()
            self.models = [keras.models.load_model(x) for x in paths]
            self.keras = True
        elif model_type == 'keras': # load models using keras
            self.models = load_keras_models(model_path)
            self.keras = True
        elif model_type == 'pytorch': # load models using pytorch 
            self.models = load_pytorch_models(model_path, CL)
            self.keras = False
        else:
            logging.error('Model type {} not supported'.format(model_type))
            exit()
        
        print(f'\t[INFO] {len(self.models)} model(s) loaded successfully')

    def get_name_and_strand(self, chrom, pos):
        """
        Retrieve gene names and strands overlapping a given chromosome position.
        
        Args:
            chrom (str): Chromosome identifier.
            pos (int): Position on the chromosome.
        
        Returns:
            tuple: Lists of gene names, strands, and their indices overlapping the given position.
        """

        # Normalize chromosome identifier to match the annotation format
        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        # Find indices of annotations overlapping the given chromosome and position
        idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                              np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                                             np.nonzero(pos <= self.tx_ends)[0]))

        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs  # Return matching gene names and strands
        else:
            return [], [], []  # Return empty lists if no matches are found

    def get_pos_data(self, idx, pos):
        """
        Calculate distances from a given position to the transcription start site, 
        transcription end site, and nearest exon boundary for a specific gene.
        
        Args:
            idx (int): Index of the gene in the annotations.
            pos (int): Position on the chromosome.
        
        Returns:
            tuple: Distances to transcription start, transcription end, and nearest exon boundary.
        """

        # Calculate distances to transcription start and end sites
        dist_tx_start = self.tx_starts[idx] - pos
        dist_tx_end = self.tx_ends[idx] - pos
        # Calculate the closest distance to an exon boundary
        dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx]) - pos, key=abs)
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)  # Package distances into a tuple

        return dist_ann
    
##############################################
## CALCULATING DELTA SCORES
##############################################

def normalise_chrom(source, target):
    """
    Normalize chromosome identifiers to ensure consistency in format (with or without 'chr' prefix).
    
    Args:
        source (str): Source chromosome identifier.
        target (str): Target chromosome identifier for comparison.
    
    Returns:
        str: Normalized chromosome identifier.
    """

    def has_prefix(x):
        return x.startswith('chr')  # Check if a chromosome name has 'chr' prefix

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')  # Remove 'chr' prefix if target doesn't have it
    elif not has_prefix(source) and has_prefix(target):
        return 'chr' + source  # Add 'chr' prefix if target has it

    return source  # Return source as is if both or neither have 'chr' prefix

def get_delta_scores(record, ann, dist_var, mask, flanking_size=10000, precision=2):
    """
    Calculate delta scores for variant impacts on splice sites.
    
    Args:
        record (pysam Record): Record containing variant information (e.g., chrom, pos, ref, alts).
        ann (Annotator): Annotator instance with annotation and reference genome data.
        dist_var (int): Max distance between variant and gained/lost splice site, defaults to 50.
        mask (bool): Mask scores representing annotated acceptor/donor gain and unannotated acceptor/donor loss, defaults to 0.
        flanking_size (int, optional): Size of the flanking region around the variant, defaults to 10000.
    
    Returns:
        list: Delta scores indicating the impact of variants on splicing.
    """

    # Define coverage and window size around the variant
    cov = 2 * dist_var + 1
    wid = flanking_size + cov
    delta_scores = []
    device = setup_device()

    # Validate the record fields
    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): {}'.format(record))
        return delta_scores

    # Get gene names and strands overlapping the variant position
    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        return delta_scores  # Return empty list if no overlapping genes are found

    # Normalize chromosome and retrieve reference sequence around the variant
    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][record.pos - wid // 2 - 1 : record.pos + wid // 2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return delta_scores

    # Check if the reference sequence matches the expected reference allele
    if seq[wid // 2 : wid // 2 + len(record.ref)].upper() != record.ref:
        logging.warning('Skipping record (ref issue): {}'.format(record))
        return delta_scores

    # Check if the sequence length matches the expected window size
    if len(seq) != wid:
        logging.warning('Skipping record (near chromosome end): {}'.format(record))
        return delta_scores

    # Skip records with a reference allele longer than the distance variable
    if len(record.ref) > 2 * dist_var:
        logging.warning('Skipping record (ref too long): {}'.format(record))
        return delta_scores

    # Iterate over each alternative allele and each gene index to calculate delta score
    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            # Skip specific alternative allele types
            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue
            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            # Handle multi-nucleotide variants
            if len(record.ref) > 1 and len(record.alts[j]) > 1:
                delta_scores.append("{}|{}|.|.|.|.|.|.|.|.".format(record.alts[j], genes[i]))
                continue

            # Calculate position-related distances
            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid // 2 + dist_ann[0], 0), max(wid // 2 - dist_ann[1], 0)]
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len - alt_len, 0)

            # Construct reference and alternative sequences with padding
            x_ref = 'N' * pad_size[0] + seq[pad_size[0]: wid - pad_size[1]] + 'N' * pad_size[1]
            x_alt = x_ref[: wid // 2] + str(record.alts[j]) + x_ref[wid // 2 + ref_len:]

            # One-hot encode the sequences
            x_ref = one_hot_encode(x_ref)[None, :]
            x_alt = one_hot_encode(x_alt)[None, :]

            '''separate handling of PyTorch and Keras models'''
            if ann.keras: # keras model handling
                # Reverse the sequences if on the negative strand
                if strands[i] == '-':
                    x_ref = x_ref[:, ::-1, ::-1]
                    x_alt = x_alt[:, ::-1, ::-1]

                # Predict scores using the models
                y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(len(ann.models))], axis=0)
                y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(len(ann.models))], axis=0)
                
                # Reverse the predicted scores if on the negative strand
                if strands[i] == '-':
                    y_ref = y_ref[:, ::-1]
                    y_alt = y_alt[:, ::-1]
                    
            else: # pytorch model handling
                
                # Reshape tensor to match the model input shape
                x_ref = x_ref.transpose(0, 2, 1)
                x_alt = x_alt.transpose(0, 2, 1)
                
                # Convert to PyTorch tensors
                x_ref = torch.tensor(x_ref, dtype=torch.float32)
                x_alt = torch.tensor(x_alt, dtype=torch.float32)

                # Reverse the sequences if on the negative strand
                if strands[i] == '-':
                    x_ref = torch.flip(x_ref, dims=[1, 2])
                    x_alt = torch.flip(x_alt, dims=[1, 2])

                # Put tensors on device
                x_ref = x_ref.to(device)
                x_alt = x_alt.to(device)
                
                # Predict scores using the models
                with torch.no_grad():
                    y_ref = torch.mean(torch.stack([ann.models[m](x_ref).detach().cpu() for m in range(len(ann.models))]), axis=0)
                    y_alt = torch.mean(torch.stack([ann.models[m](x_alt).detach().cpu() for m in range(len(ann.models))]), axis=0)
                
                # Remove flanking sequence and permute shape
                y_ref = y_ref.permute(0, 2, 1)
                y_alt = y_alt.permute(0, 2, 1)

                # Reverse the predicted scores if on the negative strand and convert back to numpy arrays
                if strands[i] == '-':
                    y_ref = torch.flip(y_ref, dims=[1])
                    y_alt = torch.flip(y_alt, dims=[1])
                
                # Convert to numpy arrays
                y_ref = y_ref.numpy()
                y_alt = y_alt.numpy()
            '''end'''

            # Manually crop the output if it exceeds the expected coverage window
            # This handles cases where the model output is not automatically cropped to the target window
            if y_ref.shape[1] > cov:
                start_idx = wid // 2 - cov // 2
                y_ref = y_ref[:, start_idx : start_idx + cov, :]
                y_alt = y_alt[:, start_idx : start_idx + cov + alt_len - ref_len, :]


            # Adjust the alternative sequence scores based on reference and alternative lengths
            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate([
                    y_alt[:, : cov // 2 + alt_len],
                    np.zeros((1, del_len, 3)),
                    y_alt[:, cov // 2 + alt_len:]
                ], axis=1)
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate([
                    y_alt[:, : cov // 2],
                    np.max(y_alt[:, cov // 2 : cov // 2 + alt_len], axis=1)[:, None, :],
                    y_alt[:, cov // 2 + alt_len:]
                ], axis=1)

            # Concatenate the reference and alternative scores
            y = np.concatenate([y_ref, y_alt])

            # Find the indices of maximum delta scores for splicing acceptor and donor sites
            idx_pa = (y[1, :, 1] - y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1] - y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2] - y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2] - y[1, :, 2]).argmax()

            # print(f"idx_pa: {idx_pa}, idx_na: {idx_na}, idx_pd: {idx_pd}, idx_nd: {idx_nd}")
            # print("cov:", cov)

            # Apply masks to delta scores based on calculated indices and provided mask
            mask_pa = np.logical_and((idx_pa - cov // 2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na - cov // 2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd - cov // 2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd - cov // 2 != dist_ann[2]), mask)

            # Create a format string with the desired precision
            format_str = "{{}}|{{}}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{{}}|{{}}|{{}}|{{}}".format(
                precision, precision, precision, precision)
            
            # Write delta scores for given alternative allele, gene, and calculated indices
            delta_scores.append(format_str.format(
                record.alts[j],
                genes[i],
                (y[1, idx_pa, 1] - y[0, idx_pa, 1]) * (1 - mask_pa),
                (y[0, idx_na, 1] - y[1, idx_na, 1]) * (1 - mask_na),
                (y[1, idx_pd, 2] - y[0, idx_pd, 2]) * (1 - mask_pd),
                (y[0, idx_nd, 2] - y[1, idx_nd, 2]) * (1 - mask_nd),
                idx_pa - cov // 2,
                idx_na - cov // 2,
                idx_pd - cov // 2,
                idx_nd - cov // 2
            ))

    return delta_scores


def _ensemble_forward(models, xb):
    """Run the ensemble mean forward pass on a batch ``xb`` (B, 4, wid) already on device.

    Returns a CPU float tensor of shape (B, Lout, 3) — the per-position class
    probabilities, permuted to match get_delta_scores' post-processing.
    """
    y = torch.mean(torch.stack([models[m](xb) for m in range(len(models))]), axis=0)
    return y.permute(0, 2, 1).detach().to('cpu')


def get_delta_scores_batched(records, ann, dist_var, mask, flanking_size=10000, precision=2, batch_size=64):
    """Batched re-implementation of :func:`get_delta_scores` over a LIST of records (PyTorch only).

    The per-variant math is identical to :func:`get_delta_scores`; the ONLY
    difference is that the model forward pass is batched across many variants
    (and the ref+alt windows) instead of one window at a time. This buffers the
    one-hot windows for every (variant, alt, gene) item, runs the model in
    sub-batches of ``batch_size``, then applies the exact same crop / indel /
    argmax / mask / format steps per item.

    Returns a list (same length and order as ``records``) of delta-score string
    lists — ready to assign to each record's ``OpenSpliceAI`` INFO field.
    """
    cov = 2 * dist_var + 1
    wid = flanking_size + cov
    device = setup_device()
    # Precision/speed knobs (env-overridable). cuDNN autotune helps because all
    # windows share one length. TF32 is the A100 fast path (~2x); it adds ~1e-3
    # batched-vs-single noise on strong signals, far below the score resolution
    # used downstream. Default: TF32 ON for speed. Set OSAI_TF32=0 for full fp32
    # (reproducible across hardware, batched==single) at lower throughput.
    _bench = os.environ.get('OSAI_CUDNN_BENCH', '1') == '1'
    _tf32 = os.environ.get('OSAI_TF32', '1') == '1'
    torch.backends.cudnn.benchmark = _bench
    torch.backends.cudnn.allow_tf32 = _tf32
    torch.backends.cuda.matmul.allow_tf32 = _tf32

    out_per_record = [[] for _ in records]   # ordered entries per record (strings or None placeholders)
    items = []                               # model work items (one per variant,alt,gene)
    ref_index = {}                           # (chrom,pos,gene_idx) -> index into ref_list (dedup)
    ref_list = []                            # UNIQUE reference windows (the 3 alts at a position share one)
    alt_list = []                            # alternate windows (one per item)

    for r, record in enumerate(records):
        entries = out_per_record[r]
        # --- record-level validation (mirrors get_delta_scores) ---
        try:
            record.chrom, record.pos, record.ref, len(record.alts)
        except TypeError:
            logging.warning('Skipping record (bad input): {}'.format(record))
            continue
        (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
        if len(idxs) == 0:
            continue
        chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
        try:
            seq = ann.ref_fasta[chrom][record.pos - wid // 2 - 1 : record.pos + wid // 2].seq
        except (IndexError, ValueError):
            logging.warning('Skipping record (fasta issue): {}'.format(record))
            continue
        if seq[wid // 2 : wid // 2 + len(record.ref)].upper() != record.ref:
            logging.warning('Skipping record (ref issue): {}'.format(record))
            continue
        if len(seq) != wid:
            logging.warning('Skipping record (near chromosome end): {}'.format(record))
            continue
        if len(record.ref) > 2 * dist_var:
            logging.warning('Skipping record (ref too long): {}'.format(record))
            continue

        for j in range(len(record.alts)):
            for i in range(len(idxs)):
                if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                    continue
                if '<' in record.alts[j] or '>' in record.alts[j]:
                    continue
                if len(record.ref) > 1 and len(record.alts[j]) > 1:
                    entries.append("{}|{}|.|.|.|.|.|.|.|.".format(record.alts[j], genes[i]))
                    continue

                dist_ann = ann.get_pos_data(idxs[i], record.pos)
                pad_size = [max(wid // 2 + dist_ann[0], 0), max(wid // 2 - dist_ann[1], 0)]
                ref_len = len(record.ref)
                alt_len = len(record.alts[j])
                del_len = max(ref_len - alt_len, 0)

                x_ref = 'N' * pad_size[0] + seq[pad_size[0]: wid - pad_size[1]] + 'N' * pad_size[1]
                x_alt = x_ref[: wid // 2] + str(record.alts[j]) + x_ref[wid // 2 + ref_len:]
                x_ref = one_hot_encode(x_ref)[None, :].transpose(0, 2, 1)   # (1, 4, wid)
                x_alt = one_hot_encode(x_alt)[None, :].transpose(0, 2, 1)
                x_ref = torch.tensor(x_ref, dtype=torch.float32)
                x_alt = torch.tensor(x_alt, dtype=torch.float32)
                if strands[i] == '-':                                       # reverse-complement input
                    x_ref = torch.flip(x_ref, dims=[1, 2])
                    x_alt = torch.flip(x_alt, dims=[1, 2])

                # Reference-window reuse: x_ref depends only on (chrom, pos, gene),
                # NOT on the alt allele -> the 3 alts at a position share one ref pass.
                rk = (record.chrom, record.pos, int(idxs[i]))
                if rk not in ref_index:
                    ref_index[rk] = len(ref_list)
                    ref_list.append(x_ref[0])                               # (4, wid)
                slot = len(entries)
                entries.append(None)                                        # placeholder, filled after inference
                items.append({
                    'r': r, 'slot': slot, 'ref_idx': ref_index[rk], 'alt_idx': len(alt_list),
                    'strand': strands[i], 'gene': genes[i], 'alt': record.alts[j],
                    'ref_len': ref_len, 'alt_len': alt_len, 'del_len': del_len, 'dist_ann': dist_ann,
                })
                alt_list.append(x_alt[0])                                   # (4, wid)

    # --- batched inference over all collected windows ---
    if items:
        # Unique reference windows all share one width (wid); the 3 alts at a
        # position reuse one ref pass (deduped above). Alt windows, however, can
        # be SHORTER (deletions) or LONGER (insertions) than wid, so they cannot
        # all be stacked into a single fixed-width batch tensor -- bucket them by
        # width and batch within each width group (SNVs, the common case, all
        # share wid and so still batch together).
        def _run_batched(tensors):
            outs = []
            with torch.inference_mode():
                for s in range(0, len(tensors), batch_size):
                    xb = torch.stack(tensors[s:s + batch_size]).to(device)
                    outs.append(_ensemble_forward(ann.models, xb).numpy())
            return np.concatenate(outs, axis=0) if outs else None

        Yref = _run_batched(ref_list)            # (nref, Lout_ref, 3), all width wid

        Yalt = [None] * len(alt_list)            # per-alt output (variable length for indels)
        width_buckets = {}
        for a, x in enumerate(alt_list):
            width_buckets.setdefault(x.shape[-1], []).append(a)
        for group in width_buckets.values():
            yb = _run_batched([alt_list[a] for a in group])   # (len(group), Lout_w, 3)
            for k, a in enumerate(group):
                Yalt[a] = yb[k]

        format_str = "{{}}|{{}}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{{:.{}f}}|{{}}|{{}}|{{}}|{{}}".format(
            precision, precision, precision, precision)

        for it in items:
            ref_len = it['ref_len']
            alt_len = it['alt_len']
            del_len = it['del_len']
            dist_ann = it['dist_ann']
            y_ref = Yref[it['ref_idx']]        # (Lout, 3); ref shared across the position's alts
            y_alt = Yalt[it['alt_idx']]
            if it['strand'] == '-':            # reverse predictions back to genomic order
                y_ref = y_ref[::-1]
                y_alt = y_alt[::-1]
            y_ref = y_ref[None, :]             # (1, Lout, 3)
            y_alt = y_alt[None, :]

            if y_ref.shape[1] > cov:
                start_idx = wid // 2 - cov // 2
                y_ref = y_ref[:, start_idx: start_idx + cov, :]
                y_alt = y_alt[:, start_idx: start_idx + cov + alt_len - ref_len, :]

            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate([
                    y_alt[:, : cov // 2 + alt_len], np.zeros((1, del_len, 3)), y_alt[:, cov // 2 + alt_len:]], axis=1)
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate([
                    y_alt[:, : cov // 2], np.max(y_alt[:, cov // 2: cov // 2 + alt_len], axis=1)[:, None, :],
                    y_alt[:, cov // 2 + alt_len:]], axis=1)

            y = np.concatenate([y_ref, y_alt])
            idx_pa = (y[1, :, 1] - y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1] - y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2] - y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2] - y[1, :, 2]).argmax()
            mask_pa = np.logical_and((idx_pa - cov // 2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na - cov // 2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd - cov // 2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd - cov // 2 != dist_ann[2]), mask)
            out_per_record[it['r']][it['slot']] = format_str.format(
                it['alt'], it['gene'],
                (y[1, idx_pa, 1] - y[0, idx_pa, 1]) * (1 - mask_pa),
                (y[0, idx_na, 1] - y[1, idx_na, 1]) * (1 - mask_na),
                (y[1, idx_pd, 2] - y[0, idx_pd, 2]) * (1 - mask_pd),
                (y[0, idx_nd, 2] - y[1, idx_nd, 2]) * (1 - mask_nd),
                idx_pa - cov // 2, idx_na - cov // 2, idx_pd - cov // 2, idx_nd - cov // 2)

    # drop any unfilled placeholders (defensive; should not occur)
    return [[e for e in entries if e is not None] for entries in out_per_record]