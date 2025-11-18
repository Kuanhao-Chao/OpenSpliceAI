from importlib.resources import files
import logging
import os, glob
import platform

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta

from openspliceai.constants import *
from openspliceai.predict.predict import *
from openspliceai.predict.utils import *
from openspliceai.rbp.metadata import extract_film_config_from_state_dict
from openspliceai.train_base.openspliceai import SpliceAI
    
##############################################
## LOADING PYTORCH AND KERAS MODELS
##############################################

def setup_device():
        """Select computation device based on availability."""
        device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
        return torch.device(device_str)

def load_pytorch_models(model_path, CL):
    """
    Loads a SpliceAI PyTorch model from given state, inferring device.
    
    Params:
    - model_path (str): Path to the model state dict, or a directory of models
    - CL (int): Context length parameter for model conversion.
    
    Returns:
    - loaded_models (list): SpliceAI model(s) loaded with given state.
    """
    
    def load_model(device, flanking_size, film_config=None):
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

        CL = 2 * np.sum(AR*(W-1))

        print(f"\t[INFO] Context nucleotides {CL}")
        print(f"\t[INFO] Sequence length (output): {SL}")
        
        model = SpliceAI(L, W, AR, film_config=film_config).to(device)
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
    
    for state_dict in models:
        try: 
            film_config = extract_film_config_from_state_dict(state_dict)
            model, params = load_model(device, CL, film_config or None)
            model.load_state_dict(state_dict)      # loads state dict
            model = model.to(device)               # puts model on device
            model.eval()                           # puts model in evaluation mode
            loaded_models.append(model)            # appends model to list of loaded models  
        except Exception as e:
            logging.error(f"Error processing model for device: {e}. Skipping...")
            
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
            logging.error(f"No valid PyTorch models found in directory: {model_path}")
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

    # Replace nucleotides with corresponding indices
    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    # Convert the sequence to one-hot encoded numpy array
    return map[np.fromstring(seq, np.int8) % 5]


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
    
    def __init__(self, ref_fasta, annotations, model_path='SpliceAI', model_type='keras', CL=80, rbp_tensor=None):
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
            annotations = './data/vcf/grch37.txt'
        elif annotations == 'grch38':
            annotations = './data/vcf/grch38.txt'

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
        self.rbp_tensor = None
        if model_path == 'SpliceAI':
            from tensorflow import keras
            paths = ('./models/spliceai/spliceai{}.h5'.format(x) for x in range(1, 6))  # Generate paths for SpliceAI models
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
        self._maybe_prepare_rbp_condition(rbp_tensor)

    def _maybe_prepare_rbp_condition(self, rbp_tensor):
        if getattr(self, "keras", False):
            if rbp_tensor is not None:
                logging.warning("RBP expression provided but current backend does not use FiLM; ignoring.")
            self.rbp_tensor = None
            return
        film_dims = set()
        for model in self.models:
            metadata = {}
            if hasattr(model, "rbp_metadata"):
                metadata = model.rbp_metadata()
            dim = metadata.get("rbp_dim")
            if dim:
                film_dims.add(dim)
        if not film_dims:
            if rbp_tensor is not None:
                logging.warning("RBP expression provided but loaded model lacks FiLM layers; ignoring vector.")
            self.rbp_tensor = None
            return
        if rbp_tensor is None:
            raise ValueError("Model contains FiLM layers; please provide --rbp-expression.")
        if len(film_dims) > 1:
            raise ValueError(f"Inconsistent FiLM dimensions detected: {sorted(film_dims)}")
        required_dim = film_dims.pop()
        tensor = rbp_tensor
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != required_dim:
            raise ValueError(f"RBP vector dimension ({tensor.shape[-1]}) does not match model requirement ({required_dim}).")
        device = next(self.models[0].parameters()).device
        self.rbp_tensor = tensor.to(device)

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
                    cond = ann.rbp_tensor
                    y_ref_preds = []
                    y_alt_preds = []
                    for m in range(len(ann.models)):
                        model = ann.models[m]
                        if cond is None:
                            y_ref_preds.append(model(x_ref).detach().cpu())
                            y_alt_preds.append(model(x_alt).detach().cpu())
                        else:
                            y_ref_preds.append(model(x_ref, cond).detach().cpu())
                            y_alt_preds.append(model(x_alt, cond).detach().cpu())
                    y_ref = torch.mean(torch.stack(y_ref_preds), axis=0)
                    y_alt = torch.mean(torch.stack(y_alt_preds), axis=0)
                
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
