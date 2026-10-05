'''
variant.py
This command annotates variants in a VCF file using SpliceAI-toolkit. It reads the input VCF file, annotates 
each variant with delta scores and delta positions, and writes the annotated variants to an output VCF file. 
It uses the Annotator class to annotate variants based on the reference genome and annotation provided. The 
annotated variants are written to the output VCF file with the 'SpliceAI' INFO field containing the delta 
scores and delta positions for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). 
'''

import logging
import pysam
from openspliceai.variant.utils import *
from tqdm import tqdm
import os
import secrets
import signal
import stat
import sys
import math
import gzip
from contextlib import redirect_stdout


_OPEN_SPLICEAI_HEADER = (
    '##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="OpenSpliceAI variant '
    'annotation. These include delta scores (DS) and delta positions (DP) for '
    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">'
)


def _add_reference_contigs(header, ref_fasta):
    """Declare every FASTA sequence in ``header`` if it is not already present.

    htslib can read a text VCF whose records use undeclared contigs, but it cannot
    safely write those records back out.  The reference is the authoritative source
    of contig names and lengths for variant scoring, including alt/random contigs that
    are intentionally absent from a primary-chromosome gene annotation.
    """
    for contig in ref_fasta.keys():
        contig = str(contig)
        if contig not in header.contigs:
            header.contigs.add(contig, length=len(ref_fasta[contig]))


def _annotation_contigs(ann):
    """Return annotation contigs plus the style used by ``normalise_chrom``."""
    contigs = {str(contig) for contig in ann.chroms}
    target = str(next(iter(ann.chroms))) if len(ann.chroms) else None
    return contigs, target


def _is_annotation_contig(chrom, contigs, target):
    """Whether ``chrom`` can have a gene annotation (allowing ``chr`` aliases)."""
    if target is None:
        return False
    return normalise_chrom(str(chrom), target) in contigs


def _validate_output(path, expected_records):
    """Perform inexpensive structural validation before publishing a VCF.

    This deliberately checks the properties most indicative of interrupted scoring:
    a non-empty newline-terminated file, a parseable VCF with the expected INFO
    declaration, well-shaped OpenSpliceAI values, and exactly one output record per
    input record.  Deeper source/output digest checks belong to the campaign audit.
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError("temporary output VCF is empty")

    with open(path, 'rb') as handle:
        compressed = handle.read(2) == b'\x1f\x8b'
    if compressed:
        try:
            with gzip.open(path, 'rb') as handle:
                last = b''
                for block in iter(lambda: handle.read(65536), b''):
                    last = block[-1:]
        except (OSError, EOFError) as exc:
            raise RuntimeError(f'temporary output VCF is not parseable: {exc}') from exc
    else:
        with open(path, 'rb') as handle:
            handle.seek(-1, os.SEEK_END)
            last = handle.read(1)
    if last != b'\n':
        raise RuntimeError('temporary output VCF is not newline-terminated')

    observed_records = 0
    try:
        with pysam.VariantFile(path) as output_vcf:
            if "OpenSpliceAI" not in output_vcf.header.info:
                raise RuntimeError("temporary output VCF lacks the OpenSpliceAI INFO header")
            for record in output_vcf:
                observed_records += 1
                values = record.info.get("OpenSpliceAI")
                if values is None:
                    continue
                if isinstance(values, str):
                    values = (values,)
                for value in values:
                    fields = str(value).split('|')
                    try:
                        if len(fields) != 10 or not fields[0] or not fields[1]:
                            raise ValueError('expected allele, symbol and eight numeric fields')
                        scores = [float(field) for field in fields[2:6]]
                        if not all(math.isfinite(score) and -1 <= score <= 1 for score in scores):
                            raise ValueError('scores must be finite probability differences')
                        for field in fields[6:]:
                            int(field)
                    except ValueError:
                        raise RuntimeError(
                            "temporary output VCF contains a malformed OpenSpliceAI annotation"
                        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"temporary output VCF is not parseable: {exc}") from exc

    if observed_records != expected_records:
        raise RuntimeError(
            "temporary output VCF record count does not match input "
            f"({observed_records} != {expected_records})"
        )


def _output_path(output_vcf):
    """Return an absolute filesystem destination, or ``None`` for stdout/streams."""
    if isinstance(output_vcf, (str, bytes, os.PathLike)):
        path = os.fsdecode(os.fspath(output_vcf))
        if path != "-":
            return os.path.abspath(path)
    return None


def _temporary_output_path(destination):
    """Reserve a hidden temporary file with normal output-file permissions.

    ``tempfile.mkstemp`` always creates mode 0600, which would become the final
    mode after ``os.replace``. Use an exclusive 0666 create so the kernel applies
    the process umask for new outputs; when replacing a destination, retain that
    file's existing mode.
    """
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    destination_mode = None
    try:
        destination_mode = stat.S_IMODE(os.stat(destination).st_mode)
    except FileNotFoundError:
        pass

    for _ in range(100):
        path = os.path.join(
            directory,
            f".{os.path.basename(destination)}.{secrets.token_hex(8)}.tmp",
        )
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        try:
            if destination_mode is not None:
                os.fchmod(fd, destination_mode)
        finally:
            os.close(fd)
        return path
    raise FileExistsError("could not reserve a unique temporary output VCF")


def _install_termination_handlers(enabled):
    """Turn scheduler termination signals into exceptions so ``finally`` can clean up."""
    previous = {}
    if not enabled:
        return previous

    def _terminate(signum, _frame):
        raise InterruptedError(f"variant scoring interrupted by signal {signum}")

    for sig_name in ("SIGTERM", "SIGXCPU"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _terminate)
        except (OSError, ValueError):
            # Signal handlers can only be installed from the main thread.
            previous.pop(sig, None)
    return previous


def _restore_signal_handlers(previous):
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def variant(args):
    """Annotate a VCF with splicing delta scores (entry point for the ``variant`` subcommand).

    Reads the input VCF (``args.input_vcf``, default stdin), builds an
    ``Annotator`` from the reference genome, gene annotation
    (``grch37``/``grch38`` builtins or a custom TSV) and SpliceAI model(s)
    (PyTorch or Keras). For each variant it computes, within ``args.distance``,
    delta scores/positions for acceptor and donor gain/loss and writes them to
    the ``OpenSpliceAI`` INFO field of the output VCF (``args.output_vcf``,
    default stdout). Returns nothing.
    """
    # Capture the output stream before redirecting diagnostics, including loader
    # prints, to stderr. htslib's '-' writes to descriptor 1 directly.
    output_vcf = args.output_vcf
    with redirect_stdout(sys.stderr):
        return _variant(args, output_vcf)


def _variant(args, output_vcf):
    print("Running OpenSpliceAI with 'variant' mode")
    start_time = time.time()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Error handling for required arguments
    if None in [args.input_vcf, args.output_vcf, args.ref_genome, args.annotation, args.model, args.flanking_size]:
        logging.error('Usage: openspliceai [-h] [-m [model]] [-f [flanking_size]] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]]')
        raise ValueError('Input/output VCF, reference, annotation, model and context are required')

    # Define arguments
    ref_genome = args.ref_genome
    annotation = args.annotation
    input_vcf = args.input_vcf
    distance = args.distance
    mask = args.mask
    model = args.model
    flanking_size = args.flanking_size
    model_type = args.model_type
    precision = args.precision
    batch_size = getattr(args, 'batch_size', 1)
    validate_scoring_options(distance, mask, flanking_size, precision, batch_size)

    print(f'''Running with genome: {ref_genome}, annotation: {annotation}, 
          model(s): {model}, model_type: {model_type}, 
          input: {input_vcf}, output: {output_vcf}, 
          distance: {distance}, mask: {mask}, flanking_size: {flanking_size}, precision: {precision}''')

    # Reading input VCF file
    print('\t[INFO] Reading input VCF file')
    try:
        vcf = pysam.VariantFile(input_vcf)
    except (IOError, ValueError) as e:
        logging.error('Error reading input file: {}'.format(e))
        raise ValueError(f'Error reading input file: {e}') from e

    output = None
    temp_output = None
    expected_records = 0
    previous_handlers = {}
    ann = None
    try:
        # Build the annotator before creating any output. A missing/corrupt model must
        # not truncate an already valid destination.
        logging.info('Initializing Annotator class')
        ann = Annotator(ref_genome, annotation, model, model_type, flanking_size)

        # Add the annotation and every reference contig to the in-memory input header.
        # Records produced by pysam retain this header, so they can also receive the new
        # INFO field without a per-record header translation.
        header = vcf.header
        if 'OpenSpliceAI' not in header.info:
            header.add_line(_OPEN_SPLICEAI_HEADER)
        _add_reference_contigs(header, ann.ref_fasta)
        ann_contigs, ann_contig_style = _annotation_contigs(ann)

        # Filesystem outputs are always written beside the destination and atomically
        # published. Keep stdout/file-like behavior for public CLI compatibility.
        destination = _output_path(output_vcf)
        if destination is not None:
            temp_output = _temporary_output_path(destination)
            output_target = temp_output
        else:
            output_target = '-' if output_vcf == '-' else output_vcf
        previous_handlers = _install_termination_handlers(temp_output is not None)

        print('\t[INFO] Generating output VCF file')
        try:
            mode = 'wz' if destination and destination.endswith(('.gz', '.bgz')) else 'w'
            output = pysam.VariantFile(output_target, mode=mode, header=header)
        except (IOError, ValueError) as e:
            logging.error('Error generating output VCF file: {}'.format(e))
            raise

        # Obtain delta scores for each variant in VCF. Unsupported annotation
        # contigs are intentionally emitted unchanged as coverage-only records.
        if batch_size > 1 and model_type == 'pytorch':
            # Batched inference: buffer records, score their windows in one batched
            # forward pass, then write in input order. Much faster on many-variant VCFs.
            FLUSH = max(batch_size, 1024)   # records buffered per flush
            buf = []

            def _flush(records_buf):
                supported = [
                    (idx, rec) for idx, rec in enumerate(records_buf)
                    if _is_annotation_contig(rec.chrom, ann_contigs, ann_contig_style)
                ]
                scores_by_idx = {}
                if supported:
                    results = get_delta_scores_batched(
                        [rec for _, rec in supported], ann, distance, mask,
                        flanking_size, precision, batch_size
                    )
                    if len(results) != len(supported):
                        raise RuntimeError('Batched scoring did not return one result per record')
                    scores_by_idx = {
                        idx: scores for (idx, _), scores in zip(supported, results)
                    }
                for idx, rec in enumerate(records_buf):
                    scores = scores_by_idx.get(idx)
                    if scores:
                        rec.info['OpenSpliceAI'] = scores
                    output.write(rec)

            for record in tqdm(vcf):
                expected_records += 1
                buf.append(record)
                if len(buf) >= FLUSH:
                    _flush(buf)
                    buf = []
            if buf:
                _flush(buf)
        else:
            for record in tqdm(vcf):
                expected_records += 1
                scores = None
                if _is_annotation_contig(record.chrom, ann_contigs, ann_contig_style):
                    scores = get_delta_scores(
                        record, ann, distance, mask, flanking_size, precision
                    )
                if scores:
                    record.info['OpenSpliceAI'] = scores
                output.write(record)

        output.close()
        output = None

        if temp_output is not None:
            _validate_output(temp_output, expected_records)
            os.replace(temp_output, destination)
            temp_output = None

        logging.info('Annotation completed and written to output VCF file')
    finally:
        if output is not None:
            try:
                output.close()
            except Exception:
                pass
        vcf.close()
        if ann is not None and hasattr(ann, 'close'):
            ann.close()
        if temp_output is not None:
            try:
                os.unlink(temp_output)
            except FileNotFoundError:
                pass
        _restore_signal_handlers(previous_handlers)
    
    print("--- %s seconds ---" % (time.time() - start_time))
