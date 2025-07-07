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
import numpy as np
from openspliceai.variant.utils import *
from tqdm import tqdm
import os

# NOTE: if running with gpu, note that cudnn version should be 8.9.6 or higher, numpy <2.0.0

def variant(args):
    print("Running SpliceAI-toolkit with 'variant' mode")
    start_time = time.time()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Error handling for required arguments
    if None in [args.input_vcf, args.output_vcf, args.ref_genome, args.annotation, args.model, args.flanking_size]:
        logging.error('Usage: openspliceai [-h] [-m [model]] [-f [flanking_size]] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]]')
        exit(1)

    # Define arguments
    ref_genome = args.ref_genome
    annotation = args.annotation
    input_vcf = args.input_vcf
    output_vcf = args.output_vcf
    distance = args.distance
    mask = args.mask
    model = args.model
    flanking_size = args.flanking_size
    model_type = args.model_type
    precision = args.precision
    
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
        exit(1)

    # Adding annotation to the header
    header = vcf.header
    header.add_line('##INFO=<ID=OpenSpliceAI,Number=.,Type=String,Description="OpenSpliceAI variant '
                    'annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')

    # Generating output VCF file
    print('\t[INFO] Generating output VCF file')
    os.makedirs(os.path.dirname(output_vcf), exist_ok=True)
    try:
        output = pysam.VariantFile(output_vcf, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('Error generating output VCF file: {}'.format(e))
        exit(1)

    # Setup the Annotator based on reference genome and annotation
    logging.info('Initializing Annotator class')
    ann = Annotator(ref_genome, annotation, model, model_type, flanking_size)

    # Obtain delta score for each variant in VCF
    for record in tqdm(vcf):
        scores = get_delta_scores(record, ann, distance, mask, flanking_size, precision)
        if scores:
            record.info['OpenSpliceAI'] = scores
        output.write(record)

    # Close input and output VCF files
    vcf.close()
    output.close()
    logging.info('Annotation completed and written to output VCF file')
    
    print("--- %s seconds ---" % (time.time() - start_time))