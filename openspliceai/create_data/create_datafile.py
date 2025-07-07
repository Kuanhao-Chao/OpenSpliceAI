"""
Filename: create_datafile.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Create datafile.h5 for training and testing splice site prediction models.
"""

import os
from Bio.Seq import Seq  # Import the Seq class
from Bio import SeqIO, SeqRecord
import time
import openspliceai.create_data.utils as utils
import openspliceai.create_data.paralogs as paralogs

donor_motif_counts = {}  # Initialize counts
acceptor_motif_counts = {}  # Initialize counts

def get_sequences_and_labels(db, output_dir, seq_dict, chrom_dict, train_or_test, parse_type="canonical", biotype="protein-coding", canonical_only=True, write_fasta=False):
    """
    Extract sequences for each protein-coding gene, reverse complement sequences for genes on the reverse strand,
    and label donor and acceptor sites correctly based on strand orientation.
    """
    fw_stats = open(os.path.join(output_dir, f"{train_or_test}_stats.txt"), "w")
    if write_fasta:
        fasta_handle = open(os.path.join(output_dir, f"{train_or_test}.fa"), "w")  # Open a file to write FASTA format sequences
    NAME = []      # Gene Name
    CHROM = []     # Chromosome
    STRAND = []    # Strand in which the gene lies (+ or -)
    TX_START = []  # Position where transcription starts
    TX_END = []    # Position where transcription ends
    SEQ = []       # Nucleotide sequence
    LABEL = []     # Label for each nucleotide in the sequence
    for gene in db.features_of_type('gene'):
        if "exception" in gene.attributes.keys() and gene.attributes["exception"][0] == "trans-splicing":
            continue
        if gene.seqid not in chrom_dict:
            continue
        if biotype == "protein-coding":
            if gene.attributes["gene_biotype"][0] != "protein_coding":
                continue
        elif biotype == "non-coding":
            if gene.attributes["gene_biotype"][0] != "lncRNA" and gene.attributes["gene_biotype"][0] != "ncRNA":
                continue
        elif biotype == "all":
            if gene.attributes["gene_biotype"][0] != "protein_coding" and gene.attributes["gene_biotype"][0] != "lncRNA" and gene.attributes["gene_biotype"][0] != "ncRNA":
                continue
        else:
            continue
        chrom_dict[gene.seqid] += 1
        gene_id = gene.id

        # Process the gene sequence
        gene_seq = seq_dict[gene.seqid].seq[gene.start-1:gene.end].upper()  # Extract gene sequence
        if gene.strand == '-':
            gene_seq = gene_seq.reverse_complement() # reverse complement the sequence
        gene_seq = str(gene_seq.upper())
        print(f"\tProcessing gene {gene_id} on chromosome {gene.seqid}..., len(gene_seq): {len(gene_seq)}")
        labels = [0] * len(gene_seq)  # Initialize all labels to 0
        if biotype =="protein-coding":
            transcripts = list(db.children(gene, featuretype='mRNA', order_by='start'))
        elif biotype =="non-coding":
            transcripts = list(db.children(gene, level=1, order_by='start'))
        if len(transcripts) == 0:
            continue
        elif len(transcripts) > 1:
            print(f"\tGene {gene_id} has multiple transcripts: {len(transcripts)}")

        # Selecting which mode to process the data
        transcripts_ls = []
        if parse_type == 'canonical':
            max_trans = transcripts[0]
            max_len = max_trans.end - max_trans.start + 1
            for transcript in transcripts:
                if transcript.end - transcript.start + 1 > max_len:
                    max_trans = transcript
                    max_len = transcript.end - transcript.start + 1
            transcripts_ls = [max_trans]
        elif parse_type == 'all_isoforms':
            transcripts_ls = transcripts
        
        # Process transcripts
        for transcript in transcripts_ls:
            exons = list(db.children(transcript, featuretype='exon', order_by='start'))
            if len(exons) > 1:
                for i in range(len(exons) - 1):
                    # Donor site is one base after the end of the current exon
                    first_site = exons[i].end - gene.start  # Adjusted for python indexing
                    # Acceptor site is at the start of the next exon
                    second_site = exons[i + 1].start - gene.start          
                    # Adjusted for python indexing
                    if gene.strand == '+':
                        d_idx = first_site
                        a_idx = second_site
                    elif gene.strand == '-':
                        d_idx = len(labels) - second_site-1
                        a_idx = len(labels) - first_site-1
                    # Check if the donor / acceptor sites are valid
                    d_motif = str(gene_seq[d_idx+1:d_idx+3])
                    a_motif = str(gene_seq[a_idx-2:a_idx])
                    if not canonical_only:
                        labels[d_idx] = 2   # Mark donor site
                        labels[a_idx] = 1  # Mark acceptor site
                    elif (d_motif == "GT" and a_motif == "AG") or (d_motif == "GC" and a_motif == "AG") or (d_motif == "AT" and a_motif == "AC"):
                        labels[d_idx] = 2   # Mark donor site
                        labels[a_idx] = 1  # Mark acceptor site
            labels_str = ''.join(str(num) for num in labels)
            NAME.append(gene_id)
            CHROM.append(gene.seqid)
            STRAND.append(gene.strand)
            TX_START.append(str(gene.start))
            TX_END.append(str(gene.end))
            SEQ.append(gene_seq)
            LABEL.append(labels_str)

            # Write to the FASTA file if the path is provided
            if write_fasta:
                record = SeqRecord.SeqRecord(
                    Seq(gene_seq),
                    id=gene_id,
                    description=f"{gene.seqid}:{gene.start}-{gene.end}({gene.strand})"
                )
                SeqIO.write(record, fasta_handle, "fasta")

            fw_stats.write(f"{gene.seqid}\t{gene.start}\t{gene.end}\t{gene.id}\t{1}\t{gene.strand}\n")
            utils.check_and_count_motifs(gene_seq, labels, donor_motif_counts, acceptor_motif_counts)
    print(f"Total SEQ: {len(SEQ)}")
    fw_stats.close()
    if write_fasta:
        fasta_handle.close()  # Close the FASTA file handle
    return [NAME, CHROM, STRAND, TX_START, TX_END, SEQ, LABEL]


def create_datafile(args):
    print("Running OpenSpliceAI with 'create-data' mode")
    print("--- Step 1: Creating datafile.h5 ... ---")
    start_time = time.process_time()
    
    # Use gffutils to parse annotation file
    os.makedirs(args.output_dir, exist_ok=True)
    db = utils.create_or_load_db(args.annotation_gff, db_file=f'{args.annotation_gff}_db')
    seq_dict = SeqIO.to_dict(SeqIO.parse(args.genome_fasta, "fasta"))
        
    # Find all distinct chromosomes and split them
    TRAIN_CHROM_GROUP, TEST_CHROM_GROUP = utils.split_chromosomes(seq_dict, method=args.split_method, split_ratio=args.split_ratio)
    print("* TRAIN_CHROM_GROUP: ", TRAIN_CHROM_GROUP)
    print("* TEST_CHROM_GROUP: ", TEST_CHROM_GROUP)

    # Collect sequences and labels for testing and/or training groups
    if args.chr_split == 'test':
        print("> Creating test datafile...")
        test_data = get_sequences_and_labels(db, args.output_dir, seq_dict, TEST_CHROM_GROUP, 'test', parse_type=args.parse_type, biotype=args.biotype, canonical_only=args.canonical_only, write_fasta=args.write_fasta)
        paralogs.write_h5_file(args.output_dir, "test", test_data)
    elif args.chr_split == 'train-test':
        print("> Creating train datafile...")
        train_data = get_sequences_and_labels(db, args.output_dir, seq_dict, TRAIN_CHROM_GROUP, 'train', parse_type=args.parse_type, biotype=args.biotype, canonical_only=args.canonical_only, write_fasta=args.write_fasta)
        print("> Creating test datafile...")
        test_data = get_sequences_and_labels(db, args.output_dir, seq_dict, TEST_CHROM_GROUP, 'test', parse_type=args.parse_type, biotype=args.biotype, canonical_only=args.canonical_only, write_fasta=args.write_fasta)
        if args.remove_paralogs:
            # Remove homologous sequences
            print("> Removing homologous sequences in test...")
            train_data, test_data = paralogs.remove_paralogous_sequences(train_data, test_data, args.min_identity, args.min_coverage, args.output_dir, "test")

        # Split the training data into training and validation sets (90:10 split)
        print("> Splitting training data into training and validation sets...")
        train_data, val_data = utils.split_train_val(train_data, args.val_split_ratio)
        if args.remove_paralogs:
            # Remove homologous sequences
            print("> Removing homologous sequences in validation...")
            train_data, val_data = paralogs.remove_paralogous_sequences(train_data, val_data, args.min_identity, args.min_coverage, args.output_dir, "validation")

        # Write the filtered data to h5 files
        paralogs.write_h5_file(args.output_dir, "train", train_data)
        paralogs.write_h5_file(args.output_dir, "validation", val_data)
        paralogs.write_h5_file(args.output_dir, "test", test_data)
    utils.print_motif_counts(donor_motif_counts, acceptor_motif_counts)    
    print("--- %s seconds ---" % (time.process_time() - start_time))
