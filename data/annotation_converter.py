import argparse
import os
import gffutils
import openspliceai.create_data.utils as utils

def get_canonical_transcript(gene, db):
    """
    Select the canonical transcript for a gene (longest transcript).
    """
    transcripts = list(db.children(gene, featuretype='mRNA', order_by='start'))
    if not transcripts:
        return None
    
    max_trans = transcripts[0]
    max_len = max_trans.end - max_trans.start + 1
    for transcript in transcripts:
        length = transcript.end - transcript.start + 1
        if length > max_len:
            max_trans = transcript
            max_len = length
    return max_trans

def main():
    parser = argparse.ArgumentParser(description="Convert GFF3 annotation to TSV format.")
    parser.add_argument("gff_file", help="Path to the input GFF3 file.")
    parser.add_argument("output_file", help="Path to the output TSV file.")
    args = parser.parse_args()

    print(f"Processing {args.gff_file}...")
    db = utils.create_or_load_db(args.gff_file, db_file=f'{args.gff_file}_db')

    with open(args.output_file, 'w') as out_f:
        # Write header
        out_f.write("#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\n")

        for gene in db.features_of_type('gene'):
            # Filter for protein_coding genes
            if 'gene_biotype' in gene.attributes and gene.attributes['gene_biotype'][0] != 'protein_coding':
                continue
            
            # Handle potential different attribute names or missing attributes if necessary
            # For now, sticking to the logic seen in create_datafile.py which checks gene_biotype

            transcript = get_canonical_transcript(gene, db)
            if not transcript:
                continue

            exons = list(db.children(transcript, featuretype='exon', order_by='start'))
            if not exons:
                continue

            # Extract coordinates
            tx_start = transcript.start - 1
            tx_end = transcript.end
            
            exon_starts = []
            exon_ends = []
            
            for exon in exons:
                exon_starts.append(str(exon.start - 1))
                exon_ends.append(str(exon.end))
            
            exon_start_str = ",".join(exon_starts) + ","
            exon_end_str = ",".join(exon_ends) + ","
            
            # Name
            gene_name = gene.attributes.get('Name', [gene.id])[0]
            # Sometimes Name is not present, fallback to ID. 
            # In Ensembl GFF, Name usually exists.
            
            out_f.write(f"{gene_name}\t{gene.seqid}\t{gene.strand}\t{tx_start}\t{tx_end}\t{exon_start_str}\t{exon_end_str}\n")

    print(f"Done. Output written to {args.output_file}")

if __name__ == "__main__":
    main()
