import argparse
import os
import gffutils
import gffutils

# Removed import openspliceai.create_data.utils as utils to avoid sklearn dependency

def create_or_load_db(gff_file, db_file='gff.db'):
    """
    Create a gffutils database from a GFF file, or load it if it already exists.
    """
    if not os.path.exists(db_file):
        print("Creating new database...")
        db = gffutils.create_db(gff_file, dbfn=db_file, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
    else:
        print("Loading existing database...")
        db = gffutils.FeatureDB(db_file)
    return db

def get_canonical_transcript(gene, db):
    """
    Select the canonical transcript for a gene (longest transcript).
    """
    # Look for both 'mRNA' and 'transcript' feature types
    transcripts_mRNA = list(db.children(gene, featuretype='mRNA', order_by='start'))
    transcripts_transcript = list(db.children(gene, featuretype='transcript', order_by='start'))
    
    transcripts = transcripts_mRNA + transcripts_transcript
        
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
    db = create_or_load_db(args.gff_file, db_file=f'{args.gff_file}_db')

    with open(args.output_file, 'w') as out_f:
        # Write header
        out_f.write("#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\n")

        for gene in db.features_of_type('gene'):
            # Filter for protein_coding genes
            # Check gene_biotype (Standard/Ensembl) or gene_type (MANE/Gencode)
            is_protein_coding = False
            if 'gene_biotype' in gene.attributes and gene.attributes['gene_biotype'][0] == 'protein_coding':
                is_protein_coding = True
            elif 'gene_type' in gene.attributes and gene.attributes['gene_type'][0] == 'protein_coding':
                is_protein_coding = True
            
            # If biotype/type is specified but not protein_coding, skip.
            # If neither attribute exists, decision depends on strictness. 
            # Given the loop logic "Filter for protein_coding", we should probably skip if we can't confirm it's protein_coding?
            # Or if it's missing, maybe include it?
            # Previous logic was "if biotype exists and != protein_coding then skip". 
            # I will preserve that: if we find a type attribute and it says NOT protein_coding, we skip.
            # If we find NO type attribute, we include it (assumed safe or user wants it).
            # BUT, the user's issue was empty output, implying things got filtered out or not found.
            # In MANE GFF, `gene_type=protein_coding` IS present.
            # So I should check:
            
            skip = False
            if 'gene_biotype' in gene.attributes:
                if gene.attributes['gene_biotype'][0] != 'protein_coding':
                    skip = True
            elif 'gene_type' in gene.attributes:
                if gene.attributes['gene_type'][0] != 'protein_coding':
                    skip = True
            
            # If we want to strictly enforce protein_coding when the attribute exists.
            # For MANE, we saw gene_type=protein_coding.
            
            if skip:
                continue

            transcript = get_canonical_transcript(gene, db)
            if not transcript:
                continue

            exons = list(db.children(transcript, featuretype='exon', order_by='start'))
            if not exons:
                continue

            # Extract coordinates
            # GFF is 1-based, output requires 0-based start, 1-based end.
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
            # Try 'Name', then 'gene_name', then 'ID'
            gene_name = gene.id
            if 'Name' in gene.attributes:
                gene_name = gene.attributes['Name'][0]
            elif 'gene_name' in gene.attributes:
                gene_name = gene.attributes['gene_name'][0]
            
            out_f.write(f"{gene_name}\t{gene.seqid}\t{gene.strand}\t{tx_start}\t{tx_end}\t{exon_start_str}\t{exon_end_str}\n")

    print(f"Done. Output written to {args.output_file}")

if __name__ == "__main__":
    main()
