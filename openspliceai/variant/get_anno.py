import argparse
import gffutils
import os

def parse_arguments():
    """Parse options for the standalone historical annotation converter."""
    parser = argparse.ArgumentParser(description="Convert GFF file to custom tabular format.")
    parser.add_argument(
        "--gff",
        required=True,
        help="Path to the input GFF file (e.g., data/GRCh38_chr21_chr22.gff)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output custom annotation file (e.g., data/grch38_chr21_chr22.txt)."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the temporary GFF database. If not provided, a temporary database will be created."
    )
    return parser.parse_args()

def create_gff_database(gff_file, db_file=None):
    """
    Create a gffutils database from the GFF file.
    If db_file is None, a temporary in-memory database is created.
    """
    if db_file and os.path.exists(db_file):
        print(f"Connecting to existing GFF database: {db_file}")
        db = gffutils.FeatureDB(db_file)
    else:
        if db_file:
            print(f"Creating new GFF database: {db_file}")
        else:
            print("Creating in-memory GFF database.")
        db = gffutils.create_db(
            gff_file,
            dbfn=db_file if db_file else ":memory:",
            force=True,
            keep_order=True,
            merge_strategy='create_unique',
            sort_attribute_values=True,
            verbose=True
        )
    return db

def extract_gene_info(db):
    """
    Extract gene information from the GFF database.
    Returns a list of dictionaries containing gene details.
    """
    gene_data = []
    for gene in db.features_of_type('gene'):
        # extract gene name
        gene_name = None
        for key in ['gene_name', 'Name', 'gene', 'ID']:
            if key in gene.attributes:
                gene_name = gene.attributes[key][0]
                break
        if not gene_name:
            gene_name = gene.id  # fallback to gene ID if name not available

        chrom = gene.seqid
        strand = gene.strand
        tx_start = gene.start
        tx_end = gene.end

        # Extract exons
        exons = list(db.children(gene, featuretype='exon', order_by='start'))
        if not exons:
            continue  # Skip genes without exons

        exon_starts = []
        exon_ends = []
        for exon in exons:
            exon_starts.append(str(exon.start))
            exon_ends.append(str(exon.end))
        
        # Concatenate exon positions with commas
        exon_start_str = ",".join(exon_starts) + ","
        exon_end_str = ",".join(exon_ends) + ","

        gene_entry = {
            "NAME": gene_name,
            "CHROM": chrom,
            "STRAND": strand,
            "TX_START": tx_start,
            "TX_END": tx_end,
            "EXON_START": exon_start_str,
            "EXON_END": exon_end_str
        }
        gene_data.append(gene_entry)
    return gene_data

def write_output(gene_data, output_file):
    """
    Write the extracted gene information to the output file in tab-separated format.
    """
    header = ["#NAME", "CHROM", "STRAND", "TX_START", "TX_END", "EXON_START", "EXON_END"]
    with open(output_file, "w") as out_f:
        out_f.write("\t".join(header) + "\n")
        for gene in gene_data:
            row = [
                gene["NAME"],
                gene["CHROM"],
                gene["STRAND"],
                str(gene["TX_START"]),
                str(gene["TX_END"]),
                gene["EXON_START"],
                gene["EXON_END"]
            ]
            out_f.write("\t".join(row) + "\n")
    print(f"Output written to {output_file}")

def main():
    """Run the module entrypoint with its documented command arguments."""
    args = parse_arguments()

    # Create or connect to GFF database
    db = create_gff_database(args.gff, args.db)

    # Extract gene information
    gene_data = extract_gene_info(db)

    # Write to output file
    write_output(gene_data, args.output)

if __name__ == "__main__":
    main()