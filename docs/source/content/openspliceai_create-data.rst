
|


.. _create-data_subcommand:

create-data
===========

The ``create-data`` subcommand processes genomic sequences (FASTA) and genome annotations (GFF/GTF) to produce HDF5-formatted (Hierarchical Data Format version 5) datasets for splice site prediction. It transforms a reference genome and its annotations into one-hot-encoded pre-mRNA sequences (input tensor :math:`X`) along with corresponding splice site labels (output tensor :math:`Y`). This subcommand is part of the OpenSpliceAI toolkit and is designed to create high-quality training and testing datasets.

.. Overview
.. --------

.. The ``create-data`` subcommand converts genomic and annotation files into structured datasets suitable for machine learning. It performs the following key tasks:

.. - **Input Processing**: Reads a reference genome in FASTA format and a corresponding annotation file in GFF/GTF.
.. - **Canonical Transcript Selection**: For each gene locus, the longest (canonical) transcript is chosen (default: protein-coding only).
.. - **Sequence Chunking and One-Hot Encoding**: Gene sequences are divided into overlapping chunks, one-hot encoded, and padded as needed.
.. - **Splice Site Labeling**: Generates labels for donor, acceptor, and non-splice sites based on curated annotations.
.. - **Training/Testing Splitting**: Chromosomes are split into training and testing sets (default ~80:20 ratio) using either automatic or user-specified schemes.
.. - **Quality Control**: Filters out pseudogenes and detects/removes paralogous gene sequences to prevent data leakage.

Input Files
-----------

1. **Reference Genome (FASTA)**
   
   A FASTA file containing the genomic sequence.

   **Example:**

   ``GCF_000001405.40_GRCh38.p14_genomic.fna <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/GCF_000001405.40_GRCh38.p14_genomic.fna>``

2. **Reference Annotation (GFF/GTF)**
   
   A GFF3 file containing genome annotations.

   **Example:**

   ``MANE.GRCh38.v1.3.refseq_genomic.gff <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/MANE.GRCh38.v1.3.refseq_genomic.gff>``

|

Output Files
------------

The subcommand produces several outputs:

- **Training Dataset:**

  ``dataset_train.h5 <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/dataset_train.h5>``

- **Validation Dataset:**

  ``dataset_validation.h5``

- **Testing Dataset:**

  ``dataset_test.h5 <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/dataset_test.h5>``

- **Intermediate Files:**

  - ``datafile_train.h5``
  - ``datafile_validation.h5``
  - ``datafile_test.h5``
  - ``stats.txt``

.. note::

   With the default ``--chr-split train-test``, the subcommand writes three datasets:
   ``dataset_train.h5``, ``dataset_validation.h5``, and ``dataset_test.h5`` (along with the
   matching ``datafile_*.h5`` intermediates). When ``--chr-split test`` is used, only the
   ``test`` split is produced.

These files contain the processed gene sequences (tensor :math:`X`) and splice site labels (tensor :math:`Y`).

|

Usage
------

.. code-block:: text


   usage: openspliceai create-data [-h] --annotation-gff ANNOTATION_GFF --genome-fasta GENOME_FASTA --output-dir OUTPUT_DIR
                                 [--parse-type {canonical,all_isoforms}] [--biotype {protein-coding,non-coding}] [--chr-split {train-test,test}]
                                 [--split-method {random,human}] [--split-ratio SPLIT_RATIO] [--canonical-only] [--flanking-size FLANKING_SIZE]
                                 [--verify-h5] [--remove-paralogs] [--min-identity MIN_IDENTITY] [--min-coverage MIN_COVERAGE] [--write-fasta]

   optional arguments:
         -h, --help            show this help message and exit
         --annotation-gff ANNOTATION_GFF
                                 Path to the GFF file
         --genome-fasta GENOME_FASTA
                                 Path to the FASTA file
         --output-dir OUTPUT_DIR
                                 Output directory to save the data
         --parse-type {canonical,all_isoforms}
                                 Type of transcript processing
         --biotype {protein-coding,non-coding}
                                 Biotype of transcript processing
         --chr-split {train-test,test}
                                 Whether to obtain testing or both training and testing groups
         --split-method {random,human}
                                 Chromosome split method for training and testing dataset
         --split-ratio SPLIT_RATIO
                                 Ratio of training and testing dataset
         --canonical-only      Flag to obtain only canonical splice site pairs
         --flanking-size FLANKING_SIZE
                                 Sum of flanking sequence lengths on each side of input (i.e. 40+40)
         --verify-h5           Verify the generated HDF5 file(s)
         --remove-paralogs     Remove paralogous sequences between training and testing dataset
         --min-identity MIN_IDENTITY
                                 Minimum minimap2 alignment identity for paralog removal between training and testing dataset
         --min-coverage MIN_COVERAGE
                                 Minimum minimap2 alignment coverage for paralog removal between training and testing dataset
         --write-fasta         Flag to write out sequences into fasta files


|


Examples
--------

Example: Creating the Human MANE Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate training and testing datasets for the human MANE dataset, first download the following files:

1. **Reference Genome**:  
   ``GCF_000001405.40_GRCh38.p14_genomic.fna``

2. **Reference Annotation**:  
   ``MANE.GRCh38.v1.3.refseq_genomic.gff``

Then, execute the command:

.. code-block:: bash

   openspliceai create-data \
      --genome-fasta  GCF_000001405.40_GRCh38.p14_genomic.fna \
      --annotation-gff MANE.GRCh38.v1.3.refseq_genomic.gff \
      --output-dir train_test_dataset_MANE/

After a successful run, the following files will be generated:

- **dataset_train.h5** (main file for training)
- **dataset_validation.h5** (held-out validation split)
- **dataset_test.h5** (main file for testing)
- **Intermediate Files**: ``datafile_train.h5``, ``datafile_validation.h5``, ``datafile_test.h5``, and ``stats.txt``


|

Processing Pipeline
-------------------

1. Gene Sequence Processing
   ===========================

   - **Sequence Conversion**:  
     Each gene is transformed into a 3D tensor :math:`X` using one-hot encoding. The nucleotides are represented as follows:
     
     - **A** = [1, 0, 0, 0]
     - **C** = [0, 1, 0, 0]
     - **G** = [0, 0, 1, 0]
     - **T (or U)** = [0, 0, 0, 1]
     - **N** = [0, 0, 0, 0]

   - **Chunking and Padding**:  
     The gene sequence is split into overlapping chunks with:
     
     - **Window size (W)**: Default 5,000 nucleotides.
     - **Flanking sequence (F)**: Typical sizes include 80, 400, 2,000, or 10,000 nucleotides.
     
     The tensor :math:`X` is given by:

     .. math::
        \lceil L / W \rceil \times (F + W) \times 4

     where :math:`L` is the gene length. Any remaining sequence is padded with ``N`` characters so that each chunk is a multiple of the window size.

     *Example*: For a gene of 12,000 nucleotides with ``W = 5000`` and ``F = 10,000``, the tensor :math:`X` will have the shape:

     .. math::
        \lceil 12000 / 5000 \rceil \times (10000 + 5000) \times 4 = 3 \times 15000 \times 4

2. Label Generation
   =================

   - **Label Tensor Construction**:  
     Labels are generated from genome annotations and encoded as follows:

     - **Donor site**: [0, 0, 1]
     - **Acceptor site**: [0, 1, 0]
     - **Non-splice site**: [1, 0, 0]
     - **Padding**: [0, 0, 0]

     The resulting label tensor :math:`Y` has the shape:

     .. math::
        \lceil L / W \rceil \times W \times 3

3. Gene Sequence Chunking
   ========================

   - **Overlapping Chunks**:  
     Following SpliceAI’s methodology, gene sequences are divided into overlapping chunks using a step size equal to the window size (5,000 nucleotides). Flanking sequences (also defaulted to 5,000 nucleotides) are appended on each side.

     *Example*: A 22,000-nucleotide gene is divided into 5 chunks, resulting in a tensor shape of ``(5, 15,000, 4)`` for the sequences and ``(5, 5,000, 3)`` for the labels.

4. Canonical Transcript Selection
   ===============================

   - For each gene locus, the longest transcript is selected as the canonical transcript.
   - By default, the ``--biotype`` flag is set to **protein-coding**, thereby filtering out non-coding genes.

5. Training and Testing Data Splitting
   =====================================

   - **Automatic Splitting**:  
     If not explicitly provided, the toolkit splits chromosomes into training and testing sets with an approximate 80:20 ratio.
     
     - Chromosome lengths are retrieved from the GFFUtils database
       (https://github.com/daler/gffutils).
     - For non-human species, a random shuffling method is applied by default.
     
   - **User-Specified Splitting**:  
     Use the ``--chr-split`` option to manually specify chromosome assignments.

6. Removal of Pseudogenes and Paralogous Genes
   =============================================

   - **Pseudogenes Filtering**:  
     Genes marked as ``pseudogene`` (either in the feature type or via the ``gene_biotype`` attribute) are excluded from the test dataset.
     
   - **Paralogous Gene Removal**:  
     To avoid sequence similarity between training and testing sets (and thus data leakage), the toolkit uses *mappy* (a Python wrapper for minimap2) with the ``--asm20`` argument (allowing a divergence threshold of 5%). Test sequences sharing over 80% similarity and 80% coverage with training sequences are removed.

7. Splice Site Labeling Options
   =============================

   - **Canonical-only Mode**:  
     The ``--canonical-only`` flag restricts label generation to conserved splice site motifs. These include:
     
     - **U2-snRNP motifs**: ``GT-AG`` and ``GC-AG``
     - **U12-snRNP motifs**: ``GT-AG`` and ``AT-AC``

     This option helps mitigate the effect of misannotated splice sites.

|


Conclusion
----------

The ``create-data`` subcommand provides a robust framework for transforming genomic and annotation data into machine-learning-ready datasets. Its careful handling of transcript selection, sequence chunking, one-hot encoding, and rigorous filtering of pseudogenes and paralogous sequences ensures high-quality training and testing sets for splice site prediction. For more details and troubleshooting, please refer to the OpenSpliceAI help command or the official documentation.



|
|
|
|
|


.. image:: ../_images/jhu-logo-dark.png
   :alt: My Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ../_images/jhu-logo-white.png
   :alt: My Logo
   :class: logo, header-image only-dark
