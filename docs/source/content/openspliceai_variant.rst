
|


.. _variant_subcommand:

variant
=======

The ``variant`` subcommand evaluates the impact of genomic variants (SNPs and small INDELs) on splice sites by comparing model predictions on the reference (wild-type) sequence to predictions on the altered (mutant) sequence. It annotates a VCF file with “delta” scores for four events—acceptor gain, acceptor loss, donor gain, and donor loss—along with the relative position of each event. These delta scores reflect how much the mutation modifies the model’s predicted splice site strength and location.

|

Overview
--------

Similar to SpliceAI’s variant annotation approach (Jaganathan et al., 2019), the ``variant`` subcommand:

- **Parses** a user-provided VCF file.
- **Retrieves** the reference genome context from a FASTA file.
- **Loads** a trained OpenSpliceAI model (PyTorch *or* Keras).
- **Predicts** splice site probabilities for both wild-type and mutated sequences.
- **Computes** the maximum change in donor or acceptor probability (delta scores) within a fixed window (default ±50 nt) around each variant for each event.
- **Outputs** an annotated VCF, including delta scores and delta positions for each allele.

|

Input Files
-----------

1. **VCF File**

   A standard VCF file containing variants (SNPs, small insertions, and deletions).

2. **Reference Genome (FASTA)**

   A FASTA file used to extract wild-type sequences. The subcommand checks that the reference allele in the VCF matches the reference genome.

3. **Annotation File**

   A gene annotation table that defines the genomic regions to consider. You may pass the
   built-in shortcuts ``grch37`` (GENCODE V24lift37 canonical) or ``grch38`` (GENCODE V24
   canonical) — these tables **ship inside the package**
   (``openspliceai/variant/annotations/{grch37,grch38}.txt``) and are resolved automatically
   regardless of the current working directory — or supply a path to your own custom
   annotation file in the same format. Variants outside of annotated genes or too close to the
   chromosome ends (by default, within the flanking region) are skipped.

4. **Trained Model Checkpoint(s)**

   A directory or file containing one or more OpenSpliceAI model checkpoints (PyTorch ``.pt`` or Keras ``.h5``). The subcommand can average predictions across multiple models.

|

Output Files
------------

The primary output is a **VCF file** with added OpenSpliceAI annotations for each variant that passes filtering. Only SNVs and simple INDELs (REF or ALT is a single base) within genes are annotated. Variants in multiple genes have separate predictions for each gene. Each variant line in the annotated VCF contains a string in the ``INFO`` field with the format:

.. code-block:: text

   ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Field
     - Description
   * - ALLELE
     - Alternate allele
   * - SYMBOL
     - Gene symbol
   * - DS_AG
     - Delta score (acceptor gain)
   * - DS_AL
     - Delta score (acceptor loss)
   * - DS_DG
     - Delta score (donor gain)
   * - DS_DL
     - Delta score (donor loss)
   * - DP_AG
     - Delta position (acceptor gain)
   * - DP_AL
     - Delta position (acceptor loss)
   * - DP_DG
     - Delta position (donor gain)
   * - DP_DL
     - Delta position (donor loss)

- **Delta Scores**: Acceptor gain, acceptor loss, donor gain, and donor loss. The scores range from 0 to 1, and can be interpreted as the probability of the variant being splice-site-altering.
- **Delta Positions**: Relative positions (±50 by default) of these maximum changes. Positive values are downstream, negative values are upstream.

For example,
.. code-block:: text

   A|MYGENE|0.27|0.00|0.09|0.02|3|-4|7|-2

This string shows:
- Alternate allele is A
- We are on MYGENE
- The base positon 3 downstream of the variant has the highest acceptor gain score of 0.27
- The base position 4 upstream of the variant has the highest acceptor loss score of 0.00 (no loss)
- The base position 7 downstream of the variant has the highest donor gain score of 0.09
- The base position 2 upstream of the variant has the highest donor loss score of 0.02

|

Delta Score Computation
-----------------------

The “delta” score measures how much a mutation changes splice site predictions within a fixed window around the variant (default ±50 nucleotides, adjusted by ``-D`` parameter). For each variant, we compute reference predictions (:math:`d_{ref}`, :math:`a_{ref}`) and alternative predictions (:math:`d_{alt}`, :math:`a_{alt}`) for donor (:math:`d`) and acceptor (:math:`a`) channels:

.. math::
   \mathrm{DS}(\mathrm{Acceptor\,Gain}) = \max\bigl(a_{alt} - a_{ref}\bigr)
   :label: eq:12

.. math::
   \mathrm{DS}(\mathrm{Acceptor\,Loss}) = \max\bigl(a_{ref} - a_{alt}\bigr)
   :label: eq:13

.. math::
   \mathrm{DS}(\mathrm{Donor\,Gain}) = \max\bigl(d_{alt} - d_{ref}\bigr)
   :label: eq:14

.. math::
   \mathrm{DS}(\mathrm{Donor\,Loss}) = \max\bigl(d_{ref} - d_{alt}\bigr)
   :label: eq:15

where each maximum is taken over a window of 101 positions (±50) centered on the variant. The position of the maximum difference is recorded as the “delta position” (negative if upstream of the variant, positive if downstream).

|

Usage
-----

.. code-block:: text

   usage: openspliceai variant [-h] -R reference -A annotation [-I [input]] [-O [output]] [-D [distance]] [-M [mask]] [--model MODEL]
                              [--flanking-size {80,400,2000,10000}] [--model-type {keras,pytorch}] [--precision PRECISION]
                              [--batch-size BATCH_SIZE]

   optional arguments:
         -h, --help            show this help message and exit
         -R, --ref-genome reference
                                 path to the reference genome fasta file
         -A, --annotation annotation
                                 "grch37" (GENCODE V24lift37 canonical annotation file in package), "grch38" (GENCODE V24 canonical annotation file in package), or path to a similar custom gene annotation file
         -I, --input-vcf [input]
                                 path to the input VCF file, defaults to standard in
         -O, --output-vcf [output]
                                 path to the output VCF file, defaults to standard out
         -D, --distance [distance]
                                 maximum distance between the variant and gained/lost splice site, defaults to 50
         -M, --mask [mask]     mask scores representing annotated acceptor/donor gain and unannotated acceptor/donor loss, defaults to 0
         --model, -m MODEL     Path to a SpliceAI model file, or path to a directory of SpliceAI models, or "SpliceAI" for the default model
         --flanking-size, -f {80,400,2000,10000}
                                 Sum of flanking sequence lengths on each side of input (i.e. 40+40)
         --model-type, -t {keras,pytorch}
                                 Type of model file (keras or pytorch)
         --precision, -p PRECISION
                                 Number of decimal places to round the output scores
         --batch-size, -b BATCH_SIZE
                                 Number of windows per GPU forward pass. >1 enables batched inference (pytorch only) for large speedups on many-variant inputs; 1 (default) preserves the exact original per-variant path

|

Batched inference (``--batch-size``)
------------------------------------

By default (``--batch-size 1``) ``variant`` scores one window at a time, exactly reproducing
the original per-variant code path. On a **PyTorch** model you can pass ``--batch-size N`` (with
``N > 1``) to score many variants per forward pass, which gives large speedups on VCFs with many
variants — especially on a GPU. The batched path is numerically equivalent to the default: it
deduplicates the shared reference window across a position's alternate alleles and groups windows
by length so single-nucleotide and INDEL variants are handled correctly, then applies the same
crop / argmax / mask / formatting steps.

.. note::

   - Batching only applies to PyTorch models. With ``--model-type keras`` the subcommand always
     uses the per-variant path regardless of ``--batch-size``.
   - A good starting point on a GPU is ``--batch-size 32`` to ``256``; larger batches use more
     GPU memory. On CPU, batching still helps but the gains are smaller.
   - Two optional environment variables tune throughput on NVIDIA GPUs:
     ``OSAI_CUDNN_BENCH=1`` enables cuDNN autotuning (all windows share one length, so this is
     effective), and ``OSAI_TF32=1`` (the default) enables the TF32 fast path on Ampere/A100
     hardware. TF32 adds ~1e-3 of noise — far below the rounded score resolution — so set
     ``OSAI_TF32=0`` if you need output that is bit-for-bit identical to the per-variant path.

.. code-block:: bash

   openspliceai variant \
      --input-vcf many_variants.vcf \
      --ref-genome GRCh38.fa \
      --annotation grch38 \
      --model /path/to/pytorch_models/ \
      --model-type pytorch \
      --flanking-size 10000 \
      --batch-size 64 \
      --output-vcf annotated_variants.vcf

|

Examples
--------

Example: Discovering pathogenic human variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openspliceai variant \
      --input-vcf input_variants.vcf \
      --ref-genome GRCh38.fa \
      --annotation grch38 \
      --model /path/to/pytorch_models/ \
      --model-type pytorch \
      --flanking-size 400 \
      --distance 100 \
      --mask 1 \
      --output-vcf annotated_variants.vcf

This command:

1. **Loads** the reference genome from ``GRCh38.fa``.
2. **Reads** the built-in ``grch38`` gene annotation that ships inside the package.
3. **Scans** the directory ``/path/to/pytorch_models/`` for PyTorch checkpoints, averaging predictions from all found models.
4. **Computes** donor and acceptor delta scores within ±100 nucleotides of each variant.
5. **Masks** scores representing annotated acceptor/donor gain and unannotated acceptor/donor loss. (This is useful for novel/pathogenic variant discovery).
6. **Writes** a new VCF (``annotated_variants.vcf``) with the masked annotations for the four delta scores and positions in the ``INFO`` field.

|

Example: Annotating all variant in VCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openspliceai variant \
      --input-vcf sample_variants.vcf \
      --ref-genome GRCh37.fa \
      --annotation grch37 \
      --model /path/to/keras_models/ \
      --model-type keras \
      --flanking-size 10000 \
      --output-vcf annotated_samples.vcf
      --precision 3

This command:

1. **Loads** the reference genome from ``GRCh37.fa``.
2. **Reads** the built-in ``grch37`` gene annotation that ships inside the package.
3. **Scans** the directory ``/path/to/keras_models/`` for Keras checkpoints, averaging predictions from all found models.
4. **Computes** donor and acceptor delta scores within ±50 nucleotides of each variant.
5. **Writes** a new VCF (``annotated_samples.vcf``) with all annotations for the four delta scores and positions in the ``INFO`` field (these are unmasked by default, so all scores are included). The scores are rounded to three decimal places.

|

Processing Pipeline
-------------------

#. **VCF Parsing/Filtering**

   - For each variant, the subcommand checks if it lies within an annotated gene region. If it isn't, it will be filtered out.  
   - Variants that are too close to the chromosome ends (< ``flanking-size`` / 2 bases on either side), have deletions of length > 2 * ``distance``, or have reference alleles mismatching the FASTA are automatically skipped.

#. **Reference & Mutant Sequence Extraction**

   - A window of :math:`2 \times \text{dist_var} + 1` around the variant is extracted from the FASTA (e.g., 101 nt for ``--dist-var=50``).
   - For each alternative allele, the subcommand constructs a mutant sequence by substituting the variant base(s).

#. **Model Loading & Prediction**

   - The user may supply **PyTorch** (``.pt``) or **Keras** (``.h5``) model checkpoints.  
   - If a directory is provided, predictions from each model are averaged.  
   - Wild-type (reference) and mutant sequences are one-hot encoded and fed into the model(s) in evaluation mode.

#. **Delta Score Calculation**

   - The difference in donor/acceptor probabilities (:math:`d_{alt}-d_{ref}`, :math:`a_{alt}-a_{ref}`) is computed across the window.  
   - The maximum positive or negative differences yield the four delta scores: donor gain, donor loss, acceptor gain, and acceptor loss (Equations :eq:`eq:12`–:eq:`eq:15`).

#. **VCF Annotation**

   - The four delta scores and their positions (relative to the variant site) are appended to each variant’s INFO field.  
   - The annotated VCF includes the annotations in the ``INFO`` field, following the string format described above.

|

Workflow
~~~~~~~~
.. image:: ../_images/variant_workflow.png
   :alt: Variant Workflow
   :align: center

|

Conclusion
----------

The ``variant`` subcommand enables fine-grained analysis of how single-nucleotide changes or small variants affect splice site usage. Essentially, it provides a convenient, post-hoc annotation step for variant effect prediction.

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
   :align: center