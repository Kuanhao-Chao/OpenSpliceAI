
|

.. _Q&A:

Q & A
=====

.. dropdown:: Q: What is OpenSpliceAI?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-dark font-weight-bolder howtoclass
    :body: bg-dark text-left

    OpenSpliceAI is an open‐source, trainable deep learning framework for splice site prediction. It is a PyTorch‐based reimplementation and extension of the original SpliceAI model (Jaganathan et al., 2019). OpenSpliceAI replicates the core architecture of SpliceAI—a deep residual convolutional neural network—but enhances it with improved computational efficiency, modular design, and flexible training options. This framework supports training from scratch, transfer learning across species, model calibration, and variant effect analysis, making it a versatile tool for both human and non‐human genomic studies.

|

.. dropdown:: Q: Why do we need OpenSpliceAI?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    While SpliceAI set a high standard for splice site prediction, its reliance on outdated TensorFlow/Keras frameworks and human-specific training data limits its broader applicability. OpenSpliceAI addresses these limitations by:
    
    - **Utilizing PyTorch:** This modern framework offers better GPU efficiency, lower memory overhead, and seamless integration with contemporary machine learning workflows.
    - **Supporting Retraining and Transfer Learning:** Users can easily retrain models on species-specific datasets, reducing biases inherent in human-centric models.
    - **Improving Efficiency:** Enhanced data preprocessing, parallelized prediction, and optimized memory management allow for large-scale analyses on standard hardware.
    
    In essence, OpenSpliceAI provides researchers with a flexible, efficient, and extensible platform for splice site prediction across diverse species.

|

.. dropdown:: Q: I want to use OpenSpliceAI to predict splice sites. What should I do?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    To predict splice sites using OpenSpliceAI:
    
    1. **Obtain a Pre-trained Model:** You can use one of the available pre-trained models (e.g., OpenSpliceAI-MANE) provided by the project.
    2. **Prepare Your Input Files:** Provide a FASTA file containing your DNA sequences. If you wish to restrict predictions to protein-coding genes, include a GFF annotation file.
    3. **Run the Predict Subcommand:** Use the `predict` subcommand to process your input. The tool will one-hot encode your sequences, perform window-based predictions, and output BED files listing the coordinates and scores for predicted donor and acceptor sites.
    
    For detailed instructions, refer to the :ref:`quick-start_predict` page.

|

.. dropdown:: Q: I have a variant of interest. How can I use OpenSpliceAI to predict its effect on splicing?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    OpenSpliceAI offers the `variant` subcommand for evaluating the impact of genetic variants on splicing. To use it:
    
    1. **Prepare Your Files:** Ensure you have a VCF file containing the variants, a reference genome in FASTA format, and a gene annotation file.
    2. **Run the Variant Subcommand:** The tool compares splice site predictions between the wild-type (reference) sequence and the mutant sequence for each variant.
    3. **Delta Score Calculation:** For each variant, delta scores are computed for donor gain, donor loss, acceptor gain, and acceptor loss by assessing the maximum change in predicted splice site probability within a ±50 nt window.
    4. **Output:** The annotated VCF file will include these delta scores and the relative positions of the most significant changes.
    
    This process allows you to quantify how a variant may disrupt normal splicing patterns. See the :ref:`quick-start_variant` page for a step-by-step guide.

|

.. dropdown:: Q: How is OpenSpliceAI trained?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    OpenSpliceAI training is a multi-step process that mirrors the original SpliceAI methodology while adding modern enhancements:
    
    - **Data Preprocessing:** The `create-data` subcommand converts genomic sequences (FASTA) and annotations (GFF/GTF) into one-hot encoded tensors stored in HDF5 format. Only protein-coding genes are typically included.
    - **Model Training:** The `train` subcommand uses these datasets to train a deep residual convolutional neural network with adaptive learning techniques. OpenSpliceAI employs the AdamW optimizer with adaptive schedulers (MultiStepLR or CosineAnnealingWarmRestarts) and supports early stopping.
    - **Loss Functions:** Training can be performed using the standard categorical cross-entropy loss or an alternative focal loss, which emphasizes harder-to-classify examples.
    
    These methods ensure that the resulting models are both accurate and computationally efficient. More details are available in the online methods section of the documentation.

|

.. dropdown:: Q: How much resource do I need to train an OpenSpliceAI model myself?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    The hardware requirements for training OpenSpliceAI depend on your dataset size and chosen model parameters (such as flanking sequence length). In our study:
    
    - **CPU Resources:** Data preprocessing was performed on a 24-core Intel Xeon processor.
    - **GPU Resources:** Training was executed on a single Nvidia A100 GPU with 40 GB of memory. This allowed us to efficiently train models on the human MANE dataset and several non-human species.
    
    In general, for training on human-scale datasets, a modern GPU (e.g., Nvidia RTX series or better) with at least 16–24 GB of memory is recommended. Training on smaller species may require less GPU memory. Additionally, efficient data batching and optimized code in OpenSpliceAI help reduce the overall computational burden.

|

.. dropdown:: Q: What's the difference between scratch-training and transfer-learning? And what are the pros and cons?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    **Scratch-training** involves training a model from random initialization using species-specific datasets. This approach:
    
    - **Pros:**
      - Can yield optimal performance when ample high-quality data are available.
      - Allows the model to fully adapt to the unique characteristics of the target species.
    
    - **Cons:**
      - Requires a longer training time.
      - Demands more computational resources.
      - May suffer from instability if the dataset is limited.
    
    **Transfer-learning** leverages a pre-trained model (typically trained on human data) and fine-tunes it on a new dataset. This approach:
    
    - **Pros:**
      - Significantly reduces training time and resource requirements.
      - Often yields more stable and accurate results, especially for species with limited data.
      - Benefits from the already learned general splicing features.
    
    - **Cons:**
      - May not fully capture species-specific splicing nuances if the source and target domains are too dissimilar.
      - The fine-tuning process requires careful selection of layers to unfreeze to avoid overfitting.
    
    The choice between scratch-training and transfer-learning depends on the availability of training data and the degree of similarity between the source and target species.

|

.. dropdown:: Q: How do we interpret OpenSpliceAI output?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    The primary outputs of OpenSpliceAI are generated by the `predict` and `variant` subcommands:
    
    - **Predict Output:**  
      The `predict` subcommand produces BED files that list the predicted donor and acceptor sites along with their probability scores. These scores represent the likelihood that a specific nucleotide is a splice site. Higher scores indicate greater confidence in the prediction.
    
    - **Variant Output:**  
      The `variant` subcommand annotates a VCF file with “delta” scores and delta positions for each variant. These scores quantify the change in splice site strength between the wild-type and mutated sequences. For example, a delta score for acceptor gain reflects the maximum increase in the acceptor site probability within a 101-nt window around the variant.
    
    In both cases, the outputs are designed to be directly interpretable and can be further analyzed or integrated into downstream genomic pipelines.

|

.. dropdown:: Q: My predicted donor/acceptor scores are unexpectedly low. What went wrong?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    If a gene you *know* is spliced comes back with only weak peaks (e.g. all scores below ~0.3),
    the prediction is almost always missing one of three things. In order of how often they bite:

    1. **Wrong strand.** OpenSpliceAI scores the sequence on the strand you give it. If your gene is
       on the **minus strand** but you extracted the plus strand, the splice motifs are reverse-complemented
       and the model sees essentially the antisense sequence, so scores collapse. **Fix:** pass a GFF with
       ``-a/--annotation`` — OpenSpliceAI then reverse-complements minus-strand genes for you — or
       reverse-complement the sequence yourself before running ``predict``.

    2. **Not enough flanking context.** The deep models need real genomic sequence on **both sides** of
       every position they score: ``flanking_size/2`` bp on each side (so **5,000 bp on each side for the
       10,000 nt model**). A short, standalone FASTA (just the gene, no flanks) is padded with ``N`` out to
       that length, and ``N`` padding suppresses splice-site scores. **Fix:** give the model real context —
       run on the **whole chromosome** with ``-a`` (recommended), or extract your gene **± 5,000 bp**.
       Since v0.0.7, ``predict`` (a) prints a ``[WARN]`` when an input sequence is shorter than the required
       context, and (b) with ``-a`` automatically includes ``flanking_size/2`` bp of real flanking around
       every gene (configurable with ``--gene-flank``).

    3. **Wrong locus / gene identity.** Double-check the coordinates and assembly. Paralogs and pseudogenes
       are easy to grab by accident (for example, in mouse mm39 the housekeeping **Gapdh** is on
       ``chr6``, while ``chr7`` carries the testis-specific paralog **Gapdhs** on the minus strand).

    **Recommended command for a single gene of interest:**

    .. code-block:: bash

        # score every gene in genomic context (handles strand + flanking automatically)
        openspliceai predict \
            -m models/openspliceai-mouse/10000nt/ \   # a directory => 5-model ensemble
            -f 10000 \
            -i genome.fa \                            # whole chromosome / genome FASTA
            -a annotation.gff \                       # gene coordinates (handles strand)
            -o results/ -t 0.5

    Genuine splice sites in a well-annotated gene should score close to **1.0** once strand and flanking
    are correct.

|

.. dropdown:: Q: How do you evaluate the OpenSpliceAI prediction?
    :animate: fade-in-slide-down
    :container: + shadow
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    Evaluation of OpenSpliceAI predictions involves several complementary metrics:
    
    - **Top-k Accuracy:**  
      This metric compares the top k predicted splice sites (for both donor and acceptor channels) with the ground truth annotations. For example, Top-1 accuracy considers the highest-scoring prediction, while Top-2 accuracy doubles the number of sites evaluated.
    
    - **Standard Classification Metrics:**  
      Accuracy, precision, recall, and F1-score are calculated based on the predicted class for each nucleotide position (donor, acceptor, or non-splice).
    
    - **Area Under the Precision-Recall Curve (AUPRC):**  
      AUPRC provides a summary of prediction quality, especially useful when dealing with imbalanced data where splice sites are rare.
    
    - **Calibration Metrics:**  
      Negative log-likelihood (NLL) and expected calibration error (ECE) assess how well the predicted probabilities match the observed outcomes.
    
    - **In Silico Mutagenesis (ISM) Studies:**  
      By systematically mutating positions in a sequence and observing changes in predicted splice site strength, ISM analyses provide insights into the model’s sensitivity and the biological relevance of its predictions.
    
    These evaluation methods, combined with benchmarking against SpliceAI-Keras, demonstrate that OpenSpliceAI not only achieves high predictive accuracy but also offers reliable and well-calibrated probability estimates.


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