
.. raw:: html

    <script type="text/javascript">

        let mutation_lvl_1_fuc = function(mutations) {
            var dark = document.body.dataset.theme == 'dark';

            if (document.body.dataset.theme == 'auto') {
                dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            }
            
            document.getElementsByClassName('sidebar_ccb')[0].src = dark ? '../../_static/JHU_ccb-white.png' : "../../_static/JHU_ccb-dark.png";
            document.getElementsByClassName('sidebar_wse')[0].src = dark ? '../../_static/JHU_wse-white.png' : "../../_static/JHU_wse-dark.png";



            for (let i=0; i < document.getElementsByClassName('summary-title').length; i++) {
                console.log(">> document.getElementsByClassName('summary-title')[i]: ", document.getElementsByClassName('summary-title')[i]);

                if (dark) {
                    document.getElementsByClassName('summary-title')[i].classList = "summary-title card-header bg-dark font-weight-bolder";
                    document.getElementsByClassName('summary-content')[i].classList = "summary-content card-body bg-dark text-left docutils";
                } else {
                    document.getElementsByClassName('summary-title')[i].classList = "summary-title card-header bg-light font-weight-bolder";
                    document.getElementsByClassName('summary-content')[i].classList = "summary-content card-body bg-light text-left docutils";
                }
            }

        }
        document.addEventListener("DOMContentLoaded", mutation_lvl_1_fuc);
        var observer = new MutationObserver(mutation_lvl_1_fuc)
        observer.observe(document.body, {attributes: true, attributeFilter: ['data-theme']});
        console.log(document.body);
    </script>

|

.. _quick-start_predict:

Quick Start Guide: ``predict``
==============================


This guide provides a brief walkthrough for using the ``predict`` subcommand in OpenSpliceAI to **predict splice sites** from DNA sequences. If you haven't done so already, please see the :ref:`Installation` page for details on installing and configuring OpenSpliceAI.

|

.. |download_icon| raw:: html

   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

   <i class="fa fa-download"></i>


Before You Begin
----------------

- **Install OpenSpliceAI**: Ensure you have installed OpenSpliceAI and its dependencies as described in the :ref:`Installation` page.
- **Check Example Scripts**: We provide an example script `examples/predict/predict_cmd.sh <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/predict/predict_cmd.sh>`_ |download_icon|

|


One-liner Start
-----------------

OpenSpliceAI can predict splice sites from DNA sequences with a default **10,000 nt** flanking region. You need:

1. **A reference genome (FASTA)** : `chr22.fa <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/data/chr22.fa>`_ |download_icon|

2. **A reference annotation (GTF)** : `chr22.gff <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/data/chr22.gff>`_ |download_icon|

  
3. **A pre-trained OpenSpliceAI model or directory of models**: 
    - `GitHub (models/openspliceai-mane/10000nt/) <https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane/10000nt/>`_ |download_icon| or
    -  `FTP site (OSAI-MANE/10000nt/) <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/OSAI-MANE/10000nt/>`_ |download_icon|

Run the following commands (adapt or replace filenames as needed):

.. code-block:: python

    openspliceai predict \
        -m 10000nt/ \
        -o results \
        -f 10000 \
        -i data/chr22.fa \
        -a data/chr22.gff \
        -t 0.9

This command will generate prediction results in the specified output directory (``results/``). The predictions will be based on the input FASTA file (``chr22.fa``) and the annotation file (``chr22.gff``). The results will include a GFF file with predicted splice sites and their scores.

.. Try OpenSpliceAI on Google Colab
.. --------------------------------

.. We have created a reproducible Google Colab notebook to demonstrate OpenSpliceAI in a user-friendly environment:

.. .. image:: https://colab.research.google.com/assets/colab-badge.svg
..    :target: https://colab.research.google.com/github/Kuanhao-Chao/LiftOn/blob/main/notebook/lifton_example.ipynb

.. Click the badge above to open the notebook and run OpenSpliceAI interactively.

|

Next Steps
-----------------

- **Explore ``predict`` Options:**  
  Dive into the :ref:`predict_subcommand` documentation to learn more about the available options for predicting splice sites.

- **Begin Variant Prediction:**
  Check out the :ref:`quick-start_variant` guide to predict the impact of genomic variants on splice sites.


We hope this quick start guide helps you get up and running with OpenSpliceAI. Happy predicting!

|
|
|
|
|


.. image:: ../../_images/jhu-logo-dark.png
   :alt: My Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ../../_images/jhu-logo-white.png
   :alt: My Logo
   :class: logo, header-image only-dark
   :align: center