.. _main:

OpenSpliceAI
============

OpenSpliceAI trains, adapts and runs splice-site models in PyTorch. It provides six
commands: create datasets, train, transfer, calibrate, predict FASTA sequences and
annotate VCF variants. Released weights are available for human and nonhuman species.

Start with :doc:`content/installation` and :doc:`content/quick_start_guide/index`.
These pages describe the development version **0.1.0.dev0**. Its scientific defaults
and prediction output changed; read :doc:`content/migration` before reusing an older
training command. Released model weights and existing production campaigns are
not retrained or replaced by these software changes.

.. toctree::
   :maxdepth: 1
   :caption: Run OpenSpliceAI

   content/installation
   content/quick_start_guide/index
   content/openspliceai_create-data
   content/openspliceai_train
   content/openspliceai_transfer
   content/openspliceai_calibrate
   content/openspliceai_predict
   content/openspliceai_variant
   content/output_explanation
   content/train_your_own_model/index
   content/pretrained_models/index
   content/how_to_page

.. toctree::
   :maxdepth: 1
   :caption: Understand and develop

   content/behind_scenes
   content/openspliceai_vs_spliceai
   content/function_manual
   content/repository
   content/development
   content/migration
   content/usage/index
   content/changelog
   content/license
   content/contact

Citation
--------

Kuan-Hao Chao, Alan Mao, Anqi Liu, Mihaela Pertea and Steven L. Salzberg,
`OpenSpliceAI provides an efficient modular implementation of SpliceAI enabling
easy retraining across nonhuman species <https://doi.org/10.7554/eLife.107454.3>`_.
Also cite `Jaganathan et al., SpliceAI <https://doi.org/10.1016/j.cell.2018.12.015>`_
when using its architecture or original weights.
