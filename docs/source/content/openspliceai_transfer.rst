
|

.. _transfer_subcommand:

transfer
========

The ``transfer`` subcommand leverages **transfer learning** to adapt an existing (pre-trained) OpenSpliceAI model to a new species or dataset. Rather than training from scratch, it loads pre-trained weights (e.g., from a human-trained model) and fine-tunes them on the user’s target data. This approach often yields faster convergence and improved accuracy compared to full retraining, especially when limited data are available.

|

Subcommand Description
----------------------

Similar to the :ref:`train_subcommand`, the ``transfer`` subcommand relies on:

- **SpliceAI Architecture**: The same deep convolutional residual network used in standard training.
- **Adaptive Learning Rate Scheduling**: MultiStepLR or CosineAnnealingWarmRestarts.
- **Early Stopping & Logging**: Optional patience-based early stopping and comprehensive logging.

However, it **initializes** the model with weights from a specified checkpoint (``--pretrained-model``), enabling partial or full unfreezing of layers:

- **``--unfreeze-all``**: Unfreeze all layers for full fine-tuning.
- **``--unfreeze <n>``**: Unfreeze the last *n* layers, leaving the rest frozen.

|

Input Files
-----------

1. **Training HDF5 File**

   As with the ``train`` subcommand, an HDF5 file containing the training data (one-hot-encoded sequences and labels).

2. **Testing HDF5 File**

   An HDF5 file for held-out testing. The data format should match the one produced by the :ref:`create-data_subcommand`.

3. **Pre-trained Model (PT File)**

   A PyTorch checkpoint file (``.pt``) containing the model weights you wish to fine-tune. For example, a human-trained model like **OpenSpliceAI-MANE**.

|

Output Files
------------

1. **Fine-Tuned Model (PT File)**

   The main output is a fine-tuned model checkpoint. It follows the same naming convention as the ``train`` subcommand (e.g., ``model_<epoch>.pt`` and ``model_best.pt`` for the best validation loss).

2. **Logs and Metrics**

   - **Loss** and **learning rate** logs.
   - **Performance** metrics (accuracy, precision, recall, F1-score, AUPRC, top-k accuracy).
   - **Checkpoint** and best-model tracking.

|

Usage
-----

.. code-block:: text

   usage: openspliceai transfer [-h] [--epochs EPOCHS] [--scheduler {MultiStepLR,CosineAnnealingWarmRestarts}] [--early-stopping] [--patience PATIENCE] --output-dir
                              OUTPUT_DIR --project-name PROJECT_NAME [--exp-num EXP_NUM] [--flanking-size {80,400,2000,10000}] [--random-seed RANDOM_SEED]
                              --pretrained-model PRETRAINED_MODEL --train-dataset TRAIN_DATASET --test-dataset TEST_DATASET [--loss {cross_entropy_loss,focal_loss}]
                              [--unfreeze-all] [--unfreeze UNFREEZE]

   optional arguments:
         -h, --help            show this help message and exit
         --epochs EPOCHS, -n EPOCHS
                                 Number of epochs for training
         --scheduler {MultiStepLR,CosineAnnealingWarmRestarts}, -s {MultiStepLR,CosineAnnealingWarmRestarts}
                                 Learning rate scheduler
         --early-stopping, -E  Enable early stopping
         --patience PATIENCE, -P PATIENCE
                                 Number of epochs to wait before early stopping
         --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                                 Output directory for model checkpoints and logs
         --project-name PROJECT_NAME, -p PROJECT_NAME
                                 Project name for the fine-tuning experiment
         --exp-num EXP_NUM, -e EXP_NUM
                                 Experiment number
         --flanking-size {80,400,2000,10000}, -f {80,400,2000,10000}
                                 Flanking sequence size
         --random-seed RANDOM_SEED, -r RANDOM_SEED
                                 Random seed for reproducibility
         --pretrained-model PRETRAINED_MODEL, -m PRETRAINED_MODEL
                                 Path to the pre-trained model
         --train-dataset TRAIN_DATASET, -train TRAIN_DATASET
                                 Path to the training dataset
         --test-dataset TEST_DATASET, -test TEST_DATASET
                                 Path to the testing dataset
         --loss {cross_entropy_loss,focal_loss}, -l {cross_entropy_loss,focal_loss}
                                 Loss function for fine-tuning
         --unfreeze-all, -A    Unfreeze all layers for fine-tuning
         --unfreeze UNFREEZE, -u UNFREEZE
                                 Number of layers to unfreeze for fine-tuning

|

Examples
--------

Example: Transfer Learning from Human Model to a New Species
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you have a **human-trained** model checkpoint called ``OpenSpliceAI-MANE-best.pt``. You can adapt it to a new species using:

.. code-block:: bash

   openspliceai transfer \
      --train-dataset dataset_train.h5 \
      --test-dataset dataset_test.h5 \
      --pretrained-model OpenSpliceAI-MANE-best.pt \
      --flanking-size 400 \
      --unfreeze 2 \
      --scheduler CosineAnnealingWarmRestarts \
      --loss focal_loss \
      --epochs 15 \
      --patience 3 \
      --early-stopping \
      --project-name new_species_transfer \
      --output-dir ./transfer_outputs/

After running, the tool:

- Loads and partially unfreezes the last 2 residual blocks of the model.
- Fine-tunes on the new species training data.
- Evaluates on the test set after each epoch.
- Saves model checkpoints (e.g., ``model_0.pt``, ``model_best.pt``) and logs.

|

Processing Pipeline
-------------------

The ``transfer`` pipeline closely mirrors the :ref:`train_subcommand`:

1. **Model Initialization**  
   - Creates a SpliceAI architecture for the specified flanking size.
   - Loads weights from the user-specified pretrained checkpoint (``--pretrained-model``).
   - Freezes or unfreezes layers based on the ``--unfreeze`` or ``--unfreeze-all`` arguments.

2. **Adaptive Learning Rate & Loss Function**  
   - Continues using **AdamW** as the optimizer, with an initial LR of 1e-4.
   - Users can choose among **MultiStepLR** or **CosineAnnealingWarmRestarts** schedulers.
   - Either **cross_entropy_loss** or **focal_loss** can be selected for training.

3. **Data Splitting**  
   - The training HDF5 is split 90:10 into training and validation sets.  
   - The test HDF5 is used for final model evaluation.

4. **Fine-Tuning & Early Stopping**  
   - Runs for up to the specified number of epochs (e.g., 15).  
   - If early stopping is enabled, training halts once validation loss fails to improve for ``--patience`` epochs.

5. **Logging & Checkpoints**  
   - Logs learning rate, loss, accuracy, precision, recall, F1, top-k accuracy, and AUPRC.  
   - Saves model checkpoints each epoch; the best model is stored as ``model_best.pt``.

|

Conclusion
----------

The ``transfer`` subcommand provides an efficient path to adapt a pre-trained OpenSpliceAI model (e.g., from human data) to a new dataset or species. By freezing or unfreezing layers, you can control how much of the original model is retained versus retrained, achieving faster convergence and often higher accuracy. Refer to the command-line usage for all available options, and see the official documentation for advanced transfer learning strategies.

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