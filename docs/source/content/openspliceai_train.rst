
|


.. _train_subcommand:

train
=====

The ``train`` subcommand takes the HDF5 datasets produced by the :ref:`create-data_subcommand` and trains a deep learning model (SpliceAI-PyTorch) to predict splice sites. It allows flexible configuration of model hyperparameters, learning rate schedulers, and loss functions, and supports early stopping. 

|

Subcommand Description
----------------------

After successfully creating training and testing HDF5 files, use the ``train`` subcommand to:

- **Read** the training dataset (used for model training) and a held-out testing dataset (for model evaluation).
- **Build** a SpliceAI-based neural network architecture (convolutional residual units).
- **Optimize** the network using AdamW, an adaptive learning rate scheduler, and an optional early stopping mechanism.
- **Generate** log files and model checkpoints (``.pt``) for further analysis or transfer learning.

|

Input Files
-----------

1. **Training HDF5 File**

   The dataset used for model training. It contains one-hot-encoded gene sequences (:math:`X`) and splice site labels (:math:`Y`).

2. **Testing HDF5 File**

   The dataset used for final model testing. It also contains one-hot-encoded sequences and labels.

|

Output Files
------------

1. **Trained Model (PT File)**

   The primary output of the ``train`` subcommand is a saved PyTorch model checkpoint (``model_<epoch>.pt``).  
   A ``model_best.pt`` is also saved whenever the validation loss improves, serving as the best-performing checkpoint.

2. **Training and Testing Logs**

   The subcommand creates log files and directories containing:
   
   - **Loss Values** (per batch/epoch)
   - **Learning Rates** (per epoch and per batch)
   - **Performance Metrics** (accuracy, precision, recall, F1-score, top-k accuracy, and AUPRC for donor and acceptor sites)

|

Usage
-----

.. code-block:: text

   usage: openspliceai train [-h] [--epochs EPOCHS] [--scheduler {MultiStepLR,CosineAnnealingWarmRestarts}] [--early-stopping] [--patience PATIENCE]
                           --output-dir OUTPUT_DIR --project-name PROJECT_NAME [--exp-num EXP_NUM] [--flanking-size {80,400,2000,10000}]
                           [--random-seed RANDOM_SEED] --train-dataset TRAIN_DATASET --test-dataset TEST_DATASET
                           [--loss {cross_entropy_loss,focal_loss}] [--model MODEL]

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
                                 Output directory to save the data
         --project-name PROJECT_NAME, -p PROJECT_NAME
                                 Project name for the train experiment
         --exp-num EXP_NUM, -e EXP_NUM
                                 Experiment number
         --flanking-size {80,400,2000,10000}, -f {80,400,2000,10000}
                                 Flanking sequence size
         --random-seed RANDOM_SEED, -r RANDOM_SEED
                                 Random seed for reproducibility
         --train-dataset TRAIN_DATASET, -train TRAIN_DATASET
                                 Path to the training dataset
         --test-dataset TEST_DATASET, -test TEST_DATASET
                                 Path to the testing dataset
         --loss {cross_entropy_loss,focal_loss}, -l {cross_entropy_loss,focal_loss}
                                 Loss function for training
         --model MODEL, -m MODEL
      
|

Examples
--------

Example: Training a Model on Human MANE Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a sample command for training a SpliceAI-PyTorch model on human MANE datasets.  
These HDF5 files can be generated via the :ref:`create-data_subcommand` or downloaded from the provided links.

.. code-block:: bash

   openspliceai train \
      --flanking-size 10000 \
      --exp-num full_dataset \
      --train-dataset /path/to/dataset_train.h5 \
      --test-dataset  /path/to/dataset_test.h5 \
      --output-dir /path/to/model_train_outdir/ \
      --project-name human_MANE_adaptive_lr \
      --random-seed 22 \
      --model SpliceAI \
      --loss cross_entropy_loss \
      --epochs 20 \
      --patience 5 \
      --scheduler ReduceLROnPlateau \
      --early-stopping \
      -d

After running the above command, you will obtain:

- **Model Checkpoints**: e.g., ``model_0.pt``, ``model_1.pt``, …, plus ``model_best.pt`` for the best validation loss.
- **Log Files**: containing training/testing metrics, learning rates, and loss curves.

|

Processing Steps
----------------

1. **Model Architecture**

   By default, OpenSpliceAI trains a convolutional residual architecture inspired by SpliceAI. The flanking sequence size (e.g., 80, 400, 2,000, or 10,000) determines the depth and dilation rates of the convolutional layers.

2. **Dataset Split**

   The training dataset is internally split into 90% for training and 10% for validation. The separate testing dataset is used for final model evaluation.

3. **Optimization and Learning Rate Scheduling**

   - **Optimizer**: AdamW with a default initial learning rate of 1e-3.
   - **Scheduler** (user-configurable):
     - **MultiStepLR** (default): reduces LR by 0.5 at specified epochs (e.g., epoch 6, 7, 8…).
     - **CosineAnnealingWarmRestarts**: smoothly reduces LR in cycles, returning to the initial LR after each cycle.
     - **ReduceLROnPlateau**: reduces LR by a factor (e.g., 0.5) if validation loss does not improve after a certain patience period.

4. **Training**

   - The model typically runs for up to 10 or 20 epochs (user-configurable).
   - After each epoch, validation metrics (loss, accuracy, etc.) are computed on the held-out 10% of the training data.
   - The final model is evaluated on the test set.

5. **Early Stopping**

   If ``--early-stopping`` is enabled, training halts once validation loss fails to improve for a specified number of epochs (``--patience``). This prevents overfitting and reduces unnecessary computation.

6. **Logging and Model Saving**

   - All training and testing metrics are saved in dedicated log files.
   - Model checkpoints (``.pt``) are saved every epoch, with ``model_best.pt`` reserved for the best validation performance.

|

Conclusion
----------

The ``train`` subcommand provides a comprehensive workflow to develop and optimize a deep learning model for splice site prediction. By leveraging user-defined flanking sizes, flexible loss functions, and advanced learning rate schedulers, you can tailor the training process to your specific dataset. Refer to the command-line usage for further customization options, and see the official documentation for advanced topics such as transfer learning.

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
