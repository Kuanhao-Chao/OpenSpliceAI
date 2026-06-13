
|


.. _calibrate_subcommand:

calibrate
=========

The ``calibrate`` subcommand performs **temperature scaling** to refine the output probabilities of an already-trained OpenSpliceAI model. By introducing a single temperature parameter :math:`T`, the subcommand ensures that the model’s predicted probabilities are more closely aligned with observed outcome likelihoods. This approach is essential for reliable decision-making in classification tasks, preventing over- or under-confident predictions.

|

Subcommand Description
----------------------

During calibration, the original model weights are **frozen** and only the temperature parameter :math:`T` is optimized. A **negative log-likelihood (NLL)** loss is used to measure the discrepancy between predicted probabilities and true labels. If the model outputs logits :math:`z`, the calibrated logits are :math:`z' = z / T`. By carefully adjusting :math:`T`, the subcommand can reduce miscalibration, as measured by the **expected calibration error (ECE)**, and ensure that probabilities reflect real-world likelihoods.

Key Features:

- **Temperature Scaling**: A post-hoc calibration method that does not alter the underlying model weights.
- **Adam + ReduceLROnPlateau**: Optimizes the temperature parameter with an initial LR of 0.01, reducing it by a factor of 0.1 upon plateaus.
- **Early Stopping**: Training halts if the calibration loss (NLL) does not improve within a specified patience (default 10 epochs).
- **ECE & Reliability Curves**: Evaluates the model’s calibration using ECE and generates reliability (calibration) curves for each class.

|

Input Files
-----------

1. **Trained Model (Pretrained Checkpoint)**

   A PyTorch checkpoint (``.pt``) file containing the trained model weights.  
   For example, a best model checkpoint from the :ref:`train_subcommand` or :ref:`transfer_subcommand`.

2. **Test (or Validation) HDF5 File**

   An HDF5 file containing one-hot-encoded sequences and labels for calibration.  
   Typically, you would use a held-out test or validation dataset that was **not** used during model training.

3. **Temperature File (Optional)**

   A saved temperature parameter (``.pt``) to load directly instead of recalculating.  
   If provided, calibration will skip re-optimization and simply apply the existing temperature value.

|

Output Files
------------

1. **Calibrated Temperature Parameter**

   A single-parameter file (``temperature.pt``) that stores the optimized temperature :math:`T`.  
   A text file (``temperature.txt``) is also created to log the final temperature value.

2. **Calibrated Probability Plots & Logs**

   - **Reliability (Calibration) Curves** for each class.
   - **Score Distributions** comparing original and calibrated probabilities.
   - **Brier Score** summaries (uncalibrated vs. calibrated).
   - **ECE** and **NLL** logs to evaluate calibration quality.

|

Usage
-----

.. code-block:: text

   usage: openspliceai calibrate [-h] [--epochs EPOCHS] [--early-stopping] [--patience PATIENCE] --output-dir OUTPUT_DIR --project-name PROJECT_NAME [--exp-num EXP_NUM]
                                 [--flanking-size {80,400,2000,10000}] [--random-seed RANDOM_SEED] [--temperature-file TEMPERATURE_FILE] --pretrained-model
                                 PRETRAINED_MODEL --train-dataset TRAIN_DATASET --test-dataset TEST_DATASET [--loss {cross_entropy_loss,focal_loss}]

   optional arguments:
         -h, --help            show this help message and exit
         --epochs EPOCHS, -n EPOCHS
                                 Number of epochs for training
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
         --temperature-file TEMPERATURE_FILE, -T TEMPERATURE_FILE
                                 Path to the temperature file
         --pretrained-model PRETRAINED_MODEL, -m PRETRAINED_MODEL
                                 Path to the pre-trained model
         --train-dataset TRAIN_DATASET, -train TRAIN_DATASET
                                 Path to the training dataset
         --test-dataset TEST_DATASET, -test TEST_DATASET
                                 Path to the testing dataset
         --loss {cross_entropy_loss,focal_loss}, -l {cross_entropy_loss,focal_loss}
                                 Loss function for fine-tuning

|

Examples
--------

Example: Calibrating a Trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you have a trained model ``model_best.pt`` and the HDF5 splits produced by
:ref:`create-data_subcommand` (``dataset_train.h5`` and ``dataset_test.h5``). Both
``--train-dataset`` and ``--test-dataset`` are required, along with ``--project-name``:

.. code-block:: bash

   openspliceai calibrate \
      --pretrained-model model_best.pt \
      --train-dataset dataset_train.h5 \
      --test-dataset dataset_test.h5 \
      --flanking-size 10000 \
      --project-name human_MANE_calibration \
      --output-dir ./calibration_results/ \
      --random-seed 42

The subcommand:

1. Loads the pre-trained model from ``model_best.pt``.
2. Freezes the model weights and initializes a temperature parameter :math:`T`.
3. Optimizes :math:`T` using negative log-likelihood (NLL) on the test dataset.
4. Logs calibration metrics (NLL, ECE) and saves a reliability curve, score distribution plots, and ``temperature.pt``.

|

Processing Steps
----------------

1. **Temperature Parameter Initialization**

   - The model architecture is loaded in inference mode.
   - A parameter :math:`T` (default 1.1) is introduced to scale logits.

2. **Collecting Logits & Labels**

   - The subcommand runs inference on the specified HDF5 test data.
   - It stores the raw logits (pre-softmax) and the ground-truth labels for each chunk.

3. **Temperature Optimization**

   - **Loss Function**: Negative log-likelihood (NLL).
   - **Optimizer**: Adam with an initial LR of 0.01.
   - **Scheduler**: ReduceLROnPlateau, reducing the LR by 0.1 if no improvement is seen after 5 epochs.
   - **Early Stopping**: Patience of 10 epochs, requiring a minimum delta (1e-6) improvement in NLL to continue.

4. **Final Calibration**

   - The optimal temperature :math:`T^*` is saved to ``temperature.pt``.
   - All calibration plots (score distributions, reliability curves) and ECE metrics are generated.

5. **Reusing a Temperature File**

   - If ``--temperature-file`` is provided, the subcommand skips re-optimization and applies the existing temperature.
   - This is useful if you want to load a previously determined calibration parameter.

|

Conclusion
----------

The ``calibrate`` subcommand is essential for ensuring well-calibrated probability outputs from an OpenSpliceAI model. By adjusting the model’s logits via a learned temperature parameter, predicted probabilities more accurately reflect real-world likelihoods, improving the model’s reliability for downstream applications. For further details on temperature scaling, ECE, and reliability curves, refer to the official documentation and the references cited in the paper.


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