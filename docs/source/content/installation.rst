.. _installation:

Installation
============

Use a separate environment. The package requires Python 3.9 or newer and a
scientific stack compatible with NumPy 2. Install a PyTorch wheel appropriate to
your machine first; check its NumPy ABI and GPU runtime before long jobs.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install openspliceai
   python -m pip check
   openspliceai --help

For development, clone the repository and install its current checkout:

.. code-block:: bash

   git clone https://github.com/Kuanhao-Chao/OpenSpliceAI.git
   cd OpenSpliceAI
   python -m pip install -e '.[dev]'
   python -m pip check
   openspliceai variant --help

The development audit branch is ``audit/comprehensive-20261004``. PyPI and Bioconda
releases may have different defaults; a development checkout does not change the
published release. Bioconda users can install in a separate environment with
``conda create -n openspliceai -c conda-forge -c bioconda openspliceai``.

CPU and GPU selection
---------------------

PyTorch commands select CUDA if available, then available macOS MPS, otherwise CPU.
Set ``CUDA_VISIBLE_DEVICES=''`` before starting a command to force CPU on Linux.
Bound thread counts with ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and
``OPENBLAS_NUM_THREADS``. The historical ``N_GPUS`` configuration field is not a
request for multiple GPUs and does not discard partial batches.

.. code-block:: bash

   python -c "import torch, numpy; x=torch.ones(3); print(torch.__version__, numpy.__version__, x.numpy(), torch.cuda.is_available())"

Optional original SpliceAI backend
----------------------------------

PyTorch workflows do not require TensorFlow. Original Keras ``.h5`` models require
a compatible TensorFlow/legacy-Keras environment and weights installed separately.
The reference profile uses Python 3.11, NumPy 2.0.2, TensorFlow 2.18.0,
``tf-keras==2.18.0`` and ``spliceai==1.3.1``. Set
``TF_USE_LEGACY_KERAS=1`` before importing TensorFlow. Keep this profile separate
from a production environment. See :doc:`development` for executed and pending
compatibility evidence, and :doc:`pretrained_models/index` for model downloads.

Native dependencies
-------------------

``mappy`` provides minimap2 for paralog removal; a C/C++ compiler may be needed if
no wheel exists. h5py, pysam, SciPy, scikit-learn and Matplotlib need versions that
support the installed NumPy ABI. Use ``python -m pip check`` and the import probe
above to diagnose an environment before running scientific workflows.
