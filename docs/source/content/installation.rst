|


.. _installation:

Installation
============

Overview
--------
There are three ways to install OpenSpliceAI: via pip, through conda, or from source. OpenSpliceAI requires Python 3.9 or higher and depends on several third‐party packages, including:

- **PyTorch** – used for deep learning model training and inference. See the `PyTorch website <https://pytorch.org/>`_ for more details.
- **mappy** – provides Python bindings for minimap2, enabling fast genomic alignments. Visit its page on `PyPI <https://pypi.org/project/mappy/>`_ for further information.

|

Prerequisites
-------------
- **Python:** Version 3.9 or higher.
- Ensure that your system has the necessary compilers.
- For GPU acceleration (when using PyTorch), install the required NVIDIA drivers and CUDA toolkit:

  - **NVIDIA GPU Drivers:**  
    Visit the `official NVIDIA Driver Downloads page <https://www.nvidia.com/en-us/drivers/>`_:  
    

  - **CUDA Toolkit:**  
    Download the latest CUDA Toolkit from the `official CUDA downloads page 
    <https://developer.nvidia.com/cuda-downloads>`_.
    For detailed installation instructions, please refer to the `CUDA Installation Guide <https://docs.nvidia.com/cuda/index.html>`_.


|

Installation Methods
--------------------

|

Install through pip
~~~~~~~~~~~~~~~~~~~~~
OpenSpliceAI is available on `PyPI <https://pypi.org/project/OpenSpliceAI/>`_. Pip automatically resolves and installs all required dependencies.

.. code-block:: bash

   pip install openspliceai

This command installs third‐party libraries including:

.. admonition:: Software dependency

   * python >= 3.9.0
   * h5py >= 3.9.0
   * numpy >= 1.24.4
   * gffutils >= 0.12
   * pysam >= 0.22.0
   * pandas >= 1.5.3
   * pyfaidx >= 0.8.1.1
   * tqdm >= 4.65.2
   * torch >= 2.2.1
   * scikit-learn >= 1.4.1.post1
   * biopython >= 1.83
   * matplotlib >= 3.8.3
   * psutil >= 5.9.2
   * mappy >= 2.28


.. admonition:: Version Warning
   :class: important

   If your numpy version is >= 1.25.0, it requires Python >= 3.9. For further guidance, please refer to the scientific python ecosystem coordination guideline `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_.

|

Install through conda
~~~~~~~~~~~~~~~~~~~~~
Installing via conda is the easiest way to set up a sandboxed environment with all dependencies.
OpenSpliceAI is distributed on the `Bioconda <https://bioconda.github.io/>`_ channel, so make sure
both the ``conda-forge`` and ``bioconda`` channels are enabled:

.. code-block:: bash

   conda install -c conda-forge -c bioconda openspliceai

This command installs OpenSpliceAI and its dependencies, including PyTorch and mappy. To install or update these packages individually, you can use:

**For PyTorch:**

.. code-block:: bash

   # CPU-only version (Conda packages are no longer available):
   pip3 install torch --index-url https://download.pytorch.org/whl/cpu

   # For GPU support (Conda packages are no longer available):
   pip3 install torch --index-url https://download.pytorch.org/whl/cu126

**For mappy:**

.. code-block:: bash

   conda install -c bioconda mappy

|

Install from source
~~~~~~~~~~~~~~~~~~~~
Alternatively, install OpenSpliceAI from source by cloning the GitHub repository:

.. code-block:: bash

   git clone https://github.com/Kuanhao-Chao/OpenSpliceAI.git
   cd OpenSpliceAI
   python setup.py install

|

Detailed Installation for PyTorch and mappy
--------------------------------------------

**PyTorch:**

- **Recommended Version:** 2.2.1 or later.
- **Usage:** Essential for model training and inference in OpenSpliceAI.
- **Installation Tips:**
  
  - For GPU acceleration, ensure your NVIDIA drivers and CUDA toolkit are installed.
  - Visit the `PyTorch official site <https://pytorch.org/get-started/locally/>`_ to select the appropriate command for your operating system.

**mappy:**

- **Recommended Version:** 2.28.
- **Usage:** Provides Python bindings for minimap2 for rapid genomic alignments.
- **Installation Tips:**

  - To install via pip:

    .. code-block:: bash

       pip install mappy

  - Or via conda:

    .. code-block:: bash

       conda install -c bioconda mappy

  - For advanced usage (e.g., multithreading), refer to the `mappy GitHub repository <https://github.com/lh3/minimap2/tree/master/python>`_ or the `Bioconda mappy recipe <https://anaconda.org/bioconda/mappy>`_.

|

Check OpenSpliceAI Installation
-------------------------------
After installing, verify that OpenSpliceAI is properly set up by running:

.. code-block:: bash

   openspliceai -h

You should see the usage information and version details printed in your terminal.

|

Terminal Output Example
-------------------------
.. dropdown:: Terminal output
   :animate: fade-in-slide-down
   :title: bg-light font-weight-bolder
   :body: bg-light text-left

   .. code-block::


      ============================================================
      Deep learning framework that decodes splicing across species
      ============================================================


      ██████╗ ██████╗ ███████╗███╗   ██╗███████╗██████╗ ██╗     ██╗ ██████╗███████╗ █████╗ ██╗
      ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██╔══██╗██║     ██║██╔════╝██╔════╝██╔══██╗██║
      ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████╗██████╔╝██║     ██║██║     █████╗  ███████║██║
      ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════██║██╔═══╝ ██║     ██║██║     ██╔══╝  ██╔══██║██║
      ╚██████╔╝██║     ███████╗██║ ╚████║███████║██║     ███████╗██║╚██████╗███████╗██║  ██║██║
      ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚══════╝╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝

      0.0.7

      usage: openspliceai [-h] {create-data,train,calibrate,transfer,predict,variant} ...

      OpenSpliceAI toolkit to help you retrain your own splice site predictor

      positional arguments:
      {create-data,train,calibrate,transfer,predict,variant}
                              Subcommands: create-data, train, calibrate, predict, transfer, variant
         create-data         Create dataset for your genome for SpliceAI model training
         train               Train the SpliceAI model
         calibrate           Calibrate the SpliceAI model
         transfer            transfer a pre-trained SpliceAI model on new data.
         predict             Predict splice sites in a given sequence using the SpliceAI model
         variant             Label genetic variations with their predicted effects on splicing.

      optional arguments:
      -h, --help            show this help message and exit

|

Next Steps
-----------------
Once installation is complete, please proceed to the :ref:`quick-start_home` to begin working with OpenSpliceAI for data creation, model training, prediction, calibration, and variant analysis.


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