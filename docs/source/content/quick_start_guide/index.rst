.. _quick-start_home:

Quick start
===========

Choose a task below. Examples use placeholder paths to your assembly, annotation,
models and datasets; the CLI reference lists every argument.

.. toctree::
   :maxdepth: 1

   quickstart_variant
   quickstart_predict
   quickstart_create-data
   quickstart_train
   quickstart_transfer
   quickstart_calibrate

Run all six workflows on small synthetic inputs, without downloading a genome:

.. code-block:: bash

   python examples/tutorial/run.py --output-dir /tmp/osai-tutorial

The driver runs the real CLI and checks output schemas and calibrated-model reuse.
Its model is synthetic and is intended to verify workflow mechanics, not accuracy.
