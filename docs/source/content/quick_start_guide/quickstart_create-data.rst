.. _quick-start_create_data:

Quick start: create-data
========================

Use :doc:`../openspliceai_create-data` for input requirements and complete options.

.. code-block:: bash

   openspliceai create-data --genome-fasta genome.fa --annotation-gff annotation.gff3 --output-dir data --verify-h5

Replace every input path with a matching file. Model context and assembly must agree.
For a fully runnable small example, use ``python examples/tutorial/run.py``
from a repository checkout. See :doc:`../output_explanation` to inspect outputs.
