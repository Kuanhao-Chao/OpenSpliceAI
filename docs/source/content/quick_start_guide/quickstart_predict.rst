.. _quick-start_predict:

Quick start: predict
====================

Use :doc:`../openspliceai_predict` for input requirements and complete options.

.. code-block:: bash

   openspliceai predict --input-sequence genome.fa --model model_80nt.pt -f 80 --output-dir predictions

Replace every input path with a matching file. Model context and assembly must agree.
For a fully runnable small example, use ``python examples/tutorial/run.py``
from a repository checkout. See :doc:`../output_explanation` to inspect outputs.
