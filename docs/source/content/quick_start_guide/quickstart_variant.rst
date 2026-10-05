.. _quick-start_variant:

Quick start: variant
====================

Use :doc:`../openspliceai_variant` for input requirements and complete options.

.. code-block:: bash

   openspliceai variant -R genome.fa -A annotation.tsv -I variants.vcf -O scored.vcf.gz --model model_80nt.pt --model-type pytorch -f 80 --batch-size 8 --precision 6

Replace every input path with a matching file. Model context and assembly must agree.
For a fully runnable small example, use ``python examples/tutorial/run.py``
from a repository checkout. See :doc:`../output_explanation` to inspect outputs.
