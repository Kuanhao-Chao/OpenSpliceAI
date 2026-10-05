.. _openspliceai-vs-spliceai:

OpenSpliceAI and original SpliceAI
==================================

OpenSpliceAI provides a PyTorch implementation and training, transfer, calibration,
FASTA prediction and variant workflows. Its four architecture schedules and
probability-ensemble convention retain the underlying SpliceAI design. Released
OpenSpliceAI weights trained on different annotations/species are distinct models,
so their biological predictions are not expected to equal original SpliceAI weights.

The optional original Keras backend is compared with SpliceAI 1.3.1 on their shared
supported SNV/indel inputs. OpenSpliceAI additionally scores supported MNV/delins
and rejects REF spans beyond distance+1; the original tool returns placeholders
for some extension inputs. Shared-allele parity does not imply extension parity,
all-platform numerical identity or biological accuracy. Current executed backend
results are listed in the audit evidence, separately from skipped tests.
