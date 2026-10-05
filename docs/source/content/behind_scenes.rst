.. _behind-the-scenes-splam:

How the model and workflows work
================================

A/C/G/T channels enter a dilated residual convolutional network with skip
connections. Context schedules span 80, 400, 2000 or 10000 bases total, half on
each side. Cropping removes context and a three-class softmax returns non-splice,
acceptor and donor probabilities. Ensembles average model probabilities.

Training datasets retain 5000 output positions and 10000 stored context bases.
Smaller architectures crop only input context. Training uses observed labels;
transfer can additionally compare student probabilities to frozen teacher targets.
Temperature calibration rescales class logits before softmax. Variant scoring
compares reference and alternate profiles and reports the largest changes and
positions, with gene-boundary padding and strand-aware realignment.

See :doc:`repository` for source flow, :doc:`function_manual` for definitions,
and the six workflow pages for file formats, controls and edge cases.
