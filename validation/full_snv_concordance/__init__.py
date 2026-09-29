"""Streaming, mergeable analyses for full-genome SNV score comparisons.

The package deliberately treats SpliceAI as a comparator, not biological truth.
See :mod:`validation.full_snv_concordance.cli` for the command-line interface.
"""

from .aggregate import Aggregate, AnalysisConfig
from .vcf import Annotation, VariantGroup, VariantKey

__all__ = ["Aggregate", "AnalysisConfig", "Annotation", "VariantGroup", "VariantKey"]

__version__ = "1.0.0"
