"""
Metrics for DSPy optimization.

This module contains metrics for evaluating:
- Entailment validation quality (SUPPORTS vs RELATED distinction)
- Quote finder quality (verification + entailment + recall)
"""

from .entailment_metrics import entailment_llm_judge_metric, EntailmentQualityJudge
from .quote_finder_metrics import QuoteFinderMetric

__all__ = [
    "entailment_llm_judge_metric",
    "EntailmentQualityJudge",
    "QuoteFinderMetric"
]
