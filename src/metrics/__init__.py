"""
Metrics for DSPy optimization.

This module contains LLM-as-judge metrics for evaluating:
- Entailment validation quality (SUPPORTS vs RELATED distinction)
"""

from .entailment_metrics import entailment_llm_judge_metric, EntailmentQualityJudge

__all__ = ["entailment_llm_judge_metric", "EntailmentQualityJudge"]
