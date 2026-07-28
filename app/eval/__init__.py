"""Evaluation harnesses. F15: the RQ5 failsafe suite (§11.4). F20: the full §11 evaluation
framework (four pillars + RQ verdicts) that orchestrates the read-only pillar surfaces."""

from app.eval.evaluate import EvaluationReport, evaluate_classifier, run_full_evaluation
from app.eval.failsafe import FailsafeReport, FailsafeTestResult, run_failsafe_suite

__all__ = [
    "EvaluationReport",
    "FailsafeReport",
    "FailsafeTestResult",
    "evaluate_classifier",
    "run_full_evaluation",
    "run_failsafe_suite",
]
