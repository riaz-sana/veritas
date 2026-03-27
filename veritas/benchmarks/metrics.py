"""Benchmark metrics for verification evaluation."""
from __future__ import annotations
from veritas.core.result import Verdict

def accuracy(predictions: list[Verdict], labels: list[Verdict]) -> float:
    if not predictions:
        return 0.0
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)

def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        indices = [j for j, c in enumerate(confidences) if (lo <= c < hi) or (i == n_bins - 1 and c == hi)]
        if not indices:
            continue
        bin_conf = sum(confidences[j] for j in indices) / len(indices)
        bin_acc = sum(1 for j in indices if correct[j]) / len(indices)
        ece += (len(indices) / total) * abs(bin_acc - bin_conf)
    return ece
