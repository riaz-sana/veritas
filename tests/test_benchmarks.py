"""Tests for benchmark harness."""
import pytest
from veritas.benchmarks.metrics import accuracy, expected_calibration_error
from veritas.core.result import Verdict

def test_accuracy_all_correct():
    predictions = [Verdict.VERIFIED, Verdict.REFUTED, Verdict.PARTIAL]
    labels = [Verdict.VERIFIED, Verdict.REFUTED, Verdict.PARTIAL]
    assert accuracy(predictions, labels) == 1.0

def test_accuracy_none_correct():
    predictions = [Verdict.VERIFIED, Verdict.VERIFIED]
    labels = [Verdict.REFUTED, Verdict.REFUTED]
    assert accuracy(predictions, labels) == 0.0

def test_accuracy_partial():
    predictions = [Verdict.VERIFIED, Verdict.REFUTED, Verdict.VERIFIED]
    labels = [Verdict.VERIFIED, Verdict.REFUTED, Verdict.REFUTED]
    assert abs(accuracy(predictions, labels) - 2 / 3) < 0.01

def test_accuracy_empty():
    assert accuracy([], []) == 0.0

def test_ece_perfect_calibration():
    confidences = [0.9, 0.9, 0.1, 0.1]
    correct = [True, True, False, False]
    ece = expected_calibration_error(confidences, correct, n_bins=2)
    assert ece < 0.1

def test_ece_worst_calibration():
    confidences = [0.9, 0.9, 0.9, 0.9]
    correct = [False, False, False, False]
    ece = expected_calibration_error(confidences, correct, n_bins=1)
    assert ece > 0.8
