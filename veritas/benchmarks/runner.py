"""Benchmark runner for evaluating Veritas against standard datasets."""
from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field
from veritas.benchmarks.datasets import BenchmarkItem
from veritas.benchmarks.metrics import accuracy, expected_calibration_error
from veritas.core.config import Config
from veritas.core.result import Verdict, VerificationResult
from veritas.core.verify import verify

@dataclass
class BenchmarkResult:
    dataset: str
    total: int
    accuracy: float
    ece: float
    duration_seconds: float
    results: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"dataset": self.dataset, "total": self.total, "accuracy": round(self.accuracy, 4), "ece": round(self.ece, 4), "duration_seconds": round(self.duration_seconds, 2), "results": self.results}, indent=2)

async def run_benchmark(items: list[BenchmarkItem], dataset_name: str = "custom", config: Config | None = None) -> BenchmarkResult:
    start = time.monotonic()
    predictions, labels, confidences, correct, per_item = [], [], [], [], []
    for item in items:
        try:
            result = await verify(claim=item.claim, domain=item.domain, config=config)
            pred = result.verdict
            expected = Verdict(item.expected_verdict)
            predictions.append(pred)
            labels.append(expected)
            confidences.append(result.confidence)
            correct.append(pred == expected)
            per_item.append({"claim": item.claim, "expected": item.expected_verdict, "predicted": pred.value, "confidence": result.confidence, "correct": pred == expected, "summary": result.summary})
        except Exception as e:
            per_item.append({"claim": item.claim, "expected": item.expected_verdict, "error": str(e)})
    duration = time.monotonic() - start
    return BenchmarkResult(dataset=dataset_name, total=len(items), accuracy=accuracy(predictions, labels), ece=expected_calibration_error(confidences, correct) if confidences else 0.0, duration_seconds=duration, results=per_item)
