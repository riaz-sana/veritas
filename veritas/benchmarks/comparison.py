"""Isolation vs debate comparison for benchmarking."""
from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field
from veritas.benchmarks.datasets import BenchmarkItem
from veritas.benchmarks.metrics import accuracy, expected_calibration_error
from veritas.core.config import Config
from veritas.core.result import Verdict, VerificationResult
from veritas.orchestration.runner import VerificationRunner
from veritas.orchestration.debate_runner import DebateRunner
from veritas.providers.claude import ClaudeProvider
from veritas.providers.search import BraveSearchProvider, TavilySearchProvider


@dataclass
class ComparisonResult:
    """Results from an isolation-vs-debate comparison."""
    dataset: str
    total: int
    isolation_accuracy: float
    debate_accuracy: float
    isolation_ece: float
    debate_ece: float
    isolation_duration: float
    debate_duration: float
    per_item: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "# Isolation vs Debate Comparison",
            "",
            f"Dataset: {self.dataset} ({self.total} items)",
            "",
            "| Metric | Isolation | Debate | Delta |",
            "|--------|-----------|--------|-------|",
            f"| Accuracy | {self.isolation_accuracy:.2%} | {self.debate_accuracy:.2%} | {self.isolation_accuracy - self.debate_accuracy:+.2%} |",
            f"| ECE | {self.isolation_ece:.4f} | {self.debate_ece:.4f} | {self.isolation_ece - self.debate_ece:+.4f} |",
            f"| Duration | {self.isolation_duration:.1f}s | {self.debate_duration:.1f}s | {self.isolation_duration - self.debate_duration:+.1f}s |",
            "",
        ]
        winner = "Isolation" if self.isolation_accuracy >= self.debate_accuracy else "Debate"
        lines.append(f"**Winner on accuracy: {winner}**")
        if self.isolation_accuracy > self.debate_accuracy:
            lines.append("")
            lines.append("Isolation-divergent verification outperforms shared-context debate,")
            lines.append("supporting the thesis that agent isolation prevents conformity bias.")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "dataset": self.dataset,
            "total": self.total,
            "isolation": {"accuracy": round(self.isolation_accuracy, 4), "ece": round(self.isolation_ece, 4), "duration_seconds": round(self.isolation_duration, 2)},
            "debate": {"accuracy": round(self.debate_accuracy, 4), "ece": round(self.debate_ece, 4), "duration_seconds": round(self.debate_duration, 2)},
            "per_item": self.per_item,
        }, indent=2)


async def run_comparison(
    items: list[BenchmarkItem],
    dataset_name: str = "custom",
    config: Config | None = None,
) -> ComparisonResult:
    """Run the same items through both isolation and debate modes."""
    if config is None:
        config = Config()
    config.validate()

    llm = ClaudeProvider(model=config.model, api_key=config.anthropic_api_key)
    if config.search_provider == "tavily":
        search = TavilySearchProvider(api_key=config.search_api_key)
    else:
        search = BraveSearchProvider(api_key=config.search_api_key)

    isolation_runner = VerificationRunner(llm_provider=llm, search_provider=search, config=config)
    debate_runner = DebateRunner(llm_provider=llm, search_provider=search, config=config)

    iso_preds, iso_labels, iso_confs, iso_correct = [], [], [], []
    deb_preds, deb_labels, deb_confs, deb_correct = [], [], [], []
    per_item = []

    iso_start = time.monotonic()
    for item in items:
        try:
            expected = Verdict(item.expected_verdict)
            result = await isolation_runner.run(claim=item.claim, context=None, domain=item.domain, references=[])
            iso_preds.append(result.verdict)
            iso_labels.append(expected)
            iso_confs.append(result.confidence)
            iso_correct.append(result.verdict == expected)
            per_item.append({"claim": item.claim, "expected": item.expected_verdict, "isolation_verdict": result.verdict.value, "isolation_confidence": result.confidence})
        except Exception as e:
            per_item.append({"claim": item.claim, "expected": item.expected_verdict, "isolation_error": str(e)})
    iso_duration = time.monotonic() - iso_start

    deb_start = time.monotonic()
    for i, item in enumerate(items):
        try:
            expected = Verdict(item.expected_verdict)
            result = await debate_runner.run(claim=item.claim, context=None, domain=item.domain, references=[])
            deb_preds.append(result.verdict)
            deb_labels.append(expected)
            deb_confs.append(result.confidence)
            deb_correct.append(result.verdict == expected)
            if i < len(per_item) and "claim" in per_item[i]:
                per_item[i]["debate_verdict"] = result.verdict.value
                per_item[i]["debate_confidence"] = result.confidence
        except Exception as e:
            if i < len(per_item):
                per_item[i]["debate_error"] = str(e)
    deb_duration = time.monotonic() - deb_start

    return ComparisonResult(
        dataset=dataset_name,
        total=len(items),
        isolation_accuracy=accuracy(iso_preds, iso_labels),
        debate_accuracy=accuracy(deb_preds, deb_labels),
        isolation_ece=expected_calibration_error(iso_confs, iso_correct) if iso_confs else 0.0,
        debate_ece=expected_calibration_error(deb_confs, deb_correct) if deb_confs else 0.0,
        isolation_duration=iso_duration,
        debate_duration=deb_duration,
        per_item=per_item,
    )
