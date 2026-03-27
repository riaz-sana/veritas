"""Dataset loaders for benchmarking."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class BenchmarkItem:
    claim: str
    expected_verdict: str
    domain: str = "general"
    source: str = ""

def load_truthfulqa() -> list[BenchmarkItem]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("TruthfulQA requires the 'datasets' package. Install with: pip install veritas-verify[benchmarks]")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    items = []
    for row in ds:
        items.append(BenchmarkItem(claim=row["question"], expected_verdict="VERIFIED", domain="general", source="truthfulqa"))
    return items

def load_sample() -> list[BenchmarkItem]:
    return [
        BenchmarkItem(claim="Water boils at 100 degrees Celsius at sea level.", expected_verdict="VERIFIED", domain="scientific"),
        BenchmarkItem(claim="The Great Wall of China is visible from space.", expected_verdict="REFUTED", domain="general"),
        BenchmarkItem(claim="The first iPhone was released in 2006.", expected_verdict="REFUTED", domain="technical"),
        BenchmarkItem(claim="Python is a compiled language.", expected_verdict="REFUTED", domain="technical"),
        BenchmarkItem(claim="Light travels at approximately 300,000 km/s.", expected_verdict="PARTIAL", domain="scientific"),
        BenchmarkItem(claim="All birds can fly.", expected_verdict="REFUTED", domain="scientific"),
        BenchmarkItem(claim="The Earth is the third planet from the Sun.", expected_verdict="VERIFIED", domain="scientific"),
        BenchmarkItem(claim="JavaScript was created by Sun Microsystems.", expected_verdict="REFUTED", domain="technical"),
    ]
