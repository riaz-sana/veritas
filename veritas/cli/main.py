"""Veritas CLI — command-line interface for claim verification."""
from __future__ import annotations
import asyncio
import json
import sys
from typing import Optional
import typer
from rich.console import Console
from veritas.core.config import Config, VeritasConfigError
from veritas.core.verify import verify

app = typer.Typer(name="veritas", help="Adversarial parallel verification of AI outputs.")
console = Console()

@app.command()
def check(
    claim: str = typer.Argument("", help="The claim to verify"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full evidence chain"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Domain hint"),
    ref: Optional[list[str]] = typer.Option(None, "--ref", "-r", help="Reference document path"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    no_search: bool = typer.Option(False, "--no-search", help="Disable web search"),
    stdin: bool = typer.Option(False, "--stdin", help="Read claim from stdin"),
):
    """Verify a claim for factual accuracy."""
    if stdin:
        claim = sys.stdin.read().strip()
    if not claim:
        console.print("[red]Error: claim must not be empty[/red]")
        raise typer.Exit(code=1)
    try:
        config = Config()
        if no_search:
            config.search_api_key = ""
        result = asyncio.run(verify(claim=claim, domain=domain, references=ref or [], model=model, config=config))
        if output_json:
            console.print_json(json.dumps(result.to_dict(), indent=2))
        elif verbose:
            console.print(result.report())
        else:
            console.print(str(result))
    except VeritasConfigError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def shell():
    """Start an interactive verification shell."""
    from veritas.cli.shell import run_shell
    run_shell()

@app.command()
def benchmark(
    dataset: str = typer.Option("sample", "--dataset", "-d", help="Dataset: sample, truthfulqa"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
):
    """Run benchmarks against standard datasets."""
    from veritas.benchmarks.datasets import load_sample, load_truthfulqa
    from veritas.benchmarks.runner import run_benchmark

    loaders = {
        "sample": load_sample,
        "truthfulqa": load_truthfulqa,
    }

    if dataset not in loaders:
        console.print(f"[red]Unknown dataset: {dataset}. Available: {', '.join(loaders)}[/red]")
        raise typer.Exit(code=1)

    try:
        items = loaders[dataset]()
        console.print(f"Running benchmark on {len(items)} items from [bold]{dataset}[/bold]...")

        config = Config()
        if model_name:
            config.model = model_name

        result = asyncio.run(run_benchmark(items, dataset_name=dataset, config=config))

        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"  Accuracy: {result.accuracy:.2%}")
        console.print(f"  ECE:      {result.ece:.4f}")
        console.print(f"  Duration: {result.duration_seconds:.1f}s")

        if output:
            with open(output, "w") as f:
                f.write(result.to_json())
            console.print(f"\nFull results written to {output}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

@app.command()
def compare(
    dataset: str = typer.Option("sample", "--dataset", "-d", help="Dataset: sample, truthfulqa"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    model_name: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
):
    """Compare isolation-divergent vs shared-context debate verification."""
    from veritas.benchmarks.datasets import load_sample, load_truthfulqa
    from veritas.benchmarks.comparison import run_comparison

    loaders = {"sample": load_sample, "truthfulqa": load_truthfulqa}
    if dataset not in loaders:
        console.print(f"[red]Unknown dataset: {dataset}. Available: {', '.join(loaders)}[/red]")
        raise typer.Exit(code=1)

    try:
        items = loaders[dataset]()
        console.print(f"Running isolation vs debate comparison on {len(items)} items from [bold]{dataset}[/bold]...")
        console.print("This runs each claim TWICE (isolation + debate). Please wait.\n")

        config = Config()
        if model_name:
            config.model = model_name

        result = asyncio.run(run_comparison(items, dataset_name=dataset, config=config))
        console.print(result.summary())

        if output:
            with open(output, "w") as f:
                f.write(result.to_json())
            console.print(f"\nFull results written to {output}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
