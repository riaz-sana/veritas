"""Interactive verification shell."""
from __future__ import annotations
import asyncio
from rich.console import Console
from veritas.core.config import Config, VeritasConfigError
from veritas.core.verify import verify

console = Console()

def run_shell():
    console.print("[bold]Veritas v0.1.0[/bold] — Type a claim to verify. /help for commands.\n")
    verbose = False
    try:
        Config().validate()
    except VeritasConfigError as e:
        console.print(f"[red]{e}[/red]")
        return
    while True:
        try:
            line = console.input("[bold cyan]veritas>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd = line.lower()
            if cmd in ("/quit", "/exit"):
                console.print("Bye!")
                break
            elif cmd == "/verbose":
                verbose = not verbose
                console.print(f"Verbose mode {'on' if verbose else 'off'}.")
            elif cmd == "/help":
                console.print("Commands: /verbose, /quit, /help")
                console.print("Type any claim to verify it.")
            else:
                console.print(f"Unknown command: {line}")
            continue
        try:
            result = asyncio.run(verify(line))
            if verbose:
                console.print(result.report())
            else:
                console.print(str(result))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        console.print()
