"""The ``dantalion`` command-line interface.

A thin Typer layer over the library: each command resolves a provider from a
``kind:model`` spec, calls into the package, and renders the result. The heavy
lifting lives in the library and is tested there; the commands here stay small
enough to read at a glance.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from dantalion import __version__
from dantalion.cli.render import render_eval, render_report
from dantalion.config import Settings, make_provider
from dantalion.domains.anomaly import Dataset, investigate
from dantalion.errors import DantalionError
from dantalion.eval import default_scenarios, provider_solver, run_eval

app = typer.Typer(
    add_completion=False,
    help="A model-agnostic, local-first autonomous investigation agent.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(__version__)


@app.command()
def models(spec: str = typer.Argument(..., help="A provider spec, e.g. ollama:llama3.1")) -> None:
    """Show the resolved capabilities for a model spec."""
    settings = Settings()
    provider = make_provider(spec, base_url=settings.base_url, api_key=settings.api_key)
    console.print_json(provider.capabilities().model_dump_json())


@app.command(name="investigate")
def investigate_cmd(
    dataset: Path = typer.Argument(..., exists=True, help="Path to a CSV/JSONL dataset"),
    alert: str = typer.Option("investigate the anomaly in this data", "--alert", "-a"),
    model: str | None = typer.Option(None, "--model", "-m", help="Provider spec"),
    plan: bool = typer.Option(True, "--plan/--no-plan"),
    review: bool = typer.Option(True, "--review/--no-review"),
    max_steps: int = typer.Option(12, "--max-steps"),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON"),
    record: Path | None = typer.Option(None, "--record", help="Save a replay cassette here"),
) -> None:
    """Investigate an anomaly in a local dataset and print the report."""
    settings = Settings()
    provider = make_provider(
        model or settings.model, base_url=settings.base_url, api_key=settings.api_key
    )

    cassette = None
    if record is not None:
        from dantalion.trace import RecordingProvider

        recording = RecordingProvider(provider)
        provider = recording
        cassette = recording.cassette

    try:
        result = investigate(
            provider,
            Dataset.load(dataset),
            alert,
            plan=plan,
            review=review,
            max_steps=max_steps,
        )
    except DantalionError as exc:
        console.print(f"[red]investigation failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    if cassette is not None and record is not None:
        cassette.save(record)

    if as_json:
        console.print_json(result.report.model_dump_json())
    else:
        render_report(result.report, console)


@app.command(name="eval")
def eval_cmd(
    model: str | None = typer.Option(None, "--model", "-m"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Run the synthetic-incident evaluation suite."""
    settings = Settings()
    provider = make_provider(
        model or settings.model, base_url=settings.base_url, api_key=settings.api_key
    )
    report = run_eval(default_scenarios(), provider_solver(provider))
    if as_json:
        payload = {"summary": report.summary(), "scores": [vars(s) for s in report.scores]}
        console.print_json(json.dumps(payload))
    else:
        render_eval(report, console)


def main() -> None:
    app()
