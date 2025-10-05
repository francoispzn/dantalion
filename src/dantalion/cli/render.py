"""Rich rendering of reports and eval results for the terminal.

Kept separate from the command wiring so the formatting can be unit-tested on its
own, and so the commands stay thin: parse arguments, call the library, hand the
result to one of these.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dantalion.domains.anomaly.report import Investigation
from dantalion.eval.metrics import EvalReport


def render_report(report: Investigation, console: Console) -> None:
    """Print an investigation report as panels and a ranked hypothesis table."""
    console.print(Panel(report.summary or "(no summary)", title="summary", expand=False))
    console.print(f"[bold]root cause:[/bold] {report.root_cause}")

    if report.timeline:
        console.print("\n[bold]timeline[/bold]")
        for moment in report.timeline:
            console.print(f"  • {moment}")

    table = Table(title="hypotheses", show_lines=False)
    table.add_column("confidence", justify="right")
    table.add_column("statement")
    table.add_column("evidence")
    for hypothesis in sorted(report.hypotheses, key=lambda h: h.confidence, reverse=True):
        table.add_row(
            f"{hypothesis.confidence:.2f}", hypothesis.statement, "; ".join(hypothesis.evidence)
        )
    console.print(table)

    _bullet_section(console, "recommended actions", report.recommended_actions)
    _bullet_section(console, "open questions", report.open_questions)


def render_eval(report: EvalReport, console: Console) -> None:
    """Print a per-scenario eval table plus the aggregate summary."""
    table = Table(title="evaluation")
    table.add_column("scenario")
    table.add_column("root cause", justify="center")
    table.add_column("tool use", justify="right")
    table.add_column("steps", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("strategy")
    for score in report.scores:
        table.add_row(
            score.name,
            "[green]hit[/green]" if score.root_cause_hit else "[red]miss[/red]",
            f"{score.relevant_tool_use:.0%}",
            str(score.steps),
            str(score.tokens),
            score.report_strategy,
        )
    console.print(table)
    summary = report.summary()
    console.print(
        f"accuracy [bold]{summary['accuracy']}[/bold] over {summary['scenarios']} scenarios, "
        f"{summary['total_tokens']} tokens"
    )


def _bullet_section(console: Console, title: str, items: list[str]) -> None:
    if not items:
        return
    console.print(f"\n[bold]{title}[/bold]")
    for item in items:
        console.print(f"  • {item}")
