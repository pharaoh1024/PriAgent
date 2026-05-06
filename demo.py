"""
PriAgent demo — runs a full end-to-end audit on sample data.

Run:  python demo.py
Requires: ANTHROPIC_API_KEY set in .env or environment.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# Make sure package root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

from schemas.data_models import DataFlow, AuditSummary, RiskCategory
from workflow.priagent_graph import PriAgentRunner
from data.sample_code import SAMPLE_DECOMPILED_CODE
from config import DATA_DIR

console = Console()


def load_sample_data():
    flows_path = DATA_DIR / "sample_flows.json"
    policy_path = DATA_DIR / "sample_policy.txt"

    with open(flows_path, encoding="utf-8") as f:
        flows_raw = json.load(f)
    policy_text = policy_path.read_text(encoding="utf-8")

    flows = [DataFlow(**r) for r in flows_raw]
    return flows, policy_text


def print_banner():
    console.print(Panel(
        "[bold cyan]PriAgent[/bold cyan] — Collaborative Multi-Agent Framework "
        "for Android Privacy Compliance Auditing\n"
        "[dim]AAAI 2026 Paper Implementation · Enhanced with LangChain + LangGraph[/dim]",
        box=box.DOUBLE,
        expand=False,
    ))


def print_summary(summary: AuditSummary):
    # ── Stats panel ──────────────────────────────────────────────────────
    stats_text = (
        f"App            : [bold]{summary.app_name}[/bold]\n"
        f"Session ID     : {summary.session_id}\n"
        f"Raw flows input: {summary.total_raw_flows}\n"
        f"FP eliminated  : [green]{summary.false_positives_eliminated}[/green] "
        f"({summary.fp_reduction_rate*100:.1f}% reduction)\n"
        f"Flows verified : {summary.processing_cost_estimate.get('flows_verified', '?')}\n"
        f"Reflections    : {summary.processing_cost_estimate.get('reflection_triggered', 0)}\n"
        f"Errors         : {summary.processing_cost_estimate.get('errors', 0)}"
    )
    console.print(Panel(stats_text, title="[bold]Audit Statistics[/bold]", expand=False))

    # ── Violations table ─────────────────────────────────────────────────
    if summary.confirmed_violations:
        table = Table(title="Confirmed Violations", box=box.ROUNDED, show_lines=True)
        table.add_column("Flow ID", style="cyan", no_wrap=True)
        table.add_column("Risk", style="bold red")
        table.add_column("Data Types")
        table.add_column("Briefing", max_width=55)
        table.add_column("Recommendation", max_width=40)

        for r in summary.confirmed_violations:
            risk_color = "red" if r.risk_category == RiskCategory.HIGH_RISK else "yellow"
            table.add_row(
                r.flow_id,
                f"[{risk_color}]{r.risk_category.value}[/{risk_color}]",
                ", ".join(r.data_types_involved) or "—",
                r.analyst_briefing[:200] + ("…" if len(r.analyst_briefing) > 200 else ""),
                r.recommendation[:150] + ("…" if len(r.recommendation) > 150 else ""),
            )
        console.print(table)
    else:
        console.print("[green]No confirmed violations found.[/green]")

    # ── Declared behaviours ───────────────────────────────────────────────
    if summary.declared_behaviors:
        console.print(
            f"\n[dim]{len(summary.declared_behaviors)} flow(s) are compliant with "
            f"the stated privacy policy.[/dim]"
        )


def main():
    print_banner()

    console.print("\n[bold]Loading sample data…[/bold]")
    flows, policy_text = load_sample_data()
    console.print(f"  {len(flows)} raw flows loaded from sample_flows.json")
    console.print(f"  Policy text: {len(policy_text)} characters")

    console.print("\n[bold]Initialising PriAgent runner (builds RAG index on first run)…[/bold]")
    runner = PriAgentRunner()
    console.print("  Runner ready.")

    console.print(f"\n[bold]Starting audit pipeline…[/bold]")
    console.print("  Stage 1  FlowShaper — abstracting data flows")
    console.print("  Stage 2  LocusFinder — identifying semantic loci")
    console.print("  Stage 3  SemanticVerifier — parallel RAG-powered verification")
    console.print("  Stage 4a PolicyScanner — parsing privacy policy")
    console.print("  Stage 4b ComplianceArbiter — synthesising final report\n")

    summary: AuditSummary = runner.run(
        app_name="Super Calculator (Demo)",
        raw_flows=flows,
        decompiled_code=SAMPLE_DECOMPILED_CODE,
        policy_text=policy_text,
    )

    console.print("\n[bold green]Audit complete.[/bold green]\n")
    print_summary(summary)

    # Save JSON report
    report_path = DATA_DIR / f"audit_report_{summary.session_id}.json"
    report_path.write_text(
        json.dumps(summary.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Full JSON report saved to: {report_path}[/dim]")


if __name__ == "__main__":
    main()
