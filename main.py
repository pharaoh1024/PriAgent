"""
PriAgent CLI — main entry point.

Usage:
  python main.py analyze --app-name "MyApp" --flows flows.json \\
                         --code code_dir/ --policy policy.txt
  python main.py demo
  python main.py build-kb
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import typer
from rich.console import Console

app = typer.Typer(help="PriAgent: Multi-Agent Android Privacy Compliance Auditor")
console = Console()


@app.command()
def demo():
    """Run the full pipeline on built-in sample data."""
    from demo import main as run_demo
    run_demo()


@app.command()
def build_kb():
    """(Re-)build the Android API FAISS knowledge base index."""
    import shutil
    from config import FAISS_INDEX_DIR
    if FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)
        console.print(f"[yellow]Removed existing index at {FAISS_INDEX_DIR}[/yellow]")
    from rag.knowledge_base import AndroidAPIKnowledgeBase
    console.print("[bold]Building FAISS knowledge base…[/bold]")
    kb = AndroidAPIKnowledgeBase()
    console.print(f"[green]Knowledge base built at {FAISS_INDEX_DIR}[/green]")


@app.command()
def analyze(
    app_name: str = typer.Option(..., help="Application name"),
    flows: Path = typer.Option(..., help="Path to JSON file with static analysis flows"),
    policy: Path = typer.Option(..., help="Path to plain-text privacy policy"),
    code_json: Path = typer.Option(
        None,
        "--code-json",
        help="JSON file mapping method signature -> source code (optional)"
    ),
    output: Path = typer.Option(None, help="Output JSON report path (default: stdout)"),
):
    """Audit a real app: provide static analysis flows, decompiled code, and privacy policy."""
    from schemas.data_models import DataFlow
    from workflow.priagent_graph import PriAgentRunner

    # Load flows
    raw = json.loads(flows.read_text(encoding="utf-8"))
    data_flows = [DataFlow(**r) for r in raw]
    console.print(f"Loaded {len(data_flows)} flows from {flows}")

    # Load policy
    policy_text = policy.read_text(encoding="utf-8")

    # Load decompiled code (optional)
    decompiled_code = {}
    if code_json and code_json.exists():
        decompiled_code = json.loads(code_json.read_text(encoding="utf-8"))
        console.print(f"Loaded {len(decompiled_code)} method sources from {code_json}")

    # Run
    runner = PriAgentRunner()
    summary = runner.run(
        app_name=app_name,
        raw_flows=data_flows,
        decompiled_code=decompiled_code,
        policy_text=policy_text,
    )

    result_json = json.dumps(summary.model_dump(), indent=2, ensure_ascii=False)

    if output:
        output.write_text(result_json, encoding="utf-8")
        console.print(f"[green]Report saved to {output}[/green]")
    else:
        console.print(result_json)


if __name__ == "__main__":
    app()
