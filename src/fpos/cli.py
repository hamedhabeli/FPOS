from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

import requests
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()
POLL_SECONDS = 2.0
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
MAX_VISIBLE_LOGS = 10


@dataclass
class ProjectSnapshot:
    project_id: str
    status: str
    delta_i: float
    delta_v: float
    depth: int
    iteration: int
    logs: List[Dict[str, Any]]
    progress_message: Optional[str] = None
    error: Optional[str] = None


def ask_inputs() -> tuple[str, str, str, str, List[str], List[str]]:
    base_url = Prompt.ask("[bold cyan]Backend URL[/]", default=DEFAULT_BASE_URL)
    idea = Prompt.ask("[bold cyan]Idea[/]")
    context = Prompt.ask("[bold cyan]Context[/]")
    gemini_key = Prompt.ask("[bold cyan]Gemini API key[/]", password=True)

    constraints_raw = Prompt.ask("[bold cyan]Constraints (comma-separated)[/]", default="")
    stakeholders_raw = Prompt.ask("[bold cyan]Stakeholders (comma-separated)[/]", default="")

    constraints = [x.strip() for x in constraints_raw.split(",") if x.strip()]
    stakeholders = [x.strip() for x in stakeholders_raw.split(",") if x.strip()]
    return base_url.rstrip("/"), idea, context, gemini_key, constraints, stakeholders


def create_project(
    session: requests.Session,
    base_url: str,
    idea: str,
    context: str,
    constraints: List[str],
    stakeholders: List[str],
) -> UUID:
    payload = {
        "idea": idea,
        "context": context,
        "constraints": constraints,
        "stakeholders": stakeholders,
    }
    resp = session.post(f"{base_url}/api/v1/projects", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return UUID(data["project_id"])


def fetch_status(session: requests.Session, base_url: str, project_id: UUID) -> ProjectSnapshot:
    resp = session.get(f"{base_url}/api/v1/projects/{project_id}/status", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return ProjectSnapshot(
        project_id=data["project_id"],
        status=data["status"],
        delta_i=float(data.get("delta_i", 1.0)),
        delta_v=float(data.get("delta_v", 1.0)),
        depth=int(data.get("depth", 0)),
        iteration=int(data.get("iteration", 0)),
        logs=data.get("logs", []),
        progress_message=data.get("progress_message"),
        error=data.get("error"),
    )


def fetch_roadmap(session: requests.Session, base_url: str, project_id: UUID) -> Dict[str, Any]:
    resp = session.get(f"{base_url}/api/v1/projects/{project_id}/roadmap", timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_header(snapshot: ProjectSnapshot) -> Panel:
    status_text = Text(snapshot.status.upper(), style="bold green" if snapshot.status == "completed" else "bold yellow")
    if snapshot.status == "failed":
        status_text.stylize("bold red")

    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_row("Status", status_text)
    grid.add_row("Depth", str(snapshot.depth))
    grid.add_row("Iteration", str(snapshot.iteration))
    grid.add_row("ΔI", f"{snapshot.delta_i:.3f}")
    grid.add_row("ΔV", f"{snapshot.delta_v:.3f}")
    if snapshot.progress_message:
        grid.add_row("Message", snapshot.progress_message)
    if snapshot.error:
        grid.add_row("Error", Text(snapshot.error, style="bold red"))

    return Panel(grid, title="FPOS Live State", border_style="cyan")


def build_progress(snapshot: ProjectSnapshot) -> Panel:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]Convergence[/]"),
        BarColumn(bar_width=28),
        TextColumn("Reduction: {task.completed:.0%}"),
        expand=True,
    )
    task_id = progress.add_task("convergence", total=1.0, completed=max(0.0, min(1.0, 1.0 - snapshot.delta_i)))
    progress.update(task_id, total=1.0, completed=max(0.0, min(1.0, 1.0 - snapshot.delta_i)))
    return Panel(progress, title="Uncertainty Reduction", border_style="magenta")


def build_logs(snapshot: ProjectSnapshot) -> Panel:
    table = Table(expand=True, show_lines=False)
    table.add_column("Time", style="dim", no_wrap=True)
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Depth", justify="right", no_wrap=True)
    table.add_column("Iter", justify="right", no_wrap=True)
    table.add_column("ΔI", justify="right", no_wrap=True)
    table.add_column("Message", overflow="fold")

    for log in snapshot.logs[-MAX_VISIBLE_LOGS:]:
        ts = str(log.get("ts", ""))
        agent = log.get("agent") or "orchestrator"
        depth = str(log.get("depth", 0))
        iteration = str(log.get("iteration", 0))
        delta_i = f"{float(log.get('delta_i', 0.0)):.3f}"
        message = str(log.get("message", ""))
        table.add_row(ts[-8:], agent, depth, iteration, delta_i, message)

    return Panel(table, title="Live Logs", border_style="blue")


def build_layout(snapshot: ProjectSnapshot) -> Panel:
    from rich.layout import Layout

    layout = Layout(name="root")
    layout.split_column(
        Layout(build_header(snapshot), name="header", size=7),
        Layout(build_progress(snapshot), name="progress", size=5),
        Layout(build_logs(snapshot), name="logs"),
    )
    return Panel(layout, border_style="white", title="FPOS Monitoring Console")


def build_tree(roadmap_payload: Dict[str, Any]) -> Tree:
    result = roadmap_payload.get("result", {})
    solution = result if isinstance(result, dict) else {}
    tasks = solution.get("roadmap") or solution.get("tasks") or []

    root = Tree(f"[bold green]FPOS Roadmap[/]  [dim](project {roadmap_payload.get('project_id', '')})[/]")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        owner = task.get("owner") or "Unknown"
        category = task.get("category") or "implementation"
        key = f"{owner} · {category}"
        groups.setdefault(key, []).append(task)

    for group_name in sorted(groups.keys()):
        group_branch = root.add(f"[bold cyan]{group_name}[/]  [dim]({len(groups[group_name])} tasks)[/]")
        for task in groups[group_name]:
            task_branch = group_branch.add(
                f"[bold]{task.get('title', 'Untitled')}[/]  "
                f"[dim]({task.get('estimated_effort_points', 1)} pts, {task.get('risk_level', 'medium')} risk)[/]"
            )
            desc = task.get("description")
            if desc:
                task_branch.add(Text(desc))
            dor = task.get("definition_of_ready") or []
            if dor:
                dor_branch = task_branch.add("[yellow]Definition of Ready[/]")
                for item in dor:
                    dor_branch.add(f"- {item}")
            acc = task.get("acceptance_criteria") or []
            if acc:
                acc_branch = task_branch.add("[green]Acceptance Criteria[/]")
                for item in acc:
                    acc_branch.add(f"- {item}")

    return root


def print_tree(roadmap_payload: Dict[str, Any]) -> None:
    console.print()
    console.print(build_tree(roadmap_payload))


def main() -> int:
    base_url, idea, context, gemini_key, constraints, stakeholders = ask_inputs()

    session = requests.Session()
    session.headers.update({"X-Gemini-Key": gemini_key})

    console.print("\n[bold green]Submitting project to FPOS...[/]")
    project_id = create_project(session, base_url, idea, context, constraints, stakeholders)
    console.print(f"[bold cyan]project_id[/]: {project_id}\n")

    seen_log_count = 0
    snapshot = ProjectSnapshot(
        project_id=str(project_id),
        status="pending",
        delta_i=1.0,
        delta_v=1.0,
        depth=0,
        iteration=0,
        logs=[],
    )

    with Live(build_layout(snapshot), console=console, refresh_per_second=6, screen=True) as live:
        while True:
            try:
                snapshot = fetch_status(session, base_url, project_id)
            except requests.RequestException as exc:
                live.update(Panel(f"[bold red]Network error:[/] {exc}", title="FPOS Monitoring Console"))
                time.sleep(POLL_SECONDS)
                continue

            if len(snapshot.logs) > seen_log_count:
                new_logs = snapshot.logs[seen_log_count:]
                for log in new_logs:
                    agent = log.get("agent") or "orchestrator"
                    console.log(
                        f"[cyan]{agent}[/] depth={log.get('depth', 0)} iter={log.get('iteration', 0)} "
                        f"ΔI={float(log.get('delta_i', 0.0)):.3f} :: {log.get('message', '')}"
                    )
                seen_log_count = len(snapshot.logs)

            live.update(build_layout(snapshot))

            if snapshot.status in {"completed", "failed"}:
                break

            time.sleep(POLL_SECONDS)

    if snapshot.status == "failed":
        console.print(f"\n[bold red]Project failed:[/] {snapshot.error or 'Unknown error'}")
        return 1

    console.print("\n[bold green]Project completed. Fetching roadmap...[/]")
    roadmap_payload = fetch_roadmap(session, base_url, project_id)
    print_tree(roadmap_payload)

    console.print("\n[bold green]Done.[/]")
    return 0
