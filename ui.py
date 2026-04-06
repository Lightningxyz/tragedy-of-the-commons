"""
Tragedy of the Commons — Rich Terminal Dashboard

Renders a live-updating terminal UI with resource pool visualization,
agent scoreboards, reasoning panels, and end-game summary.
"""

from __future__ import annotations
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.bar import Bar
from rich import box

POOL_GRADIENT = ["#ff4444", "#ff8800", "#ffcc00", "#88cc00", "#00cc66"]

console = Console()


def _pool_color(pool: int, capacity: int) -> str:
    """Pick a color based on pool health percentage."""
    pct = pool / capacity if capacity > 0 else 0
    idx = min(int(pct * (len(POOL_GRADIENT) - 1)), len(POOL_GRADIENT) - 1)
    return POOL_GRADIENT[idx]


def _pool_bar(pool: int, capacity: int) -> str:
    """ASCII bar chart of pool health."""
    width = 30
    filled = int((pool / capacity) * width) if capacity > 0 else 0
    empty = width - filled
    color = _pool_color(pool, capacity)
    return f"[{color}]{'█' * filled}[/]{'░' * empty} {pool}/{capacity}"


def render_round(
    round_num: int,
    max_rounds: int,
    pool_before: int,
    pool_after: int,
    replenished: int,
    capacity: int,
    agents: list,
    reasonings: dict[str, str],
    messages: dict[str, str],
    granted: dict[str, int],
    requested: dict[str, int],
    effective_requested: dict[str, int] | None = None,
    pledges: dict[str, int | None] | None = None,
    penalties: dict[str, int] | None = None,
    binding_quota: int | None = None,
):
    """Render a complete round to the terminal."""

    # ── Header ────────────────────────────────────────────────
    header = Text()
    header.append(f" ⏱  ROUND {round_num}/{max_rounds} ", style="bold white on blue")
    console.print()
    console.print(header)
    console.print()

    # ── Resource Pool ─────────────────────────────────────────
    pool_panel = Panel(
        f"Before extraction: {pool_before}\n"
        f"After extraction:  {pool_after - replenished}\n"
        f"Replenished:       [green]+{replenished}[/green]\n"
        f"Pool now:          {_pool_bar(pool_after, capacity)}\n"
        f"Binding quota:     {binding_quota if binding_quota is not None else 'none'}",
        title="🌊 Resource Pool",
        border_style=_pool_color(pool_after, capacity),
        expand=False,
        padding=(0, 2),
    )
    console.print(pool_panel)

    # ── Agent Scoreboard ──────────────────────────────────────
    table = Table(
        box=box.ROUNDED,
        title="📊 Agent Scoreboard",
        title_style="bold",
        show_lines=True,
        expand=False,
    )
    table.add_column("Agent", style="bold cyan", min_width=14)
    table.add_column("Requested", justify="center", style="yellow")
    table.add_column("Effective", justify="center", style="blue")
    table.add_column("Granted", justify="center", style="green")
    table.add_column("Pledge", justify="center", style="cyan")
    table.add_column("Penalty", justify="center", style="red")
    table.add_column("Inventory", justify="center", style="magenta bold")
    table.add_column("Message", style="dim", max_width=45)

    for agent in agents:
        req = str(requested.get(agent.name, "-"))
        eff = str((effective_requested or requested).get(agent.name, "-"))
        grt = str(granted.get(agent.name, "-"))
        pledge = pledges.get(agent.name) if pledges else None
        pledge_text = str(pledge) if pledge is not None else "-"
        penalty = str((penalties or {}).get(agent.name, 0))
        msg = messages.get(agent.name, "-")
        if len(msg) > 42:
            msg = msg[:42] + "…"
        table.add_row(
            agent.name,
            req,
            eff,
            grt,
            pledge_text,
            penalty,
            str(agent.inventory),
            msg,
        )

    console.print(table)

    # ── Reasoning Panels ──────────────────────────────────────
    panels = []
    agent_colors = ["cyan", "green", "yellow", "red", "magenta"]
    for i, agent in enumerate(agents):
        color = agent_colors[i % len(agent_colors)]
        reasoning = reasonings.get(agent.name, "—")
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "…"
        panels.append(
            Panel(
                reasoning,
                title=f"🧠 {agent.name}",
                border_style=color,
                width=40,
                padding=(0, 1),
            )
        )
    console.print(Columns(panels, padding=(0, 1)))
    console.print("─" * 80, style="dim")


def render_finale(
    round_num: int,
    pool: int,
    capacity: int,
    agents: list,
    history: list,
):
    """Render the end-of-game summary with final standings."""

    console.print()

    if pool <= 0:
        title_text = "💀  TRAGEDY OF THE COMMONS  💀"
        subtitle = f"The resource pool collapsed at round {round_num}."
        border = "bold red"
    else:
        title_text = "🎉  SIMULATION COMPLETE  🎉"
        subtitle = f"The commons survived all {round_num} rounds! Pool: {pool}/{capacity}"
        border = "bold green"

    # ── Final Standings ───────────────────────────────────────
    table = Table(
        box=box.DOUBLE_EDGE,
        title="🏆 Final Standings",
        title_style="bold",
        show_lines=True,
    )
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Agent", style="bold cyan")
    table.add_column("Total Harvested", justify="center", style="green")
    table.add_column("Avg / Round", justify="center", style="yellow")

    sorted_agents = sorted(agents, key=lambda a: a.inventory, reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    for rank, agent in enumerate(sorted_agents):
        medal = medals[rank] if rank < len(medals) else f" {rank + 1}"
        avg = f"{agent.inventory / max(1, len(agent.history)):.1f}"
        table.add_row(medal, agent.name, str(agent.inventory), avg)

    body = Group(
        Text(subtitle, justify="center"),
        Text(),
        table,
    )
    console.print(Panel(body, title=title_text, border_style=border, padding=(1, 4)))


def render_intro(
    agents: list,
    pool: int,
    capacity: int,
    max_rounds: int,
    experiment: str = "cheap_talk",
    roster: str = "heterogeneous",
    prompt_mode: str = "benchmark",
    realism: str = "perfect",
):
    """Show a nice intro banner before the simulation starts."""
    console.print()
    console.print(
        Panel(
            f"[bold]Agents:[/bold] {', '.join(a.name for a in agents)}\n"
            f"[bold]Starting Pool:[/bold] {pool}/{capacity}\n"
            f"[bold]Rounds:[/bold] {max_rounds}\n"
            f"[bold]Experiment:[/bold] {experiment}\n"
            f"[bold]Roster:[/bold] {roster}\n"
            f"[bold]Prompt Mode:[/bold] {prompt_mode}\n"
            f"[bold]Realism:[/bold] {realism}",
            title="🌍  TRAGEDY OF THE COMMONS  🌍",
            subtitle="Multi-Agent Game Theory Simulation",
            border_style="bold blue",
            padding=(1, 3),
        )
    )
    console.print()
