"""
Multi-Agent Financial Advisory System
CLI entry point powered by the shared orchestrator with Rich UI.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table
from rich.spinner import Spinner
from rich.align import Align
from rich.rule import Rule
from rich import box

from orchestrator import run_full_analysis

console = Console()

# ASCII Art Banner
WELCOME_ASCII = r"""
  _____ _                       _       _    _                _
 |  ___(_)_ __   __ _ _ __   ___(_) __ _| |  / \   __ _  ___ _ __ | |_
 | |_  | | '_ \ / _` | '_ \ / __| |/ _` | | / _ \ / _` |/ _ \ '_ \| __|
 |  _| | | | | | (_| | | | | (__| | (_| | |/ ___ \ (_| |  __/ | | | |_
 |_|   |_|_| |_|\__,_|_| |_|\___|_|\__,_|_/_/   \_\__, |\___|_| |_|\__|
                                                    |___/
"""

# Agent pipeline definition
PIPELINE_AGENTS = [
    ("Data Collection", "Historical Agent", "historical"),
    ("Data Collection", "Indicator Agent", "indicator"),
    ("Data Collection", "News Sentiment Agent", "news"),
    ("Data Collection", "Fundamental Agent", "fundamental"),
    ("Data Collection", "Macro Agent", "macro"),
    ("Data Collection", "Pair Monitor Agent", "pair_monitor"),
    ("Quantitative Analysis", "Feature Engineering Agent", "feature"),
    ("Quantitative Analysis", "Regime Agent", "regime"),
    ("Quantitative Analysis", "Forecast Agent", "forecast"),
    ("Risk & Validation", "Risk Agent", "risk"),
    ("Memory & Decision", "Memory Agent", "memory"),
    ("Memory & Decision", "Supervisor Agent", "supervisor"),
]

# Map pipeline step numbers to agent keys
STEP_TO_AGENT_KEY = {
    1: None,            # Initialization + memory
    2: "historical",    # Layer 0 parallel (historical, indicator, news, feature, backtest, pair_ledger, fundamental, macro)
    3: None,            # Layer 0 complete
    4: "pair_monitor",  # Layer 1
    5: "regime",        # Layer 1
    6: "forecast",      # Layer 2
    7: "risk",          # Layer 3
    8: "supervisor",    # Layer 4
}


class ProgressTracker:
    """Track agent progress for the live display."""

    def __init__(self):
        self.agent_status: dict[str, str] = {}
        self.messages: list[tuple[str, str, str]] = []
        self.current_report: str | None = None
        self.start_time: float = time.time()

        # Initialize all agents as pending
        for _, agent_name, _ in PIPELINE_AGENTS:
            self.agent_status[agent_name] = "pending"

    def update_agent(self, agent_key: str, status: str):
        """Update agent status by key."""
        for _, agent_name, key in PIPELINE_AGENTS:
            if key == agent_key:
                self.agent_status[agent_name] = status
                break

    def add_message(self, msg_type: str, content: str):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, msg_type, content))
        # Keep only last 10 messages
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]

    def get_elapsed(self) -> str:
        """Get formatted elapsed time."""
        elapsed = time.time() - self.start_time
        return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"


def create_layout() -> Layout:
    """Create the Rich layout for the live display."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3),
        Layout(name="report", ratio=5),
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2),
        Layout(name="messages", ratio=3),
    )
    return layout


def update_display(layout: Layout, tracker: ProgressTracker):
    """Update all panels in the live display."""

    # --- Header ---
    layout["header"].update(
        Panel(
            "[bold green]Multi-Agent Financial Advisory System[/bold green]  "
            "[dim]Powered by LLM Agents[/dim]",
            border_style="green",
            padding=(0, 2),
        )
    )

    # --- Progress Panel ---
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAD,
        padding=(0, 2),
        expand=True,
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=22)
    progress_table.add_column("Agent", style="green", justify="center", width=26)
    progress_table.add_column("Status", style="yellow", justify="center", width=14)

    # Group agents by team
    teams: dict[str, list[str]] = {}
    for team, agent_name, _ in PIPELINE_AGENTS:
        teams.setdefault(team, []).append(agent_name)

    for team, agents in teams.items():
        for i, agent_name in enumerate(agents):
            status = tracker.agent_status.get(agent_name, "pending")
            team_label = team if i == 0 else ""

            if status == "in_progress":
                status_cell = Spinner("dots", text="[blue]running[/blue]", style="bold cyan")
            else:
                color_map = {
                    "pending": "yellow",
                    "completed": "green",
                    "skipped": "dim",
                    "error": "red",
                }
                color = color_map.get(status, "white")
                status_cell = f"[{color}]{status}[/{color}]"

            progress_table.add_row(team_label, agent_name, status_cell)

        # Separator between teams
        progress_table.add_row("─" * 22, "─" * 26, "─" * 14, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Agent Progress", border_style="cyan", padding=(1, 1))
    )

    # --- Messages Panel ---
    msg_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.MINIMAL,
        show_lines=True,
        padding=(0, 1),
        expand=True,
    )
    msg_table.add_column("Time", style="cyan", width=8, justify="center")
    msg_table.add_column("Type", style="green", width=10, justify="center")
    msg_table.add_column("Content", style="white", no_wrap=False, ratio=1)

    for timestamp, msg_type, content in reversed(tracker.messages):
        wrapped = Text(content[:200], overflow="fold")
        msg_table.add_row(timestamp, msg_type, wrapped)

    layout["messages"].update(
        Panel(msg_table, title="Messages", border_style="blue", padding=(1, 1))
    )

    # --- Report Panel ---
    if tracker.current_report:
        layout["report"].update(
            Panel(
                Markdown(tracker.current_report),
                title="Current Analysis",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["report"].update(
            Panel(
                "[italic dim]Waiting for analysis results...[/italic dim]",
                title="Current Analysis",
                border_style="green",
                padding=(1, 2),
            )
        )

    # --- Footer ---
    agents_completed = sum(1 for s in tracker.agent_status.values() if s == "completed")
    agents_total = len(tracker.agent_status)
    elapsed = tracker.get_elapsed()

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(
        f"Agents: {agents_completed}/{agents_total}  |  ⏱ {elapsed}"
    )
    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def display_welcome():
    """Display the welcome screen with ASCII art."""
    welcome_content = f"[bold cyan]{WELCOME_ASCII}[/bold cyan]\n"
    welcome_content += "[bold green]Multi-Agent Financial Advisory System[/bold green]\n\n"
    welcome_content += "[bold]Analysis Pipeline:[/bold]\n"
    welcome_content += (
        "I. Data Collection → II. Quantitative Analysis → "
        "III. Risk & Validation → IV. Memory & Decision\n\n"
    )
    welcome_content += "[dim]Historical · Indicators · News · Pairs · Features · Regime · Forecast · Risk · Memory · Supervisor[/dim]"

    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome",
        subtitle="Powered by LLM Agents",
    )
    console.print(Align.center(welcome_box))
    console.print()


def create_question_box(title: str, prompt: str, default: str | None = None) -> Panel:
    """Create a styled question box, similar to TradingAgents' design."""
    content = f"[bold]{title}[/bold]\n"
    content += f"[dim]{prompt}[/dim]"
    if default:
        content += f"\n[dim]Default: {default}[/dim]"
    return Panel(content, border_style="blue", padding=(1, 2))


def get_user_selections() -> dict:
    """Gather user inputs with styled prompts."""
    # Step 1: Stock symbol
    console.print(
        create_question_box(
            "Step 1: Stock Symbol",
            "Enter the stock ticker symbol to analyze",
            "AAPL",
        )
    )
    stock_symbol = console.input("[bold cyan]> [/bold cyan]").strip().upper()
    if not stock_symbol:
        stock_symbol = "AAPL"
    console.print(f"  [green]✓[/green] Selected: [bold]{stock_symbol}[/bold]\n")

    # Step 2: Verbose mode
    console.print(
        create_question_box(
            "Step 2: Verbose Mode",
            "Enable verbose output for detailed agent logs? (y/N)",
            "N",
        )
    )
    verbose_input = console.input("[bold cyan]> [/bold cyan]").strip().lower()
    verbose = verbose_input in ("y", "yes")
    console.print(f"  [green]✓[/green] Verbose: [bold]{'Enabled' if verbose else 'Disabled'}[/bold]\n")

    return {
        "stock_symbol": stock_symbol,
        "verbose": verbose,
    }


def display_final_report(result: dict):
    """Display the final report with Rich formatting, inspired by TradingAgents."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))
    console.print()

    results = result.get("results", {})
    stock_symbol = result.get("stock_symbol", "UNKNOWN")
    timestamp = result.get("timestamp", "N/A")

    # Header info
    header_table = Table(show_header=False, box=box.ROUNDED, border_style="cyan", expand=True)
    header_table.add_column("Field", style="bold cyan", width=20)
    header_table.add_column("Value", style="white")
    header_table.add_row("Stock Symbol", stock_symbol)
    header_table.add_row("Generated", timestamp)
    console.print(header_table)
    console.print()

    # I. Data Collection Reports
    data_sections = []
    if results.get("historical", {}).get("analysis"):
        data_sections.append(("Historical Analysis", results["historical"]["analysis"]))
    if results.get("indicator", {}).get("analysis"):
        data_sections.append(("Indicator Analysis", results["indicator"]["analysis"]))
    if results.get("news", {}).get("analysis"):
        data_sections.append(("News Sentiment", results["news"]["analysis"]))
    pair_summary = results.get("pair_monitor", {}).get("summary")
    if pair_summary:
        data_sections.append(("Pair Monitor", pair_summary))

    # Fundamental Analysis
    if results.get("fundamental", {}).get("analysis"):
        data_sections.append(("Fundamental Analysis", results["fundamental"]["analysis"]))

    # Macro Analysis
    if results.get("macro", {}).get("analysis"):
        data_sections.append(("Macroeconomic Analysis", results["macro"]["analysis"]))

    if data_sections:
        console.print(Panel("[bold]I. Data Collection Reports[/bold]", border_style="cyan"))
        for title, content in data_sections:
            console.print(
                Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2))
            )
        console.print()

    # II. Quantitative Analysis
    quant_sections = []
    if results.get("feature", {}).get("summary"):
        quant_sections.append(("Feature Engineering", results["feature"]["summary"]))
    if results.get("regime", {}).get("summary"):
        quant_sections.append(("Regime Classification", results["regime"]["summary"]))
    if results.get("forecast", {}).get("summary"):
        quant_sections.append(("Probabilistic Forecast", results["forecast"]["summary"]))

    if quant_sections:
        console.print(Panel("[bold]II. Quantitative Analysis[/bold]", border_style="magenta"))
        for title, content in quant_sections:
            console.print(
                Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2))
            )
        console.print()

    # III. Risk & Validation
    risk_sections = []
    if results.get("risk", {}).get("summary"):
        risk_sections.append(("Risk Management Plan", results["risk"]["summary"]))

    if risk_sections:
        console.print(Panel("[bold]III. Risk & Validation[/bold]", border_style="red"))
        for title, content in risk_sections:
            console.print(
                Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2))
            )
        console.print()

    # III.5 Memory (Historical Prediction Performance)
    memory_data = results.get("memory", {})
    memory_summary = memory_data.get("summary")
    if memory_summary and memory_data.get("status") != "skipped":
        console.print(Panel("[bold]III.5 Historical Prediction Performance (Memory)[/bold]", border_style="yellow"))
        memory_content = memory_summary
        memory_stats = memory_data.get("memory", {})
        if memory_stats.get("prediction_count", 0) > 0:
            memory_content += f"\n\n**Directional Accuracy:** {memory_stats.get('directional_accuracy', 0):.0%}"
            memory_content += f"\n**Track Record Factor:** {memory_data.get('track_record_factor', 'N/A')}"
        console.print(
            Panel(Markdown(memory_content), title="Memory Agent", border_style="yellow", padding=(1, 2))
        )
        console.print()

    # IV. Final Recommendation
    recommendation = results.get("recommendation", {})
    rec_text = recommendation.get("recommendation", "")
    if rec_text:
        console.print(Panel("[bold]IV. Final Recommendation[/bold]", border_style="green"))
        console.print(
            Panel(Markdown(rec_text), title="Supervisor Decision", border_style="green", padding=(1, 2))
        )
        console.print()

    # Agent Status Summary Table
    console.print(Rule("Agent Status Summary", style="bold cyan"))
    status_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        border_style="cyan",
        expand=True,
    )
    status_table.add_column("Agent", style="cyan", justify="center")
    status_table.add_column("Status", style="green", justify="center")

    status_fields = [
        ("Historical Analysis", recommendation.get("historical_status", "unknown")),
        ("Indicator Analysis", recommendation.get("indicator_status", "unknown")),
        ("News Sentiment", recommendation.get("news_status", "unknown")),
        ("Pair Monitor", recommendation.get("pair_monitor_status", "unknown")),
        ("Feature Engineering", recommendation.get("feature_status", "unknown")),
        ("Regime Classification", recommendation.get("regime_status", "unknown")),
        ("Forecast", recommendation.get("forecast_status", "unknown")),
        ("Risk Management", recommendation.get("risk_status", "unknown")),
        ("Memory", recommendation.get("memory_status", "unknown")),
        ("Fundamental Analysis", recommendation.get("fundamental_status", "unknown")),
        ("Macroeconomic Analysis", recommendation.get("macro_status", "unknown")),
    ]
    for agent_name, status in status_fields:
        color = "green" if status == "success" else ("red" if status == "error" else "yellow")
        status_table.add_row(agent_name, f"[{color}]{status}[/{color}]")

    console.print(status_table)
    console.print()


def rich_progress_callback(tracker: ProgressTracker, layout: Layout, live: Live):
    """Create a progress callback that updates the Rich live display."""

    def callback(step: int, total: int, message: str):
        tracker.add_message("System", message)

        # Mark previous agents as completed, current as in_progress
        for s, agent_key in STEP_TO_AGENT_KEY.items():
            if agent_key is None:
                continue  # Skip initialization step
            if s < step:
                tracker.update_agent(agent_key, "completed")
            elif s == step:
                tracker.update_agent(agent_key, "in_progress")

        # Memory agent runs during initialization (step 1), mark it based on progress
        if step >= 2:
            tracker.update_agent("memory", "completed")

        # Layer 0 parallel agents: mark all as in_progress during step 2,
        # and completed once step 3 (Layer 0 complete) is reached.
        layer0_agents = ["historical", "indicator", "news", "feature", "fundamental", "macro"]
        if step == 2:
            for agent_key in layer0_agents:
                tracker.update_agent(agent_key, "in_progress")
        elif step >= 3:
            for agent_key in layer0_agents:
                tracker.update_agent(agent_key, "completed")

        # Update current report preview
        tracker.current_report = f"### Step {step}/{total}\n{message}\n\n*Processing...*"

        update_display(layout, tracker)
        live.refresh()

    return callback


def main():
    """Run the financial advisory pipeline from the terminal."""
    load_dotenv()

    # Welcome screen
    display_welcome()

    # Gather user inputs
    selections = get_user_selections()
    stock_symbol = selections["stock_symbol"]
    verbose = selections["verbose"]

    if not stock_symbol:
        console.print("[red]No stock symbol provided. Exiting.[/red]")
        return

    # Confirmation
    console.print(Rule("Starting Analysis", style="bold green"))
    console.print(
        Panel(
            f"[bold]Ticker:[/bold] {stock_symbol}\n"
            f"[bold]Verbose:[/bold] {'Yes' if verbose else 'No'}\n"
            f"[bold]Pipeline:[/bold] 13 agents (incl. Memory, Fundamental & Macro)",
            title="Configuration",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # Initialize tracker
    tracker = ProgressTracker()
    layout = create_layout()

    # Run analysis with live display
    with Live(layout, console=console, refresh_per_second=4) as live:
        update_display(layout, tracker)

        progress_cb = rich_progress_callback(tracker, layout, live)

        result = run_full_analysis(
            stock_symbol,
            verbose=verbose,
            persist=True,
            save_report=True,
            progress_callback=progress_cb,
        )

        # Mark all agents as completed or error
        if result.get("status") == "success":
            for _, agent_name, _ in PIPELINE_AGENTS:
                tracker.agent_status[agent_name] = "completed"
            tracker.add_message("System", "✓ Analysis completed successfully!")
            tracker.current_report = "### Analysis Complete\nAll agents have finished. Preparing final report..."
        else:
            tracker.add_message("Error", result.get("error", "Unknown error"))
            tracker.current_report = f"### Error\n{result.get('error', 'Unknown pipeline error')}"

        update_display(layout, tracker)
        live.refresh()
        time.sleep(1)  # Brief pause to show final status

    # Post-analysis output
    console.print()

    if result.get("status") != "success":
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {result.get('error', 'Unknown pipeline error')}",
                border_style="red",
                padding=(1, 2),
            )
        )
        return

    console.print("[bold cyan]Analysis Complete![/bold cyan]\n")

    # Prompt to display full report
    console.print(
        create_question_box("View Report", "Display the full analysis report? (Y/n)", "Y")
    )
    display_choice = console.input("[bold cyan]> [/bold cyan]").strip().upper()
    if display_choice not in ("N", "NO"):
        display_final_report(result)

    # Report file info
    output_file = result.get("output_file")
    if output_file:
        console.print(f"  [green]✓[/green] Full report saved to: [bold]{output_file}[/bold]")

    if result.get("persisted"):
        console.print("  [green]✓[/green] Stored recommendation, pair signals, and forecast snapshot")
    else:
        console.print("  [dim]Storage write skipped or unavailable.[/dim]")

    console.print()
    console.print(
        Panel(
            f"[bold green]Analysis of {stock_symbol} completed in {tracker.get_elapsed()}[/bold green]",
            border_style="green",
            padding=(0, 2),
        )
    )

    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Analysis interrupted by user.[/yellow]")
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        import traceback
        console.print_exception()
