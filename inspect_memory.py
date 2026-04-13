#!/usr/bin/env python3
"""
Memory Inspector
View stored prediction history, realized outcomes, and calibration data
for any stock symbol.

Usage:
    python inspect_memory.py                # Interactive mode
    python inspect_memory.py AAPL           # Inspect a specific stock
    python inspect_memory.py AAPL TSLA NVDA # Inspect multiple stocks
    python inspect_memory.py --all          # Show all stocks with data
    python inspect_memory.py --clear AAPL   # Clear memory for a specific stock
    python inspect_memory.py --clear-all    # Clear ALL memory records
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.markdown import Markdown
from rich import box

from utils.storage import Storage
from agents.memory_agent import MemoryAgent

console = Console()


def get_all_symbols(storage: Storage) -> list[str]:
    """Retrieve all distinct stock symbols that have predictions."""
    rows = storage._fetchall(
        "SELECT DISTINCT stock_symbol FROM predictions ORDER BY stock_symbol ASC"
    )
    return [r["stock_symbol"] for r in rows]


def display_tracked_predictions(storage: Storage, symbol: str) -> int:
    """Display tracked (matured) predictions with realized outcomes."""
    tracked = storage.get_tracked_predictions(symbol)

    if not tracked:
        console.print(f"  [dim]No tracked (matured) predictions for {symbol}.[/dim]\n")
        return 0

    table = Table(
        title=f"Tracked Predictions vs Actual Outcomes — {symbol}",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        border_style="cyan",
        expand=True,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Date", style="cyan", width=20, justify="center")
    table.add_column("Horizon", width=8, justify="center")
    table.add_column("Action", width=10, justify="center")
    table.add_column("P(Up)", width=10, justify="center")
    table.add_column("Realized Return", width=16, justify="right")
    table.add_column("Benchmark (SPY)", width=16, justify="right")
    table.add_column("Direction", width=10, justify="center")

    correct_count = 0
    total_count = 0

    for i, row in enumerate(tracked, 1):
        real_ret = row.get("realized_return")
        bench_ret = row.get("benchmark_return")
        prob_up = row.get("probability_up")

        # Determine direction correctness from action/probability_up
        action = row.get("action", "N/A")
        if real_ret is not None:
            total_count += 1
            action_lower = (action or "hold").lower()
            if action_lower == "buy":
                pred_dir = 1
            elif action_lower == "sell":
                pred_dir = -1
            elif prob_up is not None:
                pred_dir = 1 if prob_up >= 0.5 else -1
            else:
                pred_dir = 0
            real_dir = 1 if real_ret >= 0 else -1
            is_correct = pred_dir == real_dir
            if is_correct:
                correct_count += 1
            direction_cell = "[bold green]✓ Correct[/bold green]" if is_correct else "[bold red]✗ Wrong[/bold red]"
        else:
            direction_cell = "[dim]N/A[/dim]"

        # Format returns with color
        def fmt_return(val):
            if val is None:
                return "[dim]N/A[/dim]"
            color = "green" if val >= 0 else "red"
            return f"[{color}]{val:+.4f} ({val*100:+.2f}%)[/{color}]"

        # Format action with color
        action = row.get("action", "N/A")
        action_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(
            action.upper() if action else "", "white"
        )

        table.add_row(
            str(i),
            row.get("predicted_at", "N/A"),
            f"{row.get('horizon_days', 'N/A')}d",
            f"[{action_color}]{action}[/{action_color}]",
            f"{prob_up:.2f}" if prob_up is not None else "[dim]N/A[/dim]",
            fmt_return(real_ret),
            fmt_return(bench_ret),
            direction_cell,
        )

    console.print(table)
    console.print()

    return total_count


def display_pending_predictions(storage: Storage, symbol: str) -> None:
    """Display predictions that haven't matured yet."""
    # Get all recent predictions and filter for those without outcomes
    all_preds = storage.get_recent_predictions(symbol, limit=50)
    tracked = storage.get_tracked_predictions(symbol)
    tracked_dates = {(r["predicted_at"], r.get("horizon_days")) for r in tracked}

    pending = [
        p for p in all_preds
        if (p["created_at"], p.get("horizon_days")) not in tracked_dates
    ]

    if not pending:
        console.print(f"  [dim]No pending (unmatured) predictions for {symbol}.[/dim]\n")
        return

    table = Table(
        title=f"Pending Predictions (Not Yet Matured) — {symbol}",
        show_header=True,
        header_style="bold yellow",
        box=box.ROUNDED,
        border_style="yellow",
        expand=True,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Date", style="cyan", width=20, justify="center")
    table.add_column("Horizon", width=8, justify="center")
    table.add_column("Matures On", width=14, justify="center")
    table.add_column("Action", width=10, justify="center")
    table.add_column("P(Up)", width=8, justify="center")
    table.add_column("Model", width=14, justify="center")

    now = datetime.utcnow()

    for i, row in enumerate(pending, 1):
        horizon = row.get("horizon_days")
        created = row.get("created_at", "")

        # Calculate maturity date
        maturity_str = "[dim]N/A[/dim]"
        days_left_str = ""
        if created and horizon:
            try:
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try:
                        dt = datetime.strptime(created, fmt)
                        maturity = dt + timedelta(days=horizon)
                        days_left = (maturity - now).days
                        maturity_str = maturity.strftime("%Y-%m-%d")
                        if days_left > 0:
                            days_left_str = f" ({days_left}d left)"
                        else:
                            days_left_str = " [yellow](overdue)[/yellow]"
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        # Format return
        def fmt_return(val):
            if val is None:
                return "[dim]N/A[/dim]"
            color = "green" if val >= 0 else "red"
            return f"[{color}]{val:+.4f}[/{color}]"

        action = row.get("action", "N/A")
        action_color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(
            action.upper() if action else "", "white"
        )

        prob_up = row.get("probability_up")

        table.add_row(
            str(i),
            created,
            f"{horizon}d" if horizon else "[dim]N/A[/dim]",
            f"{maturity_str}{days_left_str}",
            f"[{action_color}]{action}[/{action_color}]",
            f"{prob_up:.2f}" if prob_up is not None else "[dim]N/A[/dim]",
            row.get("model_source", "[dim]N/A[/dim]"),
        )

    console.print(table)
    console.print()


def display_memory_stats(storage: Storage, symbol: str) -> None:
    """Display memory agent calibration stats."""
    memory_agent = MemoryAgent(verbose=False)
    result = memory_agent.recall(symbol, storage)

    memory = result.get("memory", {})
    count = memory.get("prediction_count", 0)

    # Summary panel
    status = result.get("status", "unknown")
    status_color = {"success": "green", "no_history": "yellow", "error": "red"}.get(status, "white")

    stats_table = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 2),
        expand=True,
    )
    stats_table.add_column("Metric", style="bold cyan", width=30)
    stats_table.add_column("Value", style="white")

    stats_table.add_row("Status", f"[{status_color}]{status}[/{status_color}]")
    stats_table.add_row("Tracked Predictions", str(count))

    if count > 0:
        accuracy = memory.get("directional_accuracy", 0)
        acc_color = "green" if accuracy >= 0.6 else ("yellow" if accuracy >= 0.5 else "red")
        stats_table.add_row("Directional Accuracy", f"[{acc_color}]{accuracy:.1%}[/{acc_color}]")

        stats_table.add_row("Avg Realized Return", f"{memory.get('avg_realized_return', 0):+.6f}")

        # Regime performance
        regime_perf = memory.get("regime_performance", {})
        if regime_perf:
            regime_parts = []
            for regime, stats in regime_perf.items():
                regime_parts.append(f"{regime}: {stats['accuracy']:.0%} ({stats['count']})")
            stats_table.add_row("Regime Performance", " | ".join(regime_parts))

        # Last prediction
        last_pred = memory.get("last_prediction")
        if last_pred:
            lp_correct = last_pred.get("correct_direction", False)
            lp_mark = "[green]✓[/green]" if lp_correct else "[red]✗[/red]"
            stats_table.add_row(
                "Last Prediction",
                f"{lp_mark} {last_pred.get('date', 'N/A')} | "
                f"{last_pred.get('action', 'N/A')} | "
                f"prob_up={last_pred.get('probability_up', 'N/A')} → "
                f"real={last_pred.get('realized_return', 0):+.4f}",
            )

    stats_table.add_row("", "")
    tr = result.get("track_record_factor", "N/A")
    stats_table.add_row(
        "Track Record Factor",
        f"[bold]{tr}[/bold]  [dim](applied to RiskAgent position sizing)[/dim]",
    )

    console.print(
        Panel(
            stats_table,
            title=f"Memory & Calibration Stats — {symbol}",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()

    # Summary text
    console.print(f"  [bold]Summary:[/bold] {result.get('summary', 'N/A')}\n")


def clear_memory(storage: Storage, symbol: str | None = None) -> None:
    """Clear memory records for a specific stock or all stocks."""
    target = symbol.upper().strip() if symbol else "ALL stocks"

    console.print()
    console.print(
        Panel(
            f"[bold red]⚠ WARNING: This will permanently delete memory records for {target}.[/bold red]\n\n"
            "The following data will be removed:\n"
            "  • All prediction records (predictions table)\n"
            "  • All realized outcome records (realized_outcomes table)\n\n"
            "[dim]This action cannot be undone.[/dim]",
            title="Clear Memory",
            border_style="red",
            padding=(1, 2),
        )
    )

    confirm = console.input(f"\n[bold red]Type 'yes' to confirm deletion for {target}: [/bold red]").strip().lower()
    if confirm != "yes":
        console.print("[yellow]Cancelled. No records were deleted.[/yellow]")
        return

    counts = storage.clear_memory(symbol)

    # Display results
    result_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        border_style="green",
    )
    result_table.add_column("Table", style="cyan", width=25)
    result_table.add_column("Rows Deleted", style="white", width=15, justify="center")

    total_deleted = 0
    for table_name, count in counts.items():
        result_table.add_row(table_name, str(count))
        total_deleted += count

    console.print()
    console.print(result_table)
    console.print()

    if total_deleted > 0:
        console.print(f"  [green]✓[/green] Successfully deleted [bold]{total_deleted}[/bold] records for {target}.")
    else:
        console.print(f"  [dim]No records found for {target}. Nothing was deleted.[/dim]")
    console.print()


def inspect_symbol(storage: Storage, symbol: str) -> None:
    """Run full inspection for a single stock symbol."""
    symbol = symbol.upper().strip()

    console.print()
    console.print(Rule(f"Memory Inspection: {symbol}", style="bold cyan"))
    console.print()

    # 1. Memory & Calibration Stats
    display_memory_stats(storage, symbol)

    # 2. Tracked Predictions (matured, with outcomes)
    display_tracked_predictions(storage, symbol)

    # 3. Pending Predictions (not yet matured)
    display_pending_predictions(storage, symbol)


def display_overview(storage: Storage) -> None:
    """Display an overview of all stocks with prediction data."""
    symbols = get_all_symbols(storage)

    if not symbols:
        console.print("[yellow]No prediction data found in the database.[/yellow]")
        return

    console.print()
    console.print(Rule("All Stocks Overview", style="bold cyan"))
    console.print()

    table = Table(
        title="Stocks with Prediction Data",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        border_style="cyan",
        expand=True,
    )
    table.add_column("Symbol", style="bold cyan", width=10, justify="center")
    table.add_column("Total Predictions", width=18, justify="center")
    table.add_column("Tracked", width=10, justify="center")
    table.add_column("Accuracy", width=12, justify="center")
    table.add_column("Avg Realized", width=14, justify="center")
    table.add_column("Track Record", width=12, justify="center")

    memory_agent = MemoryAgent(verbose=False)

    for sym in symbols:
        all_preds = storage.get_recent_predictions(sym, limit=1000)
        tracked = storage.get_tracked_predictions(sym)
        mem_result = memory_agent.recall(sym, storage)
        memory = mem_result.get("memory", {})

        count = memory.get("prediction_count", 0)
        accuracy = memory.get("directional_accuracy")
        avg_realized = memory.get("avg_realized_return")

        if accuracy is not None:
            acc_color = "green" if accuracy >= 0.6 else ("yellow" if accuracy >= 0.5 else "red")
            acc_str = f"[{acc_color}]{accuracy:.1%}[/{acc_color}]"
        else:
            acc_str = "[dim]N/A[/dim]"

        if avg_realized is not None:
            avg_str = f"{avg_realized:+.4f}"
        else:
            avg_str = "[dim]N/A[/dim]"

        table.add_row(
            sym,
            str(len(all_preds)),
            str(len(tracked)),
            acc_str,
            avg_str,
            str(mem_result.get("track_record_factor", "N/A")),
        )

    console.print(table)
    console.print()


def main():
    """Entry point for the memory inspector."""
    load_dotenv()

    console.print()
    console.print(
        Panel(
            "[bold cyan]Memory Inspector[/bold cyan]\n"
            "[dim]View prediction history, realized outcomes, and calibration data[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    storage = Storage()

    args = sys.argv[1:]

    if not args:
        # Interactive mode
        symbols = get_all_symbols(storage)
        if symbols:
            console.print(f"  [dim]Stocks with data: {', '.join(symbols)}[/dim]\n")

        console.print(
            Panel(
                "[bold]Enter stock symbol(s) to inspect[/bold]\n"
                "[dim]Separate multiple symbols with spaces, or type 'all' for overview[/dim]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        user_input = console.input("[bold cyan]> [/bold cyan]").strip()

        if not user_input:
            console.print("[yellow]No symbol provided. Exiting.[/yellow]")
            return

        if user_input.lower() == "all":
            args = ["--all"]
        else:
            args = user_input.split()

    if "--clear-all" in args:
        clear_memory(storage, symbol=None)
    elif "--clear" in args:
        # Get symbols after --clear flag
        clear_idx = args.index("--clear")
        clear_symbols = [s for s in args[clear_idx + 1:] if not s.startswith("--")]
        if not clear_symbols:
            console.print("[yellow]Please specify a stock symbol to clear, e.g.: --clear AAPL[/yellow]")
            console.print("[dim]Or use --clear-all to clear all records.[/dim]")
        else:
            for sym in clear_symbols:
                clear_memory(storage, symbol=sym)
    elif "--all" in args:
        display_overview(storage)
        # Also show details for each
        symbols = get_all_symbols(storage)
        for sym in symbols:
            inspect_symbol(storage, sym)
    else:
        for symbol in args:
            inspect_symbol(storage, symbol)

    console.print(
        Panel(
            "[bold green]Inspection complete.[/bold green]",
            border_style="green",
            padding=(0, 2),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        import traceback
        console.print_exception()
