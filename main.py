"""
Multi-Agent Financial Advisory System
CLI entry point powered by the shared orchestrator.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from orchestrator import run_full_analysis


def _cli_progress(step: int, total: int, message: str) -> None:
    print(f"\n[{step}/{total}] {message}")


def main():
    """Run the financial advisory pipeline from the terminal."""
    load_dotenv()
    print("Multi-Agent Financial Advisory System")
    print("=" * 60)

    stock_symbol = input("\nEnter stock symbol to analyze (e.g., AAPL, GOOGL, MSFT): ").strip().upper()
    if not stock_symbol:
        print("No stock symbol provided. Exiting.")
        return

    print(f"\nAnalyzing {stock_symbol}...")
    verbose_mode = os.getenv("VERBOSE", "false").lower() == "true"

    result = run_full_analysis(
        stock_symbol,
        verbose=verbose_mode,
        persist=True,
        save_report=True,
        progress_callback=_cli_progress,
    )
    if result.get("status") != "success":
        print(f"\nError: {result.get('error', 'Unknown pipeline error')}")
        return

    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(result.get("final_report", "No report generated."))

    output_file = result.get("output_file")
    if output_file:
        print(f"\nFull report saved to: {output_file}")

    if result.get("persisted"):
        print("✓ Stored recommendation, pair signals, and forecast snapshot")
    else:
        print("Storage write skipped or unavailable.")

    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback

        traceback.print_exc()
