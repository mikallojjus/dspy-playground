"""
Compare training results across model versions.

This script loads all training results files and displays them in a comparison table,
sorted by performance metrics.

Usage:
    # Compare all claim extractor models
    python -m src.cli.compare_models --model-type claim_extractor

    # Compare all models (all types)
    python -m src.cli.compare_models

    # Show only top 5 models
    python -m src.cli.compare_models --model-type claim_extractor --top-n 5

    # Sort by specific metric
    python -m src.cli.compare_models --model-type entailment_validator --metric accuracy

    # Filter by date (since YYYY-MM-DD)
    python -m src.cli.compare_models --since 2025-01-01

Examples:
    uv run python -m src.cli.compare_models --model-type claim_extractor
    uv run python -m src.cli.compare_models --model-type entailment_validator --top-n 3
    uv run python -m src.cli.compare_models --metric score
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import box

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.training.training_utils import load_all_results, format_duration


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_timestamp[:16]  # Fallback to first 16 chars


def get_sort_key(result: dict, metric: str) -> float:
    """Get sort key for a result based on metric name."""
    # Try to get metric from optimized section
    if "optimized" in result and isinstance(result["optimized"], dict):
        if metric in result["optimized"]:
            return float(result["optimized"][metric])

    # Try improvement section (handle both dict and float formats)
    if "improvement" in result:
        improvement = result["improvement"]
        if isinstance(improvement, dict) and metric in improvement:
            return float(improvement[metric])
        elif isinstance(improvement, (int, float)):
            # Old format: improvement is a single float
            if metric in ["score", "improvement"]:
                return float(improvement)

    # Default to optimized score
    if "optimized" in result:
        optimized = result["optimized"]
        if isinstance(optimized, dict) and "score" in optimized:
            return float(optimized["score"])
        elif isinstance(optimized, (int, float)):
            # Old format: optimized is a single float
            return float(optimized)

    return 0.0


def create_comparison_table(
    results: list[dict],
    model_type: Optional[str] = None,
    metric: str = "score",
    top_n: Optional[int] = None,
) -> Table:
    """Create Rich table comparing model results."""
    # Determine title based on model type
    if model_type:
        title_model_type = model_type.replace("_", " ").title()
        title = f"Model Comparison: {title_model_type}"
    else:
        title = "Model Comparison: All Types"

    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_style="bold magenta",
    )

    # Add columns
    table.add_column("Rank", justify="right", style="dim", width=4)
    table.add_column("Timestamp", style="cyan", width=16)
    table.add_column("Model Name", style="white", no_wrap=True, width=35)
    table.add_column("Optimizer", style="yellow", width=12)

    # Add metric-specific columns based on model type
    if not model_type or model_type == "claim_extractor":
        table.add_column("Quality Score", justify="center", width=13)
        table.add_column("Improvement", justify="center", style="green", width=11)
    elif model_type == "entailment_validator":
        table.add_column("Accuracy", justify="center", width=10)
        table.add_column("FP Rate", justify="center", width=9)
        table.add_column("Improvement", justify="center", style="green", width=11)
    elif model_type == "ad_classifier":
        table.add_column("Accuracy", justify="center", width=10)
        table.add_column("Improvement", justify="center", style="green", width=11)
    else:
        # Generic columns for mixed types or unknown
        table.add_column("Score", justify="center", width=10)
        table.add_column("Improvement", justify="center", style="green", width=11)

    table.add_column("Train Time", justify="right", width=10)
    table.add_column("Demos", justify="right", width=5)
    table.add_column("Target Met", justify="center", width=10)

    # Sort results by metric (descending)
    sorted_results = sorted(results, key=lambda r: get_sort_key(r, metric), reverse=True)

    # Limit to top N if specified
    if top_n:
        sorted_results = sorted_results[:top_n]

    # Add rows
    for rank, result in enumerate(sorted_results, 1):
        # Basic info
        timestamp = format_timestamp(result.get("timestamp", ""))
        model_name = result.get("model_name", "unknown")
        optimizer = result.get("optimizer", "unknown")

        # Metrics (handle both dict and float formats from old files)
        baseline_raw = result.get("baseline", {})
        optimized_raw = result.get("optimized", {})
        improvement_raw = result.get("improvement", {})

        # Convert to dict if they're floats (old format)
        baseline = baseline_raw if isinstance(baseline_raw, dict) else {"score": baseline_raw}
        optimized = optimized_raw if isinstance(optimized_raw, dict) else {"score": optimized_raw}
        improvement = improvement_raw if isinstance(improvement_raw, dict) else {"score": improvement_raw}

        # Training info
        train_time = format_duration(result.get("training_time_seconds", 0))
        few_shot_demos = result.get("few_shot_demos", 0)
        targets_met = result.get("targets_met", False)
        target_icon = "✓" if targets_met else "✗"

        # Determine row style (highlight best model)
        row_style = "bold green" if rank == 1 else None

        # Build row based on model type
        result_model_type = result.get("model_type", "")

        if result_model_type == "claim_extractor":
            quality_score = optimized.get("quality_score", optimized.get("score", 0))
            quality_improvement = improvement.get("quality_score", improvement.get("score", 0))
            quality_str = f"{baseline.get('quality_score', 0):.3f} → {quality_score:.3f}"
            improvement_str = f"+{quality_improvement:.3f}" if quality_improvement >= 0 else f"{quality_improvement:.3f}"

            table.add_row(
                str(rank),
                timestamp,
                model_name,
                optimizer,
                quality_str,
                improvement_str,
                train_time,
                str(few_shot_demos),
                target_icon,
                style=row_style,
            )

        elif result_model_type == "entailment_validator":
            accuracy = optimized.get("accuracy", 0)
            fp_rate = optimized.get("false_positive_rate", 0)
            accuracy_improvement = improvement.get("accuracy", 0)
            baseline_accuracy = baseline.get("accuracy", 0)
            baseline_fp = baseline.get("false_positive_rate", 0)

            accuracy_str = f"{baseline_accuracy:.1%} → {accuracy:.1%}"
            fp_str = f"{baseline_fp:.1%} → {fp_rate:.1%}"
            improvement_str = f"+{accuracy_improvement:.1%}" if accuracy_improvement >= 0 else f"{accuracy_improvement:.1%}"

            table.add_row(
                str(rank),
                timestamp,
                model_name,
                optimizer,
                accuracy_str,
                fp_str,
                improvement_str,
                train_time,
                str(few_shot_demos),
                target_icon,
                style=row_style,
            )

        elif result_model_type == "ad_classifier":
            accuracy = optimized.get("accuracy", 0)
            accuracy_improvement = improvement.get("accuracy", 0)
            baseline_accuracy = baseline.get("accuracy", 0)

            accuracy_str = f"{baseline_accuracy:.3f} → {accuracy:.3f}"
            improvement_str = f"+{accuracy_improvement:.3f}" if accuracy_improvement >= 0 else f"{accuracy_improvement:.3f}"

            table.add_row(
                str(rank),
                timestamp,
                model_name,
                optimizer,
                accuracy_str,
                improvement_str,
                train_time,
                str(few_shot_demos),
                target_icon,
                style=row_style,
            )

        else:
            # Generic row for unknown types
            score = optimized.get("score", 0)
            score_improvement = improvement.get("score", 0)
            baseline_score = baseline.get("score", 0)

            score_str = f"{baseline_score:.3f} → {score:.3f}"
            improvement_str = f"+{score_improvement:.3f}" if score_improvement >= 0 else f"{score_improvement:.3f}"

            table.add_row(
                str(rank),
                timestamp,
                model_name,
                optimizer,
                score_str,
                improvement_str,
                train_time,
                str(few_shot_demos),
                target_icon,
                style=row_style,
            )

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Compare training results across model versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-type",
        choices=["claim_extractor", "entailment_validator", "ad_classifier"],
        help="Filter by model type",
    )

    parser.add_argument(
        "--metric",
        default="score",
        help="Metric to sort by (default: score). Options: score, accuracy, improvement",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        help="Show only top N models",
    )

    parser.add_argument(
        "--since",
        help="Show only models trained since date (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    console = Console()

    # Load results from models/ directory
    models_dir = Path("models")
    if not models_dir.exists():
        console.print("[red]Error: models/ directory not found[/red]")
        console.print("No training results available yet. Train a model first.")
        return 1

    console.print(f"[cyan]Loading results from {models_dir}...[/cyan]")
    results = load_all_results(models_dir, model_type=args.model_type)

    if not results:
        if args.model_type:
            console.print(f"[yellow]No results found for model type: {args.model_type}[/yellow]")
        else:
            console.print("[yellow]No training results found[/yellow]")
        console.print("\nTrain a model first:")
        console.print("  uv run python -m src.training.train_claim_extractor")
        console.print("  uv run python -m src.training.train_entailment_validator")
        console.print("  uv run python -m src.training.train_ad_classifier")
        return 1

    # Filter by date if specified
    if args.since:
        try:
            since_date = datetime.fromisoformat(args.since)
            results = [
                r
                for r in results
                if datetime.fromisoformat(r.get("timestamp", "")) >= since_date
            ]
            if not results:
                console.print(f"[yellow]No results found since {args.since}[/yellow]")
                return 1
        except ValueError:
            console.print(f"[red]Error: Invalid date format '{args.since}'. Use YYYY-MM-DD[/red]")
            return 1

    console.print(f"[green]Found {len(results)} model(s)[/green]")
    console.print()

    # Create and display comparison table
    table = create_comparison_table(
        results,
        model_type=args.model_type,
        metric=args.metric,
        top_n=args.top_n,
    )
    console.print(table)

    # Print best model info
    if results:
        best_result = sorted(results, key=lambda r: get_sort_key(r, args.metric), reverse=True)[0]
        console.print()
        console.print("[bold green]Best Model:[/bold green]")
        console.print(f"  Path: {best_result.get('model_path', 'unknown')}")
        console.print(f"  Timestamp: {best_result.get('timestamp', 'unknown')}")

        # Show how to use this model
        console.print()
        console.print("[bold cyan]To use this model:[/bold cyan]")
        console.print(f"  1. Update src/config/settings.py")
        model_type = best_result.get("model_type", "")
        if model_type == "claim_extractor":
            console.print(f"     claim_extractor_model_path = \"{best_result.get('model_path', '')}\"")
        elif model_type == "entailment_validator":
            console.print(f"     entailment_validator_model_path = \"{best_result.get('model_path', '')}\"")
        elif model_type == "ad_classifier":
            console.print(f"     ad_classifier_model_path = \"{best_result.get('model_path', '')}\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())
