"""
CLI entry point for processing podcast episodes.

Processes episodes through the claim extraction pipeline with intelligent
filtering, progress tracking, and error handling.

Usage:
    # Process all podcasts, all unprocessed episodes
    uv run python -m src.cli.process_episodes

    # Process specific podcast, limit to 5 episodes
    uv run python -m src.cli.process_episodes --podcast-id 123 --limit 5

    # Continue processing on errors
    uv run python -m src.cli.process_episodes --continue-on-error

    # Reprocess all episodes (ignore existing claims)
    uv run python -m src.cli.process_episodes --force

Examples:
    # Process latest 10 episodes from Bankless podcast
    uv run python -m src.cli.process_episodes --podcast-id 1 --limit 10

    # Process all unprocessed episodes, continue on errors
    uv run python -m src.cli.process_episodes --continue-on-error
"""

import argparse
import asyncio
import sys
import time
from typing import List, Tuple, Optional, cast
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich import box

from src.pipeline.extraction_pipeline import ExtractionPipeline, PipelineResult
from src.cli.episode_query import EpisodeQueryService
from src.database.models import PodcastEpisode
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class ProcessingStats:
    """Overall processing statistics."""
    total_episodes: int = 0
    successful: int = 0
    failed: int = 0
    total_claims: int = 0
    total_quotes: int = 0
    total_duplicates: int = 0
    total_time: float = 0.0
    errors: List[Tuple[int, str, str]] = field(default_factory=list)  # (episode_id, name, error)

    @property
    def avg_time_per_episode(self) -> float:
        """Average processing time per successful episode."""
        if self.successful == 0:
            return 0.0
        return self.total_time / self.successful


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process podcast episodes through claim extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all podcasts, all unprocessed episodes
  %(prog)s

  # Process specific podcast, limit to 5 episodes
  %(prog)s --podcast-id 123 --limit 5

  # Continue processing on errors
  %(prog)s --continue-on-error

  # Reprocess all episodes (ignore existing claims)
  %(prog)s --force

  # Dry run (show what would be processed)
  %(prog)s --dry-run
        """
    )

    parser.add_argument(
        "--podcast-id",
        type=int,
        default=None,
        help="Process episodes from specific podcast ID (default: all podcasts)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of episodes to process (default: 0 = all)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess episodes that already have claims (default: skip processed)"
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing if an episode fails (default: stop on error)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )

    return parser.parse_args()


def display_summary(
    episodes: List[PodcastEpisode],
    podcast_id: Optional[int],
    force: bool
):
    """Display processing summary before starting."""
    console.print()

    # Create summary table
    table = Table(title="Processing Summary", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Podcast ID", str(podcast_id) if podcast_id else "All podcasts")
    table.add_row("Episodes to process", str(len(episodes)))
    table.add_row("Mode", "Reprocess all" if force else "Skip processed")
    table.add_row("Order", "Newest first")

    console.print(table)
    console.print()

    if len(episodes) == 0:
        console.print("[yellow]⚠  No episodes to process![/yellow]")
        return False

    # Show first few episodes
    if len(episodes) > 0:
        console.print("[bold]Episodes to process (first 5):[/bold]")
        for i, episode in enumerate(episodes[:5], 1):
            date_str = episode.published_at.strftime("%Y-%m-%d") if episode.published_at is not None else "No date"
            console.print(f"  {i}. [cyan]Episode {episode.id}[/cyan]: {episode.name} ({date_str})")

        if len(episodes) > 5:
            console.print(f"  ... and {len(episodes) - 5} more")
        console.print()

    return True


def display_episode_result(
    episode: PodcastEpisode,
    result: PipelineResult,
    index: int,
    total: int
):
    """Display results for a single processed episode."""
    console.print()
    console.print(
        f"[bold cyan]Episode {index}/{total}[/bold cyan]: "
        f"{episode.name} (ID: {episode.id})"
    )

    # Create results table
    table = Table(box=box.SIMPLE)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    # Claims
    table.add_row(
        "Claims",
        f"{result.stats.claims_extracted} → "
        f"{result.stats.claims_after_dedup} → "
        f"{result.stats.claims_saved} saved"
    )

    # Quotes
    table.add_row(
        "Quotes",
        f"{result.stats.quotes_before_entailment} → "
        f"{result.stats.quotes_after_entailment} (entailment) → "
        f"{result.stats.quotes_saved} saved"
    )

    # Entailment filtering
    if result.stats.entailment_filtered_quotes > 0:
        table.add_row(
            "Entailment filtered",
            f"{result.stats.entailment_filtered_quotes} quotes"
        )

    # Database duplicates
    if result.stats.database_duplicates_found > 0:
        table.add_row(
            "Duplicates merged",
            f"{result.stats.database_duplicates_found} claims"
        )

    # Processing time
    table.add_row(
        "Time",
        f"{result.stats.processing_time_seconds:.1f}s"
    )

    console.print(table)


def display_error(
    episode: PodcastEpisode,
    error: Exception,
    index: int,
    total: int
):
    """Display error for a failed episode."""
    console.print()
    console.print(
        f"[bold red]✗ Episode {index}/{total} FAILED[/bold red]: "
        f"{episode.name} (ID: {episode.id})"
    )
    console.print(f"[red]Error: {str(error)}[/red]")


def display_final_stats(stats: ProcessingStats):
    """Display final processing statistics."""
    console.print()
    console.rule("[bold]PROCESSING COMPLETE[/bold]")
    console.print()

    # Summary table
    table = Table(title="Final Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")

    # Episodes
    success_rate = (stats.successful / stats.total_episodes * 100) if stats.total_episodes > 0 else 0
    table.add_row(
        "Episodes processed",
        f"{stats.successful}/{stats.total_episodes} ({success_rate:.0f}%)"
    )

    if stats.failed > 0:
        table.add_row("Failed", f"{stats.failed}", style="red")

    # Claims and quotes
    table.add_row("Claims saved", str(stats.total_claims))
    table.add_row("Quotes saved", str(stats.total_quotes))

    if stats.total_duplicates > 0:
        table.add_row("Duplicates merged", str(stats.total_duplicates))

    # Timing
    table.add_row("Total time", f"{stats.total_time / 60:.1f} minutes")
    table.add_row(
        "Average per episode",
        f"{stats.avg_time_per_episode:.1f}s"
    )

    console.print(table)

    # Failed episodes
    if stats.errors:
        console.print()
        console.print("[bold red]Failed Episodes:[/bold red]")
        for episode_id, name, error in stats.errors:
            console.print(f"  • [red]Episode {episode_id}[/red]: {name}")
            console.print(f"    {error}")

    console.print()


async def process_episodes(
    episodes: List[PodcastEpisode],
    continue_on_error: bool = False
) -> ProcessingStats:
    """
    Process episodes through the pipeline.

    Args:
        episodes: List of episodes to process
        continue_on_error: If True, continue processing on errors

    Returns:
        ProcessingStats with results
    """
    stats = ProcessingStats(total_episodes=len(episodes))
    pipeline = ExtractionPipeline()

    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task(
            "[cyan]Processing episodes...",
            total=len(episodes)
        )

        start_time = time.time()

        for i, episode in enumerate(episodes, 1):
            try:
                # Update progress
                progress.update(
                    task,
                    description=f"[cyan]Processing episode {i}/{len(episodes)}: {episode.name[:40]}..."
                )

                # Process episode
                episode_id = cast(int, episode.id)
                logger.info(f"Processing episode {episode_id}: {episode.name}")
                result = await pipeline.process_episode(episode_id)

                # Update stats
                stats.successful += 1
                stats.total_claims += result.stats.claims_saved
                stats.total_quotes += result.stats.quotes_saved
                stats.total_duplicates += result.stats.database_duplicates_found

                # Display results
                display_episode_result(episode, result, i, len(episodes))

            except Exception as e:
                episode_id = cast(int, episode.id)
                episode_name = cast(str, episode.name)
                logger.error(
                    f"Failed to process episode {episode_id}: {e}",
                    exc_info=True
                )

                stats.failed += 1
                stats.errors.append((episode_id, episode_name, str(e)))

                # Display error
                display_error(episode, e, i, len(episodes))

                # Stop or continue?
                if not continue_on_error:
                    console.print()
                    console.print("[bold red]Stopping due to error (use --continue-on-error to continue)[/bold red]")
                    break

            finally:
                progress.advance(task)

        stats.total_time = time.time() - start_time

    return stats


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Display header
    console.rule("[bold blue]Podcast Episode Processing[/bold blue]")
    console.print()

    # Initialize services
    query_service = EpisodeQueryService()

    # Get episodes to process
    console.print("[cyan]Querying episodes...[/cyan]")
    episodes = query_service.get_episodes_to_process(
        podcast_id=args.podcast_id,
        limit=args.limit,
        force=args.force
    )

    # Display summary
    should_continue = display_summary(episodes, args.podcast_id, args.force)
    if not should_continue:
        return 0

    # Dry run?
    if args.dry_run:
        console.print("[yellow]Dry run mode - no processing will occur[/yellow]")
        console.print()
        return 0

    # Confirm with user
    if len(episodes) > 0:
        console.print("[yellow]Press Ctrl+C to cancel, or Enter to continue...[/yellow]", end="")
        try:
            input()
        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Cancelled by user[/yellow]")
            return 1

    # Process episodes
    stats = await process_episodes(episodes, args.continue_on_error)

    # Display final statistics
    display_final_stats(stats)

    # Return exit code
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print()
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
