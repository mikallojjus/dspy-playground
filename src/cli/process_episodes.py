"""
CLI entry point for processing podcast episodes.

Processes episodes through the claim extraction pipeline with intelligent
filtering, progress tracking, and error handling.

Usage:
    # Process all podcasts, all unprocessed episodes
    uv run python -m src.cli.process_episodes

    # Process specific episode by ID
    uv run python -m src.cli.process_episodes --episode-id 123

    # Process specific podcast, limit to 5 episodes
    uv run python -m src.cli.process_episodes --podcast-id 9 --limit 5

    # Continue processing on errors
    uv run python -m src.cli.process_episodes --continue-on-error

    # Reprocess all episodes (ignore existing claims)
    uv run python -m src.cli.process_episodes --force

Examples:
    # Process a specific episode
    uv run python -m src.cli.process_episodes --episode-id 456

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
from rich.text import Text
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
from src.infrastructure.logger import get_logger, get_shared_console

logger = get_logger(__name__)
console = get_shared_console()  # Use shared console to coordinate with logging


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

  # Process a specific episode by ID
  %(prog)s --episode-id 456

  # Process specific podcasts, limit to 100 episodes per podcast
  %(prog)s --podcast-ids 1,2,3 --limit 100

  # Continue processing on errors
  %(prog)s --continue-on-error

  # Reprocess all episodes (ignore existing claims)
  %(prog)s --force

  # Dry run (show what would be processed)
  %(prog)s --dry-run
        """
    )

    parser.add_argument(
        "--podcast-ids",
        type=str,
        default=None,
        help="Process episodes from specific podcast IDs, comma-separated (e.g., '1,2,3') (default: all podcasts)"
    )

    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Process a specific episode by ID (overrides --podcast-id and --limit)"
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

    args = parser.parse_args()

    # Parse comma-separated podcast IDs into a list
    if args.podcast_ids:
        try:
            args.podcast_ids = [int(pid.strip()) for pid in args.podcast_ids.split(',')]
        except ValueError:
            parser.error("--podcast-ids must be a comma-separated list of integers")
    else:
        args.podcast_ids = None

    return args


def display_summary(
    episodes: List[PodcastEpisode],
    podcast_ids: Optional[List[int]],
    episode_id: Optional[int],
    force: bool
):
    """Display processing summary before starting."""
    console.print()

    # Create summary table
    table = Table(title="Processing Summary", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    if episode_id is not None:
        table.add_row("Episode ID", str(episode_id))
    else:
        if podcast_ids:
            table.add_row("Podcast IDs", ", ".join(str(pid) for pid in podcast_ids))
        else:
            table.add_row("Podcast IDs", "All podcasts")
        table.add_row("Order", "Newest first")

    table.add_row("Episodes to process", str(len(episodes)))
    table.add_row("Mode", "Reprocess all" if force else "Skip processed")

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


def _format_filter_row(before: int, after: int, reason: str = "") -> str:
    """Format a filter row showing before → after with optional reason."""
    filtered = before - after
    if filtered > 0:
        suffix = f" (-{filtered}: {reason})" if reason else f" (-{filtered})"
        return f"{before} → {after}{suffix}"
    return f"{before}"


def _format_claims_extraction_section(stats) -> Table:
    """Create claims extraction section."""
    table = Table(title="CLAIMS EXTRACTION", box=box.ROUNDED, show_header=False, title_style="bold cyan")
    table.add_column("Metric", style="dim", width=24)
    table.add_column("Value")

    table.add_row("Extracted", f"{stats.claims_extracted} claims")
    if stats.specificity_filtered_count > 0:
        table.add_row("Specificity Filter", _format_filter_row(
            stats.claims_extracted,
            stats.claims_after_specificity_filter,
            "too vague"
        ))
    if stats.ad_claims_filtered > 0:
        table.add_row("Ad Filter", _format_filter_row(
            stats.claims_after_specificity_filter,
            stats.claims_after_ad_filter,
            "advertisements"
        ))

    return table


def _format_claims_validation_section(stats) -> Table:
    """Create claims validation section."""
    table = Table(title="CLAIMS VALIDATION", box=box.ROUNDED, show_header=False, title_style="bold cyan")
    table.add_column("Metric", style="dim", width=24)
    table.add_column("Value")

    # Step 1: Deduplication
    dedup_filtered = (stats.claims_after_ad_filter or stats.claims_after_specificity_filter) - stats.claims_after_dedup
    if dedup_filtered > 0:
        table.add_row("Deduplication", _format_filter_row(
            stats.claims_after_ad_filter or stats.claims_after_specificity_filter,
            stats.claims_after_dedup,
            "duplicates"
        ))

    # Step 2: Quote Requirement (now happens BEFORE confidence filter)
    if stats.claims_without_quotes_count > 0:
        claims_before_quote_filter = stats.claims_after_dedup
        claims_after_quote_filter = claims_before_quote_filter - stats.claims_without_quotes_count
        table.add_row("Quote Requirement", _format_filter_row(
            claims_before_quote_filter,
            claims_after_quote_filter,
            "no quotes"
        ))

    # Step 3: Confidence Filter (now happens AFTER quote requirement)
    if stats.low_confidence_filtered_count > 0:
        # Calculate claims before confidence filter
        claims_before_conf = stats.claims_after_dedup - stats.claims_without_quotes_count
        table.add_row("Confidence Filter", _format_filter_row(
            claims_before_conf,
            stats.claims_after_confidence_filter,
            "low confidence"
        ))

    return table


def _format_claims_persistence_section(stats) -> Table:
    """Create claims persistence section."""
    table = Table(title="CLAIMS PERSISTENCE", box=box.ROUNDED, show_header=False, title_style="bold cyan")
    table.add_column("Metric", style="dim", width=24)
    table.add_column("Value")

    table.add_row("Database Check", f"{stats.claims_with_quotes} → {stats.claims_saved + stats.claims_merged_to_existing} unique")
    if stats.claims_merged_to_existing > 0:
        table.add_row("Merged to Existing", f"{stats.claims_merged_to_existing} claims")
    table.add_row("✓ Saved", f"[bold green]{stats.claims_saved}[/bold green] new claims")

    return table


def _format_quotes_section(stats) -> Table:
    """Create quotes section."""
    table = Table(title="QUOTES EXTRACTION & VALIDATION", box=box.ROUNDED, show_header=False, title_style="bold cyan")
    table.add_column("Metric", style="dim", width=24)
    table.add_column("Value")

    if stats.quotes_initial_candidates > 0:
        table.add_row("Candidates Found", f"{stats.quotes_initial_candidates} initial matches")

        if stats.question_filtered_count > 0:
            table.add_row("Question Filter", _format_filter_row(
                stats.quotes_initial_candidates,
                stats.quotes_after_question_filter,
                "rhetorical"
            ))

        if stats.quality_filtered_count > 0:
            table.add_row("Quality Filter", _format_filter_row(
                stats.quotes_after_question_filter,
                stats.quotes_after_quality_filter,
                "too short/ads"
            ))

        if stats.relevance_filtered_count > 0:
            table.add_row("Relevance Filter", _format_filter_row(
                stats.quotes_after_quality_filter,
                stats.quotes_after_relevance_filter,
                f"below {0.85}"
            ))

    if stats.duplicate_quotes_count > 0:
        table.add_row("Deduplication", _format_filter_row(
            stats.quotes_before_dedup,
            stats.quotes_after_dedup,
            "duplicates"
        ))

    if stats.entailment_filtered_quotes > 0:
        table.add_row("Entailment Check", _format_filter_row(
            stats.quotes_before_entailment,
            stats.quotes_after_entailment,
            "not SUPPORTS"
        ))

    table.add_row("✓ Saved", f"[bold green]{stats.quotes_saved}[/bold green] quotes")

    return table


def _format_filtered_samples_section(filtered_items) -> Table:
    """Create filtered samples section with full text and enhanced metadata."""
    table = Table(title="FILTERED SAMPLES", box=box.ROUNDED, show_header=False, title_style="bold yellow", width=100)
    table.add_column("", style="dim", no_wrap=False)

    any_samples = False

    # Low confidence claims (with quote counts)
    if filtered_items.low_confidence_claims:
        any_samples = True
        table.add_row(f"[yellow]Low confidence claims ({len(filtered_items.low_confidence_claims)}):[/yellow]")
        table.add_row("")
        for idx, item in enumerate(filtered_items.low_confidence_claims[:3], 1):
            # Full claim text (word-wrapped automatically by rich)
            table.add_row(f" {idx}. {item.text}")
            # Enhanced metadata with quote count
            quote_count = item.metadata.get("quote_count", 0)
            quote_info = f"{quote_count} quote{'s' if quote_count != 1 else ''}"
            table.add_row(f"    [dim]→ {item.reason} ({quote_info})[/dim]")
            table.add_row("")

    # Claims without quotes
    if filtered_items.claims_without_quotes:
        any_samples = True
        if filtered_items.low_confidence_claims:
            table.add_row("")  # Extra spacing between categories
        table.add_row(f"[yellow]Claims without quotes ({len(filtered_items.claims_without_quotes)}):[/yellow]")
        table.add_row("")
        for idx, item in enumerate(filtered_items.claims_without_quotes[:3], 1):
            table.add_row(f" {idx}. {item.text}")
            table.add_row(f"    [dim]→ {item.reason}[/dim]")
            table.add_row("")

    # Vague claims (specificity filtered)
    if filtered_items.specificity_filtered:
        any_samples = True
        if filtered_items.low_confidence_claims or filtered_items.claims_without_quotes:
            table.add_row("")
        table.add_row(f"[yellow]Vague claims ({len(filtered_items.specificity_filtered)}):[/yellow]")
        table.add_row("")
        for idx, item in enumerate(filtered_items.specificity_filtered[:3], 1):
            table.add_row(f" {idx}. {item.text}")
            table.add_row(f"    [dim]→ {item.reason}[/dim]")
            table.add_row("")

    # Advertisement claims
    if filtered_items.ad_claims:
        any_samples = True
        if filtered_items.specificity_filtered or filtered_items.low_confidence_claims or filtered_items.claims_without_quotes:
            table.add_row("")
        table.add_row(f"[yellow]Advertisement claims ({len(filtered_items.ad_claims)}):[/yellow]")
        table.add_row("")
        for idx, item in enumerate(filtered_items.ad_claims[:3], 1):
            table.add_row(f" {idx}. {item.text}")
            table.add_row(f"    [dim]→ {item.reason}[/dim]")
            table.add_row("")

    # Entailment filtered quotes (non-SUPPORTS)
    if filtered_items.entailment_filtered_quotes:
        any_samples = True
        if any([filtered_items.specificity_filtered, filtered_items.ad_claims,
                filtered_items.low_confidence_claims, filtered_items.claims_without_quotes]):
            table.add_row("")
        table.add_row(f"[yellow]Non-supporting quotes ({len(filtered_items.entailment_filtered_quotes)}):[/yellow]")
        table.add_row("")
        for idx, item in enumerate(filtered_items.entailment_filtered_quotes[:3], 1):
            # Show first 120 chars for quotes (they can be long)
            quote_text = item.text if len(item.text) <= 120 else item.text[:117] + "..."
            table.add_row(f" {idx}. \"{quote_text}\"")
            table.add_row(f"    [dim]→ {item.reason}[/dim]")
            table.add_row("")

    if not any_samples:
        table.add_row("[dim]No filtered items to display[/dim]")

    return table


def _format_performance_section(stats) -> Table:
    """Create performance section."""
    table = Table(title="PERFORMANCE", box=box.ROUNDED, show_header=False, title_style="bold magenta")
    table.add_column("Stage", style="dim", width=24)
    table.add_column("Time")

    if stats.stage_timings:
        for stage, duration in stats.stage_timings.items():
            # Format stage name
            stage_name = stage.replace("_", " ").title()
            table.add_row(stage_name, f"{duration:.1f}s")

        table.add_row("─" * 24, "─" * 10)

    table.add_row("[bold]Total[/bold]", f"[bold]{stats.processing_time_seconds:.1f}s[/bold]")

    return table


def display_episode_result(
    episode: PodcastEpisode,
    result: PipelineResult,
    index: int,
    total: int
):
    """Display results for a single processed episode with detailed breakdown."""
    console.print()
    console.print(
        f"[bold cyan]Episode {index}/{total}[/bold cyan]: "
        f"{episode.name} (ID: {episode.id})"
    )

    # Display all sections
    console.print(_format_claims_extraction_section(result.stats))
    console.print(_format_claims_validation_section(result.stats))
    console.print(_format_claims_persistence_section(result.stats))
    console.print(_format_quotes_section(result.stats))
    console.print(_format_filtered_samples_section(result.stats.filtered_items))
    console.print(_format_performance_section(result.stats))


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

    # If episode_id is specified, process that specific episode
    if args.episode_id is not None:
        try:
            episode = query_service.get_episode_by_id(args.episode_id)
            if episode is None:
                console.print(f"[bold red]Error: Episode {args.episode_id} not found[/bold red]")
                return 1
            episodes = [episode]
        except ValueError as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            return 1
    else:
        episodes = query_service.get_episodes_to_process(
            podcast_ids=args.podcast_ids,
            limit=args.limit,
            force=args.force
        )

        # Filter episodes to only those with transcripts
        total_retrieved = len(episodes)
        episodes_with_transcripts = [ep for ep in episodes if ep.podscribe_transcript is not None]
        episodes_without_transcripts = total_retrieved - len(episodes_with_transcripts)

        # Log statistics
        if total_retrieved > 0:
            console.print()
            console.print(f"[cyan]Retrieved {total_retrieved} episode(s)[/cyan]")
            console.print(f"[green]  ✓ {len(episodes_with_transcripts)} with transcripts (will process)[/green]")
            if episodes_without_transcripts > 0:
                console.print(f"[yellow]  ⊘ {episodes_without_transcripts} without transcripts (skipping)[/yellow]")
            console.print()

        # Update episodes to only those with transcripts
        episodes = episodes_with_transcripts

    # Display summary
    should_continue = display_summary(episodes, args.podcast_ids, args.episode_id, args.force)
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
