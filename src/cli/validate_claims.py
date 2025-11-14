"""
CLI tool for validating claims using Gemini.

Validates claims from episodes using Google Gemini API to identify bad claims
that should be flagged for review. Follows the same pattern as process_episodes.py
with guardrails to prevent over-deletion.

Usage:
    # Validate claims from all podcasts
    uv run python -m src.cli.validate_claims

    # Validate claims from specific podcasts
    uv run python -m src.cli.validate_claims --podcast-ids 1,2,3

    # Ensure specific podcasts have 20 validated episodes
    uv run python -m src.cli.validate_claims --podcast-ids 9 --target 20

    # Dry run (report what would be flagged without updating DB)
    uv run python -m src.cli.validate_claims --dry-run

Examples:
    # Dry run to see what would be flagged
    uv run python -m src.cli.validate_claims --podcast-ids 1 --target 5 --dry-run

    # Actually flag bad claims in production
    uv run python -m src.cli.validate_claims --podcast-ids 1 --target 5
"""

import argparse
import asyncio
import sys
import time
from typing import List, Dict, Tuple, Optional, cast
from dataclasses import dataclass, field
from collections import defaultdict

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

from src.database.connection import get_db_session
from src.database.claim_repository import ClaimRepository
from src.database.models import PodcastEpisode, Claim
from src.cli.episode_query import EpisodeQueryService
from src.infrastructure.gemini_service import (
    GeminiService,
    ClaimValidationInput,
    ClaimValidationResult
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger, get_shared_console

logger = get_logger(__name__)
console = get_shared_console()


@dataclass
class ValidationStats:
    """Overall validation statistics."""
    total_episodes: int = 0
    episodes_processed: int = 0
    episodes_skipped: int = 0
    total_claims_checked: int = 0
    total_claims_flagged: int = 0
    total_time: float = 0.0
    skipped_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    flagged_claim_samples: List[Tuple[int, str]] = field(default_factory=list)  # (claim_id, text)
    errors: List[Tuple[int, str, str]] = field(default_factory=list)  # (episode_id, name, error)

    @property
    def avg_time_per_episode(self) -> float:
        """Average processing time per episode."""
        if self.episodes_processed == 0:
            return 0.0
        return self.total_time / self.episodes_processed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate claims using Gemini API to flag bad claims",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate claims from all podcasts
  %(prog)s

  # Validate claims from specific podcasts
  %(prog)s --podcast-ids 1,2,3

  # Ensure specific podcasts have 20 validated episodes
  %(prog)s --podcast-ids 1,2,3 --target 20

  # Dry run (show what would be flagged)
  %(prog)s --dry-run

  # Continue on errors
  %(prog)s --continue-on-error
        """
    )

    parser.add_argument(
        "--podcast-ids",
        type=str,
        default=None,
        help="Validate claims from specific podcast IDs, comma-separated (e.g., '1,2,3') (default: all podcasts)"
    )

    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="Target number of validated episodes per podcast (default: 0 = validate all). "
             "Requires --podcast-ids. If podcast already has target validated episodes, skips it."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be flagged without actually updating the database"
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing if an episode fails (default: stop on error)"
    )


    args = parser.parse_args()

    # Parse comma-separated podcast IDs
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
    dry_run: bool
):
    """Display validation summary before starting."""
    console.print()

    # Create summary table
    table = Table(title="Validation Summary", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    if podcast_ids:
        table.add_row("Podcast IDs", ", ".join(str(pid) for pid in podcast_ids))
    else:
        table.add_row("Podcast IDs", "All podcasts")

    table.add_row("Episodes to validate", str(len(episodes)))
    table.add_row("Min unverified claims", str(settings.min_claims_for_validation))
    table.add_row("Batch size", f"{settings.gemini_validation_batch_size} claims/call")
    table.add_row("Gemini model", settings.gemini_model)
    table.add_row("Validation scope", "Unverified claims only")
    table.add_row("Mode", "[yellow]DRY RUN[/yellow]" if dry_run else "[green]LIVE[/green]")

    console.print(table)
    console.print()

    if len(episodes) == 0:
        console.print("[yellow]⚠  No episodes to validate![/yellow]")
        return False

    # Show first few episodes
    if len(episodes) > 0:
        console.print("[bold]Episodes to validate (first 5):[/bold]")
        for i, episode in enumerate(episodes[:5], 1):
            date_str = episode.published_at.strftime("%Y-%m-%d") if episode.published_at is not None else "No date"
            console.print(f"  {i}. [cyan]Episode {episode.id}[/cyan]: {episode.name} ({date_str})")

        if len(episodes) > 5:
            console.print(f"  ... and {len(episodes) - 5} more")
        console.print()

    return True


def display_episode_result(
    episode: PodcastEpisode,
    claims_checked: int,
    claims_flagged: int,
    skipped: bool,
    skip_reason: Optional[str],
    index: int,
    total: int
):
    """Display results for a single validated episode."""
    console.print()

    if skipped:
        console.print(
            f"[bold yellow]Episode {index}/{total} SKIPPED[/bold yellow]: "
            f"{episode.name} (ID: {episode.id})"
        )
        console.print(f"[yellow]Reason: {skip_reason}[/yellow]")
    else:
        console.print(
            f"[bold cyan]Episode {index}/{total}[/bold cyan]: "
            f"{episode.name} (ID: {episode.id})"
        )

        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="dim", width=24)
        table.add_column("Value")

        table.add_row("Claims checked", str(claims_checked))

        if claims_flagged > 0:
            table.add_row(
                "Claims flagged",
                f"[bold red]{claims_flagged}[/bold red] ({claims_flagged/claims_checked*100:.1f}%)"
            )
        else:
            table.add_row("Claims flagged", "[green]0[/green]")

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


def display_final_stats(stats: ValidationStats, dry_run: bool):
    """Display final validation statistics."""
    console.print()
    console.rule("[bold]VALIDATION COMPLETE[/bold]")
    console.print()

    # Summary table
    table = Table(title="Final Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")

    # Episodes
    table.add_row("Episodes processed", f"{stats.episodes_processed}/{stats.total_episodes}")
    if stats.episodes_skipped > 0:
        table.add_row("Episodes skipped", str(stats.episodes_skipped), style="yellow")

    # Claims
    table.add_row("Claims checked", str(stats.total_claims_checked))

    if stats.total_claims_flagged > 0:
        flag_rate = (stats.total_claims_flagged / stats.total_claims_checked * 100) if stats.total_claims_checked > 0 else 0
        table.add_row(
            "Claims flagged",
            f"{stats.total_claims_flagged} ({flag_rate:.1f}%)",
            style="red" if not dry_run else "yellow"
        )
    else:
        table.add_row("Claims flagged", "0", style="green")

    # Timing
    table.add_row("Total time", f"{stats.total_time / 60:.1f} minutes")
    table.add_row("Average per episode", f"{stats.avg_time_per_episode:.1f}s")

    console.print(table)

    # Skipped reasons breakdown
    if stats.skipped_reasons:
        console.print()
        console.print("[bold yellow]Episodes Skipped:[/bold yellow]")
        for reason, count in stats.skipped_reasons.items():
            console.print(f"  • {reason}: {count} episodes")

    # Flagged samples
    if stats.flagged_claim_samples:
        console.print()
        table = Table(title="Flagged Claim Samples (first 10)", box=box.ROUNDED)
        table.add_column("Claim ID", style="red", width=12)
        table.add_column("Text", style="dim", no_wrap=False)

        for claim_id, text in stats.flagged_claim_samples[:10]:
            # Truncate long text
            display_text = text if len(text) <= 120 else text[:117] + "..."
            table.add_row(str(claim_id), display_text)

        console.print(table)

    # Errors
    if stats.errors:
        console.print()
        console.print("[bold red]Failed Episodes:[/bold red]")
        for episode_id, name, error in stats.errors:
            console.print(f"  • [red]Episode {episode_id}[/red]: {name}")
            console.print(f"    {error}")

    # Dry run warning
    if dry_run:
        console.print()
        console.print(
            "[bold yellow]⚠  DRY RUN MODE - No changes were made to the database[/bold yellow]"
        )
        console.print("[yellow]Run without --dry-run to actually flag claims[/yellow]")

    console.print()


async def validate_episode(
    episode: PodcastEpisode,
    gemini_service: GeminiService,
    claim_repo: ClaimRepository,
    validation_prompt: str,
    dry_run: bool
) -> Tuple[int, int, bool, Optional[str]]:
    """
    Validate claims for a single episode.

    Returns:
        Tuple of (claims_checked, claims_flagged, skipped, skip_reason)
    """
    episode_id = cast(int, episode.id)

    # Get unverified claim count (only_unverified=True by default)
    claim_counts = claim_repo.get_episode_claim_counts([episode_id], only_unverified=True)
    claim_count = claim_counts.get(episode_id, 0)

    # Guardrail: Skip episodes with too few unverified claims
    if claim_count < settings.min_claims_for_validation:
        return (0, 0, True, f"Only {claim_count} unverified claims (min {settings.min_claims_for_validation})")

    # Get unverified claims for this episode (include_verified=False by default)
    claims_by_episode = claim_repo.get_claims_by_episodes(
        [episode_id],
        include_flagged=False,
        include_verified=False
    )
    claims = claims_by_episode.get(episode_id, [])

    if not claims:
        return (0, 0, True, "No unverified, unflagged claims to validate")

    logger.info(f"Validating {len(claims)} unverified claims for episode {episode_id}")

    # Convert to validation input
    validation_inputs = [
        ClaimValidationInput(
            claim_id=cast(int, claim.id),
            claim_text=claim.claim_text,
            confidence=claim.confidence,
            episode_id=episode_id
        )
        for claim in claims
    ]

    # Process in batches
    all_results: List[ClaimValidationResult] = []
    batch_size = settings.gemini_validation_batch_size

    for i in range(0, len(validation_inputs), batch_size):
        batch = validation_inputs[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} claims)")

        results = await gemini_service.validate_claims_batch(batch, validation_prompt)
        all_results.extend(results)

    # Filter invalid claims
    invalid_claim_ids = [
        result.claim_id
        for result in all_results
        if not result.is_valid
    ]

    # Flag bad claims
    claims_flagged = 0
    if invalid_claim_ids:
        claims_flagged = claim_repo.flag_claims(invalid_claim_ids, dry_run=dry_run)

    # Mark all validated claims as verified (both good and bad)
    all_claim_ids = [inp.claim_id for inp in validation_inputs]
    claim_repo.mark_claims_verified(all_claim_ids, dry_run=dry_run)

    return (len(claims), claims_flagged, False, None)


async def validate_episodes(
    episodes: List[PodcastEpisode],
    validation_prompt: str,
    dry_run: bool,
    continue_on_error: bool
) -> ValidationStats:
    """
    Validate claims for multiple episodes.

    Args:
        episodes: List of episodes to validate
        validation_prompt: Validation prompt for Gemini
        dry_run: If True, don't actually flag claims
        continue_on_error: If True, continue on errors

    Returns:
        ValidationStats with results
    """
    stats = ValidationStats(total_episodes=len(episodes))

    # Initialize services
    gemini_service = GeminiService()
    session = get_db_session()
    claim_repo = ClaimRepository(session)

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
            "[cyan]Validating episodes...",
            total=len(episodes)
        )

        start_time = time.time()

        for i, episode in enumerate(episodes, 1):
            try:
                # Update progress
                progress.update(
                    task,
                    description=f"[cyan]Validating episode {i}/{len(episodes)}: {episode.name[:40]}..."
                )

                episode_id = cast(int, episode.id)
                logger.info(f"Validating episode {episode_id}: {episode.name}")

                # Validate episode
                claims_checked, claims_flagged, skipped, skip_reason = await validate_episode(
                    episode,
                    gemini_service,
                    claim_repo,
                    validation_prompt,
                    dry_run
                )

                # Update stats
                if skipped:
                    stats.episodes_skipped += 1
                    if skip_reason:
                        stats.skipped_reasons[skip_reason] += 1
                else:
                    stats.episodes_processed += 1
                    stats.total_claims_checked += claims_checked
                    stats.total_claims_flagged += claims_flagged

                    # Get flagged claim samples for reporting
                    if claims_flagged > 0 and len(stats.flagged_claim_samples) < 50:
                        claims_by_episode = claim_repo.get_claims_by_episodes(
                            [episode_id],
                            include_flagged=True,
                            include_verified=True
                        )
                        for claim in claims_by_episode.get(episode_id, []):
                            if claim.is_flagged:
                                # Add to samples (keep first 50)
                                if len(stats.flagged_claim_samples) < 50:
                                    stats.flagged_claim_samples.append((
                                        cast(int, claim.id),
                                        claim.claim_text
                                    ))

                # Commit after each episode
                if not dry_run:
                    session.commit()

                # Display results
                display_episode_result(
                    episode, claims_checked, claims_flagged, skipped, skip_reason, i, len(episodes)
                )

            except Exception as e:
                episode_id = cast(int, episode.id)
                episode_name = cast(str, episode.name)
                logger.error(
                    f"Failed to validate episode {episode_id}: {e}",
                    exc_info=True
                )

                stats.errors.append((episode_id, episode_name, str(e)))
                session.rollback()

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

    session.close()
    return stats


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Display header
    console.rule("[bold blue]Claim Validation with Gemini[/bold blue]")
    console.print()

    # Load validation prompt
    try:
        gemini_service = GeminiService()
        validation_prompt = gemini_service.get_validation_prompt()
        console.print(f"[green]✓ Loaded validation prompt ({len(validation_prompt)} chars)[/green]")
    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        console.print()
        console.print("[yellow]Please update VALIDATION_PROMPT in src/config/validation_prompt.py[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error initializing Gemini: {e}[/bold red]")
        return 1

    # Initialize query service
    query_service = EpisodeQueryService()

    # Get episodes to validate (episodes with unverified claims)
    console.print("[cyan]Querying episodes...[/cyan]")

    episodes = query_service.get_episodes_to_validate(
        podcast_ids=args.podcast_ids,
        target=args.target
    )

    if len(episodes) > 0:
        console.print()
        console.print(f"[cyan]Retrieved {len(episodes)} episode(s) to validate[/cyan]")
        console.print()

    # Display summary
    should_continue = display_summary(episodes, args.podcast_ids, args.dry_run)
    if not should_continue:
        return 0

    # Confirm with user (unless dry run)
    if len(episodes) > 0:
        if args.dry_run:
            console.print("[yellow]Dry run mode - press Enter to continue...[/yellow]", end="")
        else:
            console.print("[yellow]⚠  This will FLAG claims in the database. Press Ctrl+C to cancel, or Enter to continue...[/yellow]", end="")

        try:
            input()
        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Cancelled by user[/yellow]")
            return 1

    # Validate episodes
    stats = await validate_episodes(
        episodes,
        validation_prompt,
        args.dry_run,
        args.continue_on_error
    )

    # Display final statistics
    display_final_stats(stats, args.dry_run)

    # Return exit code
    return 0 if len(stats.errors) == 0 else 1


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
