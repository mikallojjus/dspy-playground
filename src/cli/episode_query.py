"""
Episode query service for CLI processing.

Handles database queries to find episodes to process, with filtering
for already-processed episodes and podcast selection.

Usage:
    from src.cli.episode_query import EpisodeQueryService

    service = EpisodeQueryService()
    episodes = service.get_episodes_to_process(
        podcast_id=123,
        limit=10,
        force=False
    )
"""

from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from src.database.models import PodcastEpisode, Claim
from src.database.connection import get_db_session
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class EpisodeQueryService:
    """
    Service for querying episodes to process.

    Features:
    - Filter by podcast_id (or all podcasts)
    - Skip already-processed episodes (unless force=True)
    - Order by newest first (published_at DESC)
    - Limit number of episodes
    - Check processing status

    Example:
        ```python
        service = EpisodeQueryService()

        # Get 10 newest unprocessed episodes from podcast 123
        episodes = service.get_episodes_to_process(
            podcast_id=123,
            limit=10,
            force=False
        )

        # Check if specific episode is processed
        is_processed = service.is_episode_processed(episode_id=456)
        ```
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the query service.

        Args:
            db_session: Optional database session. If not provided, creates new session.
        """
        self.session = db_session or get_db_session()
        self._owns_session = db_session is None

    def __del__(self):
        """Close session if we own it."""
        if self._owns_session and hasattr(self, "session"):
            self.session.close()

    def get_episode_by_id(self, episode_id: int) -> Optional[PodcastEpisode]:
        """
        Get a specific episode by ID.

        Args:
            episode_id: Episode ID to fetch

        Returns:
            PodcastEpisode object or None if not found

        Raises:
            ValueError: If episode doesn't have a transcript

        Example:
            ```python
            episode = service.get_episode_by_id(123)
            if episode:
                print(f"Found: {episode.name}")
            ```
        """
        logger.info(f"Fetching episode {episode_id}")

        episode = self.session.query(PodcastEpisode).filter(
            PodcastEpisode.id == episode_id
        ).first()

        if episode is None:
            logger.error(f"Episode {episode_id} not found")
            return None

        if episode.transcript is None:
            logger.error(f"Episode {episode_id} has no transcript")
            raise ValueError(f"Episode {episode_id} has no transcript")

        logger.info(f"Found episode {episode_id}: {episode.name}")
        return episode

    def get_episodes_to_process(
        self, podcast_id: Optional[int] = None, limit: int = 0, force: bool = False
    ) -> List[PodcastEpisode]:
        """
        Get episodes to process based on filters.

        Query logic:
        1. Episodes must have transcript (transcript IS NOT NULL)
        2. Filter by podcast_id if provided
        3. Skip already-processed unless force=True
        4. Order by published_at DESC (newest first), NULL dates last
        5. Apply limit if > 0

        Args:
            podcast_id: Optional podcast ID to filter by (None = all podcasts)
            limit: Maximum number of episodes (0 = no limit)
            force: If True, include already-processed episodes

        Returns:
            List of PodcastEpisode objects to process

        Example:
            ```python
            # Get all unprocessed episodes from all podcasts
            episodes = service.get_episodes_to_process()

            # Get 5 newest episodes from podcast 123
            episodes = service.get_episodes_to_process(podcast_id=123, limit=5)

            # Reprocess all episodes (force=True)
            episodes = service.get_episodes_to_process(force=True)
            ```
        """
        logger.info(
            f"Querying episodes: podcast_id={podcast_id}, "
            f"limit={limit}, force={force}"
        )

        # Base query: episodes with transcripts
        query = self.session.query(PodcastEpisode).filter(
            PodcastEpisode.transcript.isnot(None)
        )

        # Filter by podcast_id if provided
        if podcast_id is not None:
            query = query.filter(PodcastEpisode.podcast_id == podcast_id)
            logger.debug(f"Filtering by podcast_id={podcast_id}")

        # Skip already-processed episodes unless force=True
        if not force:
            # LEFT JOIN to find episodes without claims
            query = query.outerjoin(
                Claim, Claim.episode_id == PodcastEpisode.id
            ).filter(
                Claim.id.is_(None)
            )  # No claims = not processed
            logger.debug("Filtering to unprocessed episodes only (force=False)")

        # Order by newest first (published_at DESC), NULL dates last
        query = query.order_by(PodcastEpisode.published_at.desc().nulls_last())

        # Apply limit if specified
        if limit > 0:
            query = query.limit(limit)
            logger.debug(f"Applying limit={limit}")

        # Execute query
        episodes = query.all()

        logger.info(f"Found {len(episodes)} episodes to process")

        return episodes

    def is_episode_processed(self, episode_id: int) -> bool:
        """
        Check if an episode has already been processed.

        An episode is considered processed if it has any claims in the database.

        Args:
            episode_id: Episode ID to check

        Returns:
            True if episode has claims, False otherwise

        Example:
            ```python
            if service.is_episode_processed(123):
                print("Episode 123 already processed")
            else:
                print("Episode 123 needs processing")
            ```
        """
        count = (
            self.session.query(func.count(Claim.id))
            .filter(Claim.episode_id == episode_id)
            .scalar()
        )

        is_processed = count > 0

        logger.debug(
            f"Episode {episode_id}: "
            f"{'processed' if is_processed else 'unprocessed'} "
            f"({count} claims)"
        )

        return is_processed

    def get_processing_stats(self, podcast_id: Optional[int] = None) -> Dict[str, int]:
        """
        Get summary statistics for episodes.

        Args:
            podcast_id: Optional podcast ID to filter by

        Returns:
            Dict with statistics:
            - total_episodes: Total episodes with transcripts
            - processed: Episodes with claims
            - unprocessed: Episodes without claims
            - total_claims: Total claims across all episodes

        Example:
            ```python
            stats = service.get_processing_stats(podcast_id=123)
            print(f"Podcast 123: {stats['unprocessed']} episodes to process")
            ```
        """
        # Base filter
        episode_filter = PodcastEpisode.transcript.isnot(None)
        if podcast_id is not None:
            episode_filter = and_(
                episode_filter, PodcastEpisode.podcast_id == podcast_id
            )

        # Total episodes with transcripts
        total_episodes = (
            self.session.query(func.count(PodcastEpisode.id))
            .filter(episode_filter)
            .scalar()
        )

        # Processed episodes (have claims)
        processed = (
            self.session.query(func.count(func.distinct(Claim.episode_id)))
            .join(PodcastEpisode, Claim.episode_id == PodcastEpisode.id)
            .filter(episode_filter)
            .scalar()
        )

        # Total claims
        total_claims = (
            self.session.query(func.count(Claim.id))
            .join(PodcastEpisode, Claim.episode_id == PodcastEpisode.id)
            .filter(episode_filter)
            .scalar()
        )

        unprocessed = total_episodes - processed

        stats = {
            "total_episodes": total_episodes,
            "processed": processed,
            "unprocessed": unprocessed,
            "total_claims": total_claims,
        }

        logger.info(
            f"Processing stats (podcast_id={podcast_id}): "
            f"{unprocessed}/{total_episodes} episodes unprocessed, "
            f"{total_claims} total claims"
        )

        return stats

    def get_podcast_name(self, podcast_id: int) -> Optional[str]:
        """
        Get podcast name from first episode.

        Note: Podcast model doesn't exist, so we infer name from episodes.

        Args:
            podcast_id: Podcast ID

        Returns:
            Podcast name from first episode, or None if not found
        """
        episode = (
            self.session.query(PodcastEpisode)
            .filter(PodcastEpisode.podcast_id == podcast_id)
            .first()
        )

        if episode:
            # We can extract podcast name from episode metadata
            # For now, just return podcast_id as string
            return f"Podcast {podcast_id}"

        return None
